"""
Semantic "near" regions and goal poses built from perceived objects.

The evaluation annulus geometry in :mod:`qmapnav.reasoning.semantic_geometry` is
reused verbatim: a perceived instance is adapted into the same box shape by
:func:`qmapnav.mapping.perceived_geometry.perceived_box` rather than the
polygon logic being duplicated for perceived maps.

A goal pose is never the object centre. It lies in the annulus between the
inflated footprint and the semantic near radius, which keeps the robot clear
of the object while still satisfying "near" in the instruction's sense.
"""

from dataclasses import dataclass
from math import atan2
from math import cos
from math import hypot
from math import isfinite
from math import pi
from math import sin

from qmapnav.common import ObjectInstance
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.grid_planning import planned_distance
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.mapping.perceived_geometry import PerceivedBox
from qmapnav.reasoning.semantic_geometry import make_near_region
from qmapnav.reasoning.semantic_geometry import object_footprint
from qmapnav.reasoning.semantic_geometry import SemanticRegion


@dataclass(frozen=True)
class NearRegionConfig:
    """Metric definition of "near" for a perceived object."""

    near_min_clearance_m: float = 0.6
    near_max_distance_m: float = 1.5
    robot_clearance_m: float = 0.35
    low_orientation_inflation_m: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            'near_min_clearance_m',
            'near_max_distance_m',
            'robot_clearance_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not isfinite(self.low_orientation_inflation_m) or (
            self.low_orientation_inflation_m < 0.0
        ):
            raise ValueError('low_orientation_inflation_m must be >= 0')
        if self.near_max_distance_m <= self.near_min_clearance_m:
            raise ValueError('near_max_distance_m must exceed the clearance')


@dataclass(frozen=True)
class GoalPoseCandidate:
    """One traversable pose inside a semantic region, with its rationale."""

    pose_xy_yaw: tuple[float, float, float]
    approach_cost_m: float
    transition_cost_m: float | None
    has_line_of_sight: bool
    score: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'pose_xy_yaw': list(self.pose_xy_yaw),
            'approach_cost_m': self.approach_cost_m,
            'transition_cost_m': self.transition_cost_m,
            'has_line_of_sight': self.has_line_of_sight,
            'score': self.score,
        }


@dataclass(frozen=True)
class GoalPoseScoringConfig:
    """Weights trading approach cost against onward reachability."""

    approach_weight: float = 1.0
    transition_weight: float = 0.6
    line_of_sight_bonus: float = 0.35
    heading_weight: float = 0.15
    normalization_m: float = 8.0

    def __post_init__(self) -> None:
        if not isfinite(self.normalization_m) or self.normalization_m <= 0.0:
            raise ValueError('normalization_m must be finite and positive')
        for name in (
            'approach_weight',
            'transition_weight',
            'line_of_sight_bonus',
            'heading_weight',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')


def perceived_near_region(
    instance: ObjectInstance,
    *,
    config: NearRegionConfig | None = None,
    region_id: str | None = None,
) -> SemanticRegion:
    """
    Build the annular near region around one perceived instance.

    When orientation confidence is low the adapter already degrades to the
    axis-aligned box; the region is additionally inflated so a conservative
    goal pose never ends up inside the true object.
    """
    policy = config or NearRegionConfig()
    box = perceived_box(instance)
    inflation = (
        policy.low_orientation_inflation_m
        if box.used_axis_aligned_fallback
        else 0.0
    )
    inflated = perceived_box(instance, inflation_m=inflation)
    return make_near_region(
        inflated,
        min_distance=policy.near_min_clearance_m,
        max_distance=policy.near_max_distance_m,
        region_id=region_id or f'near_{inflated.object_id}',
    )


def semantic_region_satisfied(
    robot_pose_xy: tuple[float, float],
    region: SemanticRegion,
) -> bool:
    """
    Return whether the robot is semantically inside a region.

    This is deliberately independent of the executor's waypoint arrival
    radius: arriving within ``0.75 m`` of a commanded goal is not the same
    claim as standing in the object's near region.
    """
    if not isinstance(region, SemanticRegion):
        raise TypeError('region must be SemanticRegion')
    point = (float(robot_pose_xy[0]), float(robot_pose_xy[1]))
    if not all(isfinite(value) for value in point):
        raise ValueError('robot pose must be finite')
    return region.contains(point)


def sample_goal_poses(
    region: SemanticRegion,
    target_box: PerceivedBox,
    *,
    grid: OccupancyGrid2D,
    start_xy: tuple[float, float],
    config: NearRegionConfig | None = None,
    scoring: GoalPoseScoringConfig | None = None,
    next_stage_xy: tuple[float, float] | None = None,
    costs: dict | None = None,
    next_costs: dict | None = None,
) -> tuple[GoalPoseCandidate, ...]:
    """
    Score every traversable cell inside a near region as a goal pose.

    When ``next_stage_xy`` is supplied the onward travel cost is folded in, so
    stage A does not commit to a valid pose on the far side of its object that
    makes stage B expensive to reach.
    """
    policy = config or NearRegionConfig()
    weights = scoring or GoalPoseScoringConfig()
    approach_costs = (
        costs
        if costs is not None
        else cost_field(grid, start_xy, clearance=policy.robot_clearance_m)
    )
    onward_costs = next_costs
    if next_stage_xy is not None and onward_costs is None:
        onward_costs = cost_field(
            grid, next_stage_xy, clearance=policy.robot_clearance_m
        )
    centre = (target_box.centre_xyz[0], target_box.centre_xyz[1])
    min_x, min_y, max_x, max_y = region.polygon.bounds
    start_cell = grid.point_to_cell((min_x, min_y))
    stop_cell = grid.point_to_cell((max_x, max_y))
    candidates = []
    for x_index in range(
        max(0, start_cell[0]), min(grid.width, stop_cell[0] + 1)
    ):
        for y_index in range(
            max(0, start_cell[1]), min(grid.height, stop_cell[1] + 1)
        ):
            point = grid.cell_centre((x_index, y_index))
            if not region.contains(point):
                continue
            if not grid.is_free(
                point[0], point[1], clearance=policy.robot_clearance_m
            ):
                continue
            approach = approach_costs.get((x_index, y_index))
            if approach is None:
                continue
            transition = (
                None
                if onward_costs is None
                else onward_costs.get((x_index, y_index))
            )
            if onward_costs is not None and transition is None:
                # A pose that cannot reach the next stage is not usable.
                continue
            yaw = atan2(centre[1] - point[1], centre[0] - point[0])
            visible = grid.line_of_sight(point, centre)
            score = _goal_score(
                approach=approach,
                transition=transition,
                visible=visible,
                weights=weights,
            )
            candidates.append(
                GoalPoseCandidate(
                    pose_xy_yaw=(point[0], point[1], yaw),
                    approach_cost_m=approach,
                    transition_cost_m=transition,
                    has_line_of_sight=visible,
                    score=score,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-item.score, item.pose_xy_yaw),
        )
    )


def select_goal_pose(
    region: SemanticRegion,
    target_box: PerceivedBox,
    *,
    grid: OccupancyGrid2D,
    start_xy: tuple[float, float],
    config: NearRegionConfig | None = None,
    scoring: GoalPoseScoringConfig | None = None,
    next_stage_xy: tuple[float, float] | None = None,
    costs: dict | None = None,
    next_costs: dict | None = None,
) -> GoalPoseCandidate | None:
    """Return the best traversable goal pose, or ``None`` when none exists."""
    ranked = sample_goal_poses(
        region,
        target_box,
        grid=grid,
        start_xy=start_xy,
        config=config,
        scoring=scoring,
        next_stage_xy=next_stage_xy,
        costs=costs,
        next_costs=next_costs,
    )
    return ranked[0] if ranked else None


def region_excludes_footprint(
    region: SemanticRegion,
    instance: ObjectInstance,
) -> bool:
    """Return whether the object's own footprint is excluded from a region."""
    box = perceived_box(instance)
    footprint = object_footprint(box)
    return not region.contains(footprint.centre)


def approximate_region_area(
    region: SemanticRegion,
    *,
    grid: OccupancyGrid2D,
) -> float:
    """Return the usable metric area of a region on a given grid."""
    min_x, min_y, max_x, max_y = region.polygon.bounds
    start_cell = grid.point_to_cell((min_x, min_y))
    stop_cell = grid.point_to_cell((max_x, max_y))
    cell_area = grid.resolution ** 2
    total = 0.0
    for x_index in range(
        max(0, start_cell[0]), min(grid.width, stop_cell[0] + 1)
    ):
        for y_index in range(
            max(0, start_cell[1]), min(grid.height, stop_cell[1] + 1)
        ):
            if region.contains(grid.cell_centre((x_index, y_index))):
                total += cell_area
    return total


def _goal_score(
    *,
    approach: float,
    transition: float | None,
    visible: bool,
    weights: GoalPoseScoringConfig,
) -> float:
    normalizer = weights.normalization_m
    score = -weights.approach_weight * (approach / normalizer)
    if transition is not None:
        score -= weights.transition_weight * (transition / normalizer)
    if visible:
        score += weights.line_of_sight_bonus
    return score


def stage_transition_costs(
    grid: OccupancyGrid2D,
    point_xy: tuple[float, float],
    *,
    clearance: float = 0.35,
) -> dict:
    """Return the cost field used to price travel onward from a stage."""
    return cost_field(grid, point_xy, clearance=clearance)


def straight_line_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return planar Euclidean distance between two map-frame points."""
    return hypot(second[0] - first[0], second[1] - first[1])


def heading_toward(
    origin: tuple[float, float],
    target: tuple[float, float],
) -> float:
    """Return the yaw pointing from one map-frame point to another."""
    return atan2(target[1] - origin[1], target[0] - origin[0])


def pose_faces(
    pose_xy_yaw: tuple[float, float, float],
    target: tuple[float, float],
    *,
    tolerance_rad: float = pi / 6.0,
) -> bool:
    """Return whether a pose's heading points at a target within tolerance."""
    desired = heading_toward((pose_xy_yaw[0], pose_xy_yaw[1]), target)
    delta = atan2(
        sin(pose_xy_yaw[2] - desired), cos(pose_xy_yaw[2] - desired)
    )
    return abs(delta) <= tolerance_rad


def reachable_within(
    grid: OccupancyGrid2D,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    limit_m: float,
    clearance: float = 0.35,
) -> bool:
    """Return whether a goal is reachable within a travel limit."""
    distance = planned_distance(
        grid, start_xy, goal_xy, clearance=clearance
    )
    return distance is not None and distance <= limit_m


__all__ = [
    'GoalPoseCandidate',
    'GoalPoseScoringConfig',
    'NearRegionConfig',
    'approximate_region_area',
    'heading_toward',
    'perceived_near_region',
    'pose_faces',
    'reachable_within',
    'region_excludes_footprint',
    'sample_goal_poses',
    'select_goal_pose',
    'semantic_region_satisfied',
    'stage_transition_costs',
    'straight_line_distance',
]
