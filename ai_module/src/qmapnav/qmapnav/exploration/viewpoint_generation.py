"""
Deterministic viewpoint candidate generation from four evidence sources.

Every source produces poses through the same safety funnel: reject occupied
or under-cleared poses, reject poses free space cannot actually reach, reject
poses that duplicate a view already taken, and reject poses outside the
remaining travel budget. Rejections are counted so a trace can explain why a
seemingly obvious viewpoint never appeared.
"""

from dataclasses import dataclass
from dataclasses import field
from math import atan2
from math import cos
from math import degrees
from math import hypot
from math import isfinite
from math import pi
from math import sin

from qmapnav.exploration.viewpoint_candidate import ViewpointCandidate
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.grid_planning import planned_distance
from qmapnav.mapping.occupancy_grid import CELL_UNKNOWN
from qmapnav.mapping.occupancy_grid import FrontierCluster
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.occupancy_grid import Point2D


REJECTION_REASONS = (
    'no_line_of_sight',
    'occupied',
    'out_of_budget',
    'redundant',
    'unreachable',
)


@dataclass(frozen=True)
class ViewpointGenerationConfig:
    """Sampling geometry and the shared safety funnel thresholds."""

    object_min_radius_m: float = 1.5
    object_max_radius_m: float = 3.0
    radius_steps: int = 2
    bearing_count: int = 8
    tabletop_standoff_m: float = 1.6
    shelf_standoff_m: float = 1.4
    lateral_offset_m: float = 0.8
    robot_clearance_m: float = 0.35
    novelty_min_translation_m: float = 0.5
    novelty_min_yaw_deg: float = 15.0
    max_candidates_per_source: int = 12
    require_line_of_sight: bool = True

    def __post_init__(self) -> None:
        positive = (
            'object_min_radius_m',
            'object_max_radius_m',
            'tabletop_standoff_m',
            'shelf_standoff_m',
            'lateral_offset_m',
            'robot_clearance_m',
            'novelty_min_translation_m',
            'novelty_min_yaw_deg',
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('radius_steps', 'bearing_count',
                     'max_candidates_per_source'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or (
                value < 1
            ):
                raise ValueError(f'{name} must be a positive integer')
        if self.object_min_radius_m >= self.object_max_radius_m:
            raise ValueError('object_min_radius_m must be below the maximum')


@dataclass(frozen=True)
class VisitedViewpoint:
    """One pose already observed, used to reject redundant repeats."""

    pose_xy_yaw: tuple[float, float, float]
    focus_key: str = ''


@dataclass
class CandidateGenerationOutcome:
    """Accepted candidates plus a count of each rejection reason."""

    candidates: tuple[ViewpointCandidate, ...] = ()
    rejected_counts: dict[str, int] = field(default_factory=dict)

    def merge(
        self, other: 'CandidateGenerationOutcome'
    ) -> 'CandidateGenerationOutcome':
        """Combine two source outcomes, preserving rejection counts."""
        counts = dict(self.rejected_counts)
        for key, value in other.rejected_counts.items():
            counts[key] = counts.get(key, 0) + value
        return CandidateGenerationOutcome(
            candidates=self.candidates + other.candidates,
            rejected_counts=counts,
        )


def angle_delta(first: float, second: float) -> float:
    """Return the signed smallest difference between two angles."""
    return atan2(sin(first - second), cos(first - second))


def is_novel(
    pose_xy_yaw: tuple[float, float, float],
    visited: tuple[VisitedViewpoint, ...],
    config: ViewpointGenerationConfig,
    *,
    focus_key: str = '',
) -> bool:
    """
    Return whether a pose offers a meaningful new baseline.

    A pose that is both nearly co-located and nearly co-oriented with an
    earlier observation of the same focus cannot add parallax, so it is
    rejected outright rather than merely penalized.
    """
    for previous in visited:
        if previous.focus_key and focus_key and (
            previous.focus_key != focus_key
        ):
            continue
        translation = hypot(
            pose_xy_yaw[0] - previous.pose_xy_yaw[0],
            pose_xy_yaw[1] - previous.pose_xy_yaw[1],
        )
        yaw_change = abs(
            degrees(angle_delta(pose_xy_yaw[2], previous.pose_xy_yaw[2]))
        )
        if (
            translation < config.novelty_min_translation_m
            and yaw_change < config.novelty_min_yaw_deg
        ):
            return False
    return True


def accept_candidate_pose(
    pose: tuple[float, float, float],
    focus_xy: Point2D,
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    config: ViewpointGenerationConfig,
    visited: tuple[VisitedViewpoint, ...],
    costs: dict,
    max_travel_m: float | None,
    counts: dict[str, int],
    focus_key: str,
    require_line_of_sight: bool,
) -> float | None:
    """
    Apply the shared safety funnel to one sampled pose.

    Returns the planned travel distance when the pose survives every check,
    or ``None`` after incrementing the matching rejection counter.
    """
    if not grid.is_free(pose[0], pose[1], clearance=config.robot_clearance_m):
        counts['occupied'] = counts.get('occupied', 0) + 1
        return None
    if require_line_of_sight and not grid.line_of_sight(
        (pose[0], pose[1]), focus_xy
    ):
        counts['no_line_of_sight'] = counts.get('no_line_of_sight', 0) + 1
        return None
    if not is_novel(pose, visited, config, focus_key=focus_key):
        counts['redundant'] = counts.get('redundant', 0) + 1
        return None
    distance = planned_distance(
        grid,
        current_pose_xy_yaw[:2],
        (pose[0], pose[1]),
        clearance=config.robot_clearance_m,
        costs=costs,
    )
    if distance is None:
        counts['unreachable'] = counts.get('unreachable', 0) + 1
        return None
    if max_travel_m is not None and distance > max_travel_m:
        counts['out_of_budget'] = counts.get('out_of_budget', 0) + 1
        return None
    return distance


def _build_costs(
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    config: ViewpointGenerationConfig,
) -> dict:
    return cost_field(
        grid,
        current_pose_xy_yaw[:2],
        clearance=config.robot_clearance_m,
    )


def generate_object_annulus_viewpoints(
    focus_xy: Point2D,
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    prefix: str,
    config: ViewpointGenerationConfig | None = None,
    visited: tuple[VisitedViewpoint, ...] = (),
    max_travel_m: float | None = None,
    target_instance_ids: tuple[str, ...] = (),
    costs: dict | None = None,
) -> CandidateGenerationOutcome:
    """Sample safe poses on an annulus around one unresolved object."""
    policy = config or ViewpointGenerationConfig()
    field_costs = (
        costs
        if costs is not None
        else _build_costs(grid, current_pose_xy_yaw, policy)
    )
    counts: dict[str, int] = {}
    accepted = []
    steps = policy.radius_steps
    for radius_index in range(steps):
        ratio = 0.0 if steps == 1 else radius_index / (steps - 1)
        radius = policy.object_min_radius_m + ratio * (
            policy.object_max_radius_m - policy.object_min_radius_m
        )
        for bearing_index in range(policy.bearing_count):
            angle = 2.0 * pi * bearing_index / policy.bearing_count
            point = (
                focus_xy[0] + radius * cos(angle),
                focus_xy[1] + radius * sin(angle),
            )
            yaw = atan2(focus_xy[1] - point[1], focus_xy[0] - point[0])
            pose = (point[0], point[1], yaw)
            distance = accept_candidate_pose(
                pose,
                focus_xy,
                grid=grid,
                current_pose_xy_yaw=current_pose_xy_yaw,
                config=policy,
                visited=visited,
                costs=field_costs,
                max_travel_m=max_travel_m,
                counts=counts,
                focus_key=prefix,
                require_line_of_sight=policy.require_line_of_sight,
            )
            if distance is None:
                continue
            accepted.append(
                ViewpointCandidate(
                    viewpoint_id=(
                        f'{prefix}_annulus_{radius_index}_{bearing_index}'
                    ),
                    pose_xy_yaw=pose,
                    source='object_annulus',
                    target_instance_ids=target_instance_ids,
                    travel_cost_m=distance,
                    reasons=(f'annulus radius {radius:.2f} m',),
                )
            )
    return CandidateGenerationOutcome(
        candidates=tuple(accepted[: policy.max_candidates_per_source]),
        rejected_counts=counts,
    )


def generate_occluder_offset_viewpoints(
    focus_xy: Point2D,
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    prefix: str,
    config: ViewpointGenerationConfig | None = None,
    visited: tuple[VisitedViewpoint, ...] = (),
    max_travel_m: float | None = None,
    target_instance_ids: tuple[str, ...] = (),
    costs: dict | None = None,
) -> CandidateGenerationOutcome:
    """Offset laterally from the current line of sight to defeat occlusion."""
    policy = config or ViewpointGenerationConfig()
    field_costs = (
        costs
        if costs is not None
        else _build_costs(grid, current_pose_xy_yaw, policy)
    )
    counts: dict[str, int] = {}
    connector = (
        current_pose_xy_yaw[0] - focus_xy[0],
        current_pose_xy_yaw[1] - focus_xy[1],
    )
    distance = hypot(connector[0], connector[1])
    if distance < 1e-6:
        radial = (1.0, 0.0)
        distance = 1.0
    else:
        radial = (connector[0] / distance, connector[1] / distance)
    lateral = (-radial[1], radial[0])
    standoff = min(
        max(policy.object_min_radius_m, distance),
        policy.object_max_radius_m,
    )
    accepted = []
    for name, sign in (('left', 1.0), ('right', -1.0)):
        point = (
            focus_xy[0]
            + radial[0] * standoff
            + lateral[0] * sign * policy.lateral_offset_m,
            focus_xy[1]
            + radial[1] * standoff
            + lateral[1] * sign * policy.lateral_offset_m,
        )
        yaw = atan2(focus_xy[1] - point[1], focus_xy[0] - point[0])
        pose = (point[0], point[1], yaw)
        travel = accept_candidate_pose(
            pose,
            focus_xy,
            grid=grid,
            current_pose_xy_yaw=current_pose_xy_yaw,
            config=policy,
            visited=visited,
            costs=field_costs,
            max_travel_m=max_travel_m,
            counts=counts,
            focus_key=prefix,
            require_line_of_sight=policy.require_line_of_sight,
        )
        if travel is None:
            continue
        accepted.append(
            ViewpointCandidate(
                viewpoint_id=f'{prefix}_offset_{name}',
                pose_xy_yaw=pose,
                source='occluder_offset',
                target_instance_ids=target_instance_ids,
                travel_cost_m=travel,
                reasons=(f'lateral {name} baseline change',),
            )
        )
    return CandidateGenerationOutcome(
        candidates=tuple(accepted[: policy.max_candidates_per_source]),
        rejected_counts=counts,
    )


def generate_frontier_viewpoints(
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    config: ViewpointGenerationConfig | None = None,
    visited: tuple[VisitedViewpoint, ...] = (),
    max_travel_m: float | None = None,
    minimum_cluster_cells: int = 3,
    costs: dict | None = None,
) -> CandidateGenerationOutcome:
    """Stand at meaningful explored/unexplored boundaries facing the unknown."""
    policy = config or ViewpointGenerationConfig()
    field_costs = (
        costs
        if costs is not None
        else _build_costs(grid, current_pose_xy_yaw, policy)
    )
    counts: dict[str, int] = {}
    accepted = []
    clusters = grid.frontier_clusters(minimum_cells=minimum_cluster_cells)
    for cluster in clusters:
        pose = _frontier_pose(grid, cluster)
        if pose is None:
            counts['occupied'] = counts.get('occupied', 0) + 1
            continue
        travel = accept_candidate_pose(
            pose,
            (pose[0], pose[1]),
            grid=grid,
            current_pose_xy_yaw=current_pose_xy_yaw,
            config=policy,
            visited=visited,
            costs=field_costs,
            max_travel_m=max_travel_m,
            counts=counts,
            focus_key=cluster.cluster_id,
            require_line_of_sight=False,
        )
        if travel is None:
            continue
        accepted.append(
            ViewpointCandidate(
                viewpoint_id=f'{cluster.cluster_id}_view',
                pose_xy_yaw=pose,
                source='frontier',
                target_regions=(cluster.cluster_id,),
                travel_cost_m=travel,
                reasons=(f'frontier cluster of {cluster.size} cells',),
            )
        )
    return CandidateGenerationOutcome(
        candidates=tuple(accepted[: policy.max_candidates_per_source]),
        rejected_counts=counts,
    )


def _frontier_pose(
    grid: OccupancyGrid2D,
    cluster: FrontierCluster,
) -> tuple[float, float, float] | None:
    best = None
    for cell in cluster.cells:
        centre = grid.cell_centre(cell)
        offset = hypot(
            centre[0] - cluster.centroid_xy[0],
            centre[1] - cluster.centroid_xy[1],
        )
        if best is None or offset < best[0]:
            best = (offset, centre, cell)
    if best is None:
        return None
    _, centre, cell = best
    direction = _unknown_direction(grid, cell)
    if direction is None:
        return None
    return (centre[0], centre[1], direction)


def _unknown_direction(grid: OccupancyGrid2D, cell) -> float | None:
    total_x = 0.0
    total_y = 0.0
    for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = (cell[0] + delta_x, cell[1] + delta_y)
        if grid.state(neighbour) == CELL_UNKNOWN:
            total_x += delta_x
            total_y += delta_y
    if hypot(total_x, total_y) < 1e-9:
        return None
    return atan2(total_y, total_x)


__all__ = [
    'REJECTION_REASONS',
    'CandidateGenerationOutcome',
    'ViewpointGenerationConfig',
    'VisitedViewpoint',
    'accept_candidate_pose',
    'angle_delta',
    'generate_frontier_viewpoints',
    'generate_object_annulus_viewpoints',
    'generate_occluder_offset_viewpoints',
    'is_novel',
]
