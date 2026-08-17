"""
Approach, rank, and remember searches of likely support surfaces.

Negative evidence is graded rather than absolute: failing to see a cup from a
distant, oblique, partially occluded view says far less than failing to see it
from a close frontal view. Only strong negatives retire a support from the
search order, which stops the robot re-inspecting one table forever without
ever claiming false certainty.
"""

from dataclasses import dataclass
from dataclasses import field
from math import atan2
from math import cos
from math import hypot
from math import isfinite
from math import pi
from math import sin

from qmapnav.exploration.small_object_mode import is_shelf_like
from qmapnav.exploration.small_object_mode import likely_supports
from qmapnav.exploration.viewpoint_candidate import ViewpointCandidate
from qmapnav.exploration.viewpoint_generation import accept_candidate_pose
from qmapnav.exploration.viewpoint_generation import CandidateGenerationOutcome
from qmapnav.exploration.viewpoint_generation import ViewpointGenerationConfig
from qmapnav.exploration.viewpoint_generation import VisitedViewpoint
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import PerceivedBox


NEGATIVE_STRENGTHS = ('none', 'weak', 'moderate', 'strong')

_STRENGTH_ORDER = {name: index for index, name in enumerate(NEGATIVE_STRENGTHS)}


@dataclass(frozen=True)
class SupportSearchRecord:
    """What has already been tried against one support instance."""

    support_instance_id: str
    viewpoints_tried: tuple[str, ...] = ()
    target_classes_searched: tuple[str, ...] = ()
    search_confidence: float = 0.0
    last_result: str = 'none'

    def __post_init__(self) -> None:
        if self.last_result not in NEGATIVE_STRENGTHS:
            expected = ', '.join(NEGATIVE_STRENGTHS)
            raise ValueError(f'last_result must be one of: {expected}')
        if not isfinite(self.search_confidence) or not (
            0.0 <= self.search_confidence <= 1.0
        ):
            raise ValueError('search_confidence must lie in [0, 1]')
        object.__setattr__(
            self, 'viewpoints_tried', tuple(self.viewpoints_tried)
        )
        object.__setattr__(
            self,
            'target_classes_searched',
            tuple(self.target_classes_searched),
        )

    @property
    def exhausted(self) -> bool:
        """Return whether this support has been convincingly ruled out."""
        return self.last_result == 'strong'

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'support_instance_id': self.support_instance_id,
            'viewpoints_tried': list(self.viewpoints_tried),
            'target_classes_searched': list(self.target_classes_searched),
            'search_confidence': self.search_confidence,
            'last_result': self.last_result,
        }


@dataclass
class SupportSearchHistory:
    """Episode-scoped memory of every support surface already inspected."""

    records: dict[str, SupportSearchRecord] = field(default_factory=dict)

    def record_for(self, support_instance_id: str) -> SupportSearchRecord:
        """Return the record for a support, creating an empty one if new."""
        return self.records.get(
            support_instance_id,
            SupportSearchRecord(support_instance_id=support_instance_id),
        )

    def is_exhausted(self, support_instance_id: str) -> bool:
        """Return whether a support was convincingly searched already."""
        return self.record_for(support_instance_id).exhausted

    def note_observation(
        self,
        support_instance_id: str,
        *,
        target_class: str,
        viewpoint_id: str,
        found: bool,
        visible_surface_fraction: float,
        distance_m: float,
        occluded: bool = False,
    ) -> SupportSearchRecord:
        """Fold one observation into graded negative evidence."""
        previous = self.record_for(support_instance_id)
        strength = 'none' if found else classify_negative_evidence(
            visible_surface_fraction=visible_surface_fraction,
            distance_m=distance_m,
            occluded=occluded,
        )
        merged = _stronger(previous.last_result, strength)
        confidence = max(
            previous.search_confidence,
            0.0 if found else _strength_confidence(merged),
        )
        record = SupportSearchRecord(
            support_instance_id=support_instance_id,
            viewpoints_tried=_append_unique(
                previous.viewpoints_tried, viewpoint_id
            ),
            target_classes_searched=_append_unique(
                previous.target_classes_searched, target_class
            ),
            search_confidence=confidence,
            last_result=merged,
        )
        self.records[support_instance_id] = record
        return record

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping of every record."""
        return {
            key: value.to_dict()
            for key, value in sorted(self.records.items())
        }


def classify_negative_evidence(
    *,
    visible_surface_fraction: float,
    distance_m: float,
    occluded: bool = False,
    close_distance_m: float = 2.0,
) -> str:
    """
    Grade a failed search rather than declaring the target absent.

    A strong negative requires a close, largely unoccluded view of most of the
    surface; anything less stays weak or moderate so the target may still be
    found elsewhere on the same support.
    """
    if not isfinite(visible_surface_fraction) or not (
        0.0 <= visible_surface_fraction <= 1.0
    ):
        raise ValueError('visible_surface_fraction must lie in [0, 1]')
    if not isfinite(distance_m) or distance_m < 0.0:
        raise ValueError('distance_m must be finite and non-negative')
    if occluded or visible_surface_fraction < 0.35:
        return 'weak'
    if visible_surface_fraction >= 0.80 and distance_m <= close_distance_m:
        return 'strong'
    if visible_surface_fraction >= 0.55:
        return 'moderate'
    return 'weak'


def rank_support_surfaces(
    target_class: str,
    supports: tuple[PerceivedBox, ...],
    *,
    current_pose_xy_yaw: tuple[float, float, float],
    history: SupportSearchHistory | None = None,
) -> tuple[PerceivedBox, ...]:
    """
    Order supports by semantic compatibility, then distance.

    Supports already ruled out by strong negative evidence are dropped so the
    robot does not re-inspect the same table from the same side.
    """
    memory = history or SupportSearchHistory()
    preferences = likely_supports(target_class)
    ranked = []
    for support in supports:
        if memory.is_exhausted(support.object_id):
            continue
        try:
            affinity = preferences.index(support.class_name)
        except ValueError:
            affinity = len(preferences) + 1
        distance = hypot(
            support.centre_xyz[0] - current_pose_xy_yaw[0],
            support.centre_xyz[1] - current_pose_xy_yaw[1],
        )
        ranked.append((affinity, distance, support.object_id, support))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(item[3] for item in ranked)


def generate_support_surface_viewpoints(
    support: PerceivedBox,
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    config: ViewpointGenerationConfig | None = None,
    visited: tuple[VisitedViewpoint, ...] = (),
    max_travel_m: float | None = None,
    costs: dict | None = None,
    side_count: int = 4,
) -> CandidateGenerationOutcome:
    """
    Sample poses facing a support's accessible sides at a close stand-off.

    Small targets need a shorter stand-off than large furniture, but not so
    short that the whole surface leaves the field of view, so the distance is
    taken from the support's own footprint plus a class-dependent margin.
    """
    policy = config or ViewpointGenerationConfig()
    if isinstance(side_count, bool) or side_count < 1:
        raise ValueError('side_count must be a positive integer')
    field_costs = (
        costs
        if costs is not None
        else cost_field(
            grid,
            current_pose_xy_yaw[:2],
            clearance=policy.robot_clearance_m,
        )
    )
    margin = (
        policy.shelf_standoff_m
        if is_shelf_like(support.class_name)
        else policy.tabletop_standoff_m
    )
    centre = (support.centre_xyz[0], support.centre_xyz[1])
    counts: dict[str, int] = {}
    accepted = []
    for index in range(side_count):
        angle = support.yaw + 2.0 * pi * index / side_count
        radius = _side_radius(support, angle) + margin
        point = (
            centre[0] + radius * cos(angle),
            centre[1] + radius * sin(angle),
        )
        yaw = atan2(centre[1] - point[1], centre[0] - point[0])
        pose = (point[0], point[1], yaw)
        travel = accept_candidate_pose(
            pose,
            centre,
            grid=grid,
            current_pose_xy_yaw=current_pose_xy_yaw,
            config=policy,
            visited=visited,
            costs=field_costs,
            max_travel_m=max_travel_m,
            counts=counts,
            focus_key=f'support_{support.object_id}',
            require_line_of_sight=policy.require_line_of_sight,
        )
        if travel is None:
            continue
        accepted.append(
            ViewpointCandidate(
                viewpoint_id=f'support_{support.object_id}_side_{index}',
                pose_xy_yaw=pose,
                source='support_surface',
                target_instance_ids=(support.object_id,),
                target_regions=(f'support_{support.object_id}',),
                travel_cost_m=travel,
                reasons=(
                    f'{support.class_name} side {index} at '
                    f'{margin:.2f} m stand-off',
                ),
            )
        )
    return CandidateGenerationOutcome(
        candidates=tuple(accepted[: policy.max_candidates_per_source]),
        rejected_counts=counts,
    )


def _side_radius(support: PerceivedBox, angle: float) -> float:
    length, width, _ = support.dimensions_xyz
    local_x = (cos(support.yaw), sin(support.yaw))
    local_y = (-sin(support.yaw), cos(support.yaw))
    direction = (cos(angle), sin(angle))
    return (
        abs(direction[0] * local_x[0] + direction[1] * local_x[1])
        * length / 2.0
        + abs(direction[0] * local_y[0] + direction[1] * local_y[1])
        * width / 2.0
    )


def _stronger(first: str, second: str) -> str:
    return (
        first
        if _STRENGTH_ORDER[first] >= _STRENGTH_ORDER[second]
        else second
    )


def _strength_confidence(strength: str) -> float:
    return {'none': 0.0, 'weak': 0.25, 'moderate': 0.6, 'strong': 0.9}[
        strength
    ]


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    return values if value in values else values + (value,)


__all__ = [
    'NEGATIVE_STRENGTHS',
    'SupportSearchHistory',
    'SupportSearchRecord',
    'classify_negative_evidence',
    'generate_support_surface_viewpoints',
    'rank_support_surfaces',
]
