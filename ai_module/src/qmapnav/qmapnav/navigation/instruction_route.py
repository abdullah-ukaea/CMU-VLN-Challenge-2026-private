"""
The first perceived semantic route: go near A, then stop near B.

Stages are grounded through the reasoning perceived resolver rather than oracle
coordinates, and each stage's goal is a pose inside that object's semantic
near region. instruction supports exactly two ordered stages; gates, forbidden
polygons, and augmented-state planning arrive on Days 13-15.
"""

from dataclasses import dataclass
from math import isfinite

from qmapnav.common import ObjectInstance
from qmapnav.common import TaskSpecification
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.grid_planning import shortest_path
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.navigation.semantic_regions import GoalPoseScoringConfig
from qmapnav.navigation.semantic_regions import NearRegionConfig
from qmapnav.navigation.semantic_regions import perceived_near_region
from qmapnav.navigation.semantic_regions import select_goal_pose
from qmapnav.reasoning.semantic_geometry import SemanticRegion


#: Parsed actions instruction accepts as a destination stage.
DESTINATION_ACTIONS = frozenset(
    {'go_near', 'go_to', 'stop_at', 'stop_near'}
)

ROUTE_STATUSES = frozenset(
    {
        'blocked',
        'planned',
        'stage_a_only',
        'terminal_only',
        'unresolved_stage',
        'unsupported_instruction',
    }
)


class TwoStageRouteError(ValueError):
    """Indicate that a two-stage perceived route cannot be constructed."""


@dataclass(frozen=True)
class PerceivedRouteStage:
    """One ordered semantic destination grounded in a perceived instance."""

    stage_index: int
    semantic_action: str
    target_reference_id: str
    resolved_instance_id: str
    target_region: SemanticRegion
    selected_goal_pose: tuple[float, float, float]
    confidence: float
    approach_path_length_m: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.stage_index, bool) or self.stage_index < 0:
            raise ValueError('stage_index must be a non-negative integer')
        if self.semantic_action not in DESTINATION_ACTIONS:
            expected = ', '.join(sorted(DESTINATION_ACTIONS))
            raise ValueError(f'semantic_action must be one of: {expected}')
        for name in ('target_reference_id', 'resolved_instance_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')
        if not isinstance(self.target_region, SemanticRegion):
            raise TypeError('target_region must be SemanticRegion')
        pose = tuple(self.selected_goal_pose)
        if len(pose) != 3 or not all(isfinite(value) for value in pose):
            raise ValueError('selected_goal_pose must hold three finite values')
        object.__setattr__(
            self, 'selected_goal_pose', tuple(float(v) for v in pose)
        )
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must lie in [0, 1]')
        if not isfinite(self.approach_path_length_m) or (
            self.approach_path_length_m < 0.0
        ):
            raise ValueError('approach_path_length_m must be non-negative')

    @property
    def is_terminal(self) -> bool:
        """Return whether this stage is a stopping destination."""
        return self.semantic_action in {'stop_at', 'stop_near'}

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'stage': self.stage_index,
            'action': self.semantic_action,
            'target': self.resolved_instance_id,
            'target_reference_id': self.target_reference_id,
            'region_id': self.target_region.region_id,
            'goal_pose': list(self.selected_goal_pose),
            'confidence': self.confidence,
            'approach_path_length_m': self.approach_path_length_m,
        }


@dataclass(frozen=True)
class PerceivedRoutePlan:
    """Exactly two ordered stages plus explicit unresolved state."""

    stages: tuple[PerceivedRouteStage, ...]
    route_status: str
    unresolved_stages: tuple[int, ...] = ()
    total_path_length_m: float = 0.0
    oracle_mode: bool = False

    def __post_init__(self) -> None:
        if self.route_status not in ROUTE_STATUSES:
            expected = ', '.join(sorted(ROUTE_STATUSES))
            raise ValueError(f'route_status must be one of: {expected}')
        stages = tuple(self.stages)
        if not all(
            isinstance(item, PerceivedRouteStage) for item in stages
        ):
            raise TypeError('stages must hold PerceivedRouteStage values')
        indices = [item.stage_index for item in stages]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError('stages must be uniquely and ascendingly ordered')
        object.__setattr__(self, 'stages', stages)
        if self.route_status == 'planned' and len(stages) != 2:
            raise ValueError('a planned instruction route has exactly two stages')
        if self.route_status == 'terminal_only' and len(stages) != 1:
            raise ValueError('a terminal-only route has exactly one stage')
        if self.route_status == 'stage_a_only' and len(stages) != 1:
            raise ValueError('a stage-A-only route has exactly one stage')
        object.__setattr__(
            self, 'unresolved_stages', tuple(self.unresolved_stages)
        )
        if not isfinite(self.total_path_length_m) or (
            self.total_path_length_m < 0.0
        ):
            raise ValueError('total_path_length_m must be non-negative')

    @property
    def planned(self) -> bool:
        """Return whether the full ordered two-stage route is usable."""
        return self.route_status == 'planned'

    @property
    def executable(self) -> bool:
        """
        Return whether any route can be driven, including the fallback.

        The frozen partial-credit policy always attempts the terminal target,
        so a terminal-only route is still executable even though an earlier
        stage never resolved.
        """
        return self.route_status in {
            'planned', 'stage_a_only', 'terminal_only'
        }

    def to_dict(self) -> dict[str, object]:
        """Return the stable ``semantic_route_planned`` trace record."""
        return {
            'event': 'semantic_route_planned',
            'stages': [item.to_dict() for item in self.stages],
            'route_status': self.route_status,
            'unresolved_stages': list(self.unresolved_stages),
            'total_path_length_m': self.total_path_length_m,
            'oracle_mode': self.oracle_mode,
        }


def two_stage_steps(
    task: TaskSpecification,
) -> tuple[tuple[int, str, str], ...]:
    """
    Extract exactly two ordered destination stages from a parsed task.

    Raises when the instruction is not a bounded instruction route so callers
    fall back explicitly instead of silently truncating a longer instruction.
    """
    if task.task_type != 'instruction_following':
        raise TwoStageRouteError(
            'two-stage routing requires an instruction-following task'
        )
    steps = tuple(task.ordered_route_steps)
    if len(steps) != 2:
        raise TwoStageRouteError(
            f'Instruction routes support exactly two stages, got {len(steps)}'
        )
    output = []
    for step in steps:
        if step.action not in DESTINATION_ACTIONS:
            raise TwoStageRouteError(
                f'unsupported instruction stage action: {step.action}'
            )
        if len(step.entity_ids) != 1:
            raise TwoStageRouteError(
                'Instruction stages reference exactly one entity each'
            )
        output.append((step.step_index, step.action, step.entity_ids[0]))
    return tuple(output)


def plan_two_stage_route(
    task: TaskSpecification,
    resolved: dict[str, ObjectInstance],
    *,
    grid: OccupancyGrid2D,
    start_xy: tuple[float, float],
    confidences: dict[str, float] | None = None,
    region_config: NearRegionConfig | None = None,
    scoring: GoalPoseScoringConfig | None = None,
    oracle_mode: bool = False,
    allow_terminal_only: bool = False,
    allow_stage_a_only: bool = False,
) -> PerceivedRoutePlan:
    """
    Build an ordered two-stage route from already-resolved instances.

    Stage A's goal pose is chosen with stage B's location in hand, so a valid
    near pose on the wrong side of A cannot make B needlessly expensive.
    """
    policy = region_config or NearRegionConfig()
    margins = confidences or {}
    try:
        steps = two_stage_steps(task)
    except TwoStageRouteError:
        return PerceivedRoutePlan(
            stages=(),
            route_status='unsupported_instruction',
            oracle_mode=oracle_mode,
        )

    unresolved = tuple(
        index for index, _, entity_id in steps if entity_id not in resolved
    )
    if unresolved:
        terminal_entity = steps[1][2]
        if allow_terminal_only and terminal_entity in resolved:
            # Partial credit: an unresolved intermediate landmark must never
            # stop the robot attempting the terminal target.
            return _terminal_only_plan(
                steps,
                resolved,
                grid=grid,
                start_xy=start_xy,
                policy=policy,
                scoring=scoring,
                margins=margins,
                unresolved=unresolved,
                oracle_mode=oracle_mode,
            )
        first_entity = steps[0][2]
        if allow_stage_a_only and first_entity in resolved:
            # Route-first information gathering: honour the known first stage
            # instead of remaining stationary when the terminal target has
            # not yet produced stable 3D geometry.
            return _stage_a_only_plan(
                steps,
                resolved,
                grid=grid,
                start_xy=start_xy,
                policy=policy,
                scoring=scoring,
                margins=margins,
                unresolved=unresolved,
                oracle_mode=oracle_mode,
            )
        return PerceivedRoutePlan(
            stages=(),
            route_status='unresolved_stage',
            unresolved_stages=unresolved,
            oracle_mode=oracle_mode,
        )

    boxes = {
        entity_id: perceived_box(resolved[entity_id])
        for _, _, entity_id in steps
    }
    regions = {
        entity_id: perceived_near_region(
            resolved[entity_id],
            config=policy,
            region_id=f'stage_{index}_near_'
                      f'{resolved[entity_id].instance_id}',
        )
        for index, _, entity_id in steps
    }

    final_entity = steps[1][2]
    final_centre = (
        boxes[final_entity].centre_xyz[0],
        boxes[final_entity].centre_xyz[1],
    )

    stages = []
    cursor = (float(start_xy[0]), float(start_xy[1]))
    total_length = 0.0
    for order, (index, action, entity_id) in enumerate(steps):
        approach_costs = cost_field(
            grid, cursor, clearance=policy.robot_clearance_m
        )
        goal = select_goal_pose(
            regions[entity_id],
            boxes[entity_id],
            grid=grid,
            start_xy=cursor,
            config=policy,
            scoring=scoring,
            # Only stage A weighs the onward leg.
            next_stage_xy=final_centre if order == 0 else None,
            costs=approach_costs,
        )
        if goal is None:
            return PerceivedRoutePlan(
                stages=tuple(stages),
                route_status='blocked',
                unresolved_stages=(index,),
                total_path_length_m=total_length,
                oracle_mode=oracle_mode,
            )
        stages.append(
            PerceivedRouteStage(
                stage_index=order,
                semantic_action=action,
                target_reference_id=entity_id,
                resolved_instance_id=str(resolved[entity_id].instance_id),
                target_region=regions[entity_id],
                selected_goal_pose=goal.pose_xy_yaw,
                confidence=_unit(margins.get(entity_id, 1.0)),
                approach_path_length_m=goal.approach_cost_m,
            )
        )
        total_length += goal.approach_cost_m
        cursor = (goal.pose_xy_yaw[0], goal.pose_xy_yaw[1])

    return PerceivedRoutePlan(
        stages=tuple(stages),
        route_status='planned',
        total_path_length_m=total_length,
        oracle_mode=oracle_mode,
    )


def _terminal_only_plan(
    steps,
    resolved,
    *,
    grid,
    start_xy,
    policy,
    scoring,
    margins,
    unresolved,
    oracle_mode,
) -> PerceivedRoutePlan:
    """Build the single-stage fallback that still reaches the terminal."""
    index, action, entity_id = steps[1]
    box = perceived_box(resolved[entity_id])
    region = perceived_near_region(
        resolved[entity_id],
        config=policy,
        region_id=f'terminal_near_{resolved[entity_id].instance_id}',
    )
    goal = select_goal_pose(
        region,
        box,
        grid=grid,
        start_xy=start_xy,
        config=policy,
        scoring=scoring,
    )
    if goal is None:
        return PerceivedRoutePlan(
            stages=(),
            route_status='blocked',
            unresolved_stages=unresolved,
            oracle_mode=oracle_mode,
        )
    stage = PerceivedRouteStage(
        stage_index=0,
        semantic_action=action,
        target_reference_id=entity_id,
        resolved_instance_id=str(resolved[entity_id].instance_id),
        target_region=region,
        selected_goal_pose=goal.pose_xy_yaw,
        confidence=_unit(margins.get(entity_id, 1.0)),
        approach_path_length_m=goal.approach_cost_m,
    )
    return PerceivedRoutePlan(
        stages=(stage,),
        route_status='terminal_only',
        unresolved_stages=unresolved,
        total_path_length_m=goal.approach_cost_m,
        oracle_mode=oracle_mode,
    )


def _stage_a_only_plan(
    steps,
    resolved,
    *,
    grid,
    start_xy,
    policy,
    scoring,
    margins,
    unresolved,
    oracle_mode,
) -> PerceivedRoutePlan:
    """Build one route-first stage that opens terminal re-observation."""
    _, action, entity_id = steps[0]
    box = perceived_box(resolved[entity_id])
    region = perceived_near_region(
        resolved[entity_id],
        config=policy,
        region_id=f'stage_0_near_{resolved[entity_id].instance_id}',
    )
    goal = select_goal_pose(
        region,
        box,
        grid=grid,
        start_xy=start_xy,
        config=policy,
        scoring=scoring,
    )
    if goal is None:
        return PerceivedRoutePlan(
            stages=(),
            route_status='blocked',
            unresolved_stages=unresolved,
            oracle_mode=oracle_mode,
        )
    stage = PerceivedRouteStage(
        stage_index=0,
        semantic_action=action,
        target_reference_id=entity_id,
        resolved_instance_id=str(resolved[entity_id].instance_id),
        target_region=region,
        selected_goal_pose=goal.pose_xy_yaw,
        confidence=_unit(margins.get(entity_id, 1.0)),
        approach_path_length_m=goal.approach_cost_m,
    )
    return PerceivedRoutePlan(
        stages=(stage,),
        route_status='stage_a_only',
        unresolved_stages=unresolved,
        total_path_length_m=goal.approach_cost_m,
        oracle_mode=oracle_mode,
    )


def stage_waypoints(
    plan: PerceivedRoutePlan,
    stage_index: int,
    *,
    grid: OccupancyGrid2D,
    start_xy: tuple[float, float],
    clearance: float = 0.35,
    spacing_m: float = 1.0,
) -> tuple[tuple[float, float, float], ...]:
    """
    Return sparse map-frame waypoints leading to one stage's goal pose.

    The dense grid path is thinned to roughly ``spacing_m`` so the executor
    still owns one active waypoint at a time without being fed every cell.
    """
    if not plan.executable:
        raise TwoStageRouteError('cannot route an unplanned two-stage plan')
    stage = plan.stages[stage_index]
    goal = stage.selected_goal_pose
    path = shortest_path(
        grid, start_xy, (goal[0], goal[1]), clearance=clearance
    )
    if path is None:
        return ()
    thinned = [path[0]]
    for point in path[1:]:
        last = thinned[-1]
        if (
            (point[0] - last[0]) ** 2 + (point[1] - last[1]) ** 2
        ) >= spacing_m ** 2:
            thinned.append(point)
    output = []
    for point in thinned[1:]:
        output.append((point[0], point[1], goal[2]))
    output.append(goal)
    return tuple(output)


def _unit(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    'DESTINATION_ACTIONS',
    'ROUTE_STATUSES',
    'PerceivedRoutePlan',
    'PerceivedRouteStage',
    'TwoStageRouteError',
    'plan_two_stage_route',
    'stage_waypoints',
    'two_stage_steps',
]
