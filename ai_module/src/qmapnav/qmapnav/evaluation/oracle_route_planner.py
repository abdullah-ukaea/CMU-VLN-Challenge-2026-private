"""Ground-truth semantic-region resolution and deterministic grid planning."""

from dataclasses import dataclass
from itertools import combinations, product
from math import ceil, hypot, isfinite

from qmapnav.common import RouteConstraint
from qmapnav.common import RouteStep
from qmapnav.common import TaskSpecification
from qmapnav.evaluation.ground_truth import OracleObject
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.evaluation.oracle import resolve_task_entities
from qmapnav.mapping.grid_planning import GridCell
from qmapnav.mapping.grid_planning import GridPlanningError
from qmapnav.mapping.grid_planning import PlanningGrid
from qmapnav.mapping.grid_planning import shortest_path_to_regions
from qmapnav.reasoning.semantic_geometry import make_approach_region
from qmapnav.reasoning.semantic_geometry import make_between_gate
from qmapnav.reasoning.semantic_geometry import make_near_region
from qmapnav.reasoning.semantic_geometry import object_footprint
from qmapnav.reasoning.semantic_geometry import Point2D
from qmapnav.reasoning.semantic_geometry import SemanticRegion


_NON_OBSTACLE_CLASSES = frozenset(
    {
        'carpet',
        'ceiling',
        'curtain',
        'door',
        'door_frame',
        'doorway',
        'floor',
        'map_wall_decal',
        'painting',
        'picture',
        'rug',
        'tree',
        'unknown',
        'wall',
        'wall_decal',
        'window',
    }
)


class RoutePlanningError(ValueError):
    """Indicate that no valid oracle semantic route can be constructed."""


@dataclass(frozen=True)
class OraclePlannerConfig:
    """Configurable geometric and raster parameters for the Day 3 planner."""

    resolution: float = 0.25
    robot_radius: float = 0.15
    map_padding: float = 2.0
    approach_minimum_clearance: float = 0.45
    approach_maximum_distance: float = 2.00
    near_minimum_distance: float = 0.60
    near_maximum_distance: float = 1.80
    forbidden_near_distance: float = 1.50
    gate_clearance: float = 0.05
    gate_depth: float = 1.00
    maximum_grid_cells: int = 1_000_000

    def __post_init__(self) -> None:
        positive = (
            self.resolution,
            self.robot_radius,
            self.map_padding,
            self.approach_minimum_clearance,
            self.approach_maximum_distance,
            self.near_minimum_distance,
            self.near_maximum_distance,
            self.forbidden_near_distance,
            self.gate_clearance,
            self.gate_depth,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('planner distances must be finite and positive')
        if self.approach_maximum_distance <= self.approach_minimum_clearance:
            raise ValueError('approach maximum must exceed minimum clearance')
        if self.near_maximum_distance <= self.near_minimum_distance:
            raise ValueError('near maximum must exceed minimum distance')
        if (
            isinstance(self.maximum_grid_cells, bool)
            or self.maximum_grid_cells <= 0
        ):
            raise ValueError('maximum_grid_cells must be a positive integer')


@dataclass(frozen=True)
class OracleRoutePlan:
    """A collision-free route through selected ordered semantic regions."""

    waypoints_xy: tuple[Point2D, ...]
    required_regions: tuple[SemanticRegion, ...]
    forbidden_regions: tuple[SemanticRegion, ...]
    resolved_step_object_ids: tuple[tuple[int, tuple[str, ...]], ...]
    grid: PlanningGrid
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.waypoints_xy:
            raise ValueError('route must contain at least one waypoint')
        if len(self.required_regions) != len(self.resolved_step_object_ids):
            raise ValueError('each required region must have one resolved step')


def _scene_bounds(
    scene: OracleScene,
    start_xy: Point2D,
    padding: float,
) -> tuple[float, float, float, float]:
    bounds = [
        object_footprint(obj).bounds
        for obj in scene.objects
        if obj.class_name not in _NON_OBSTACLE_CLASSES
    ]
    for region in scene.regions:
        half_x = region.dimensions_xyz[0] / 2.0
        half_y = region.dimensions_xyz[1] / 2.0
        bounds.append(
            (
                region.centre_xyz[0] - half_x,
                region.centre_xyz[1] - half_y,
                region.centre_xyz[0] + half_x,
                region.centre_xyz[1] + half_y,
            )
        )
    bounds.append((start_xy[0], start_xy[1], start_xy[0], start_xy[1]))
    return (
        min(item[0] for item in bounds) - padding,
        min(item[1] for item in bounds) - padding,
        max(item[2] for item in bounds) + padding,
        max(item[3] for item in bounds) + padding,
    )


def _cell_range_for_bounds(
    grid: PlanningGrid,
    bounds: tuple[float, float, float, float],
) -> tuple[range, range]:
    min_cell = grid.point_to_cell((bounds[0], bounds[1]))
    max_cell = grid.point_to_cell((bounds[2], bounds[3]))
    x_start = max(0, min_cell[0])
    y_start = max(0, min_cell[1])
    x_stop = min(grid.width, max_cell[0] + 1)
    y_stop = min(grid.height, max_cell[1] + 1)
    return range(x_start, x_stop), range(y_start, y_stop)


def build_planning_grid(
    scene: OracleScene,
    start_xy: Point2D,
    config: OraclePlannerConfig = OraclePlannerConfig(),
) -> PlanningGrid:
    """Rasterize inflated perfect object footprints into an occupancy grid."""
    if len(start_xy) != 2 or not all(isfinite(value) for value in start_xy):
        raise ValueError('start_xy must contain two finite coordinates')
    min_x, min_y, max_x, max_y = _scene_bounds(
        scene,
        start_xy,
        config.map_padding,
    )
    width = max(1, ceil((max_x - min_x) / config.resolution))
    height = max(1, ceil((max_y - min_y) / config.resolution))
    if width * height > config.maximum_grid_cells:
        raise RoutePlanningError(
            f'planning grid requires {width * height} cells, exceeding '
            f'{config.maximum_grid_cells}'
        )
    empty_grid = PlanningGrid(
        resolution=config.resolution,
        origin_xy=(min_x, min_y),
        width=width,
        height=height,
        occupied=frozenset(),
    )
    occupied: set[GridCell] = set()
    for obj in scene.objects:
        object_bottom = obj.centre_xyz[2] - obj.dimensions_xyz[2] / 2.0
        object_top = obj.centre_xyz[2] + obj.dimensions_xyz[2] / 2.0
        if (
            obj.class_name in _NON_OBSTACLE_CLASSES
            or object_bottom > 0.35
            or object_top < 0.15
        ):
            continue
        footprint = object_footprint(obj, config.robot_radius)
        x_range, y_range = _cell_range_for_bounds(empty_grid, footprint.bounds)
        for x_index in x_range:
            for y_index in y_range:
                cell = (x_index, y_index)
                if footprint.contains(empty_grid.cell_centre(cell)):
                    occupied.add(cell)
    return PlanningGrid(
        resolution=config.resolution,
        origin_xy=(min_x, min_y),
        width=width,
        height=height,
        occupied=frozenset(occupied),
    )


def _object_pairs(
    entity_ids: tuple[str, ...],
    candidates: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, str], ...]:
    if len(entity_ids) == 1:
        return tuple(combinations(candidates[entity_ids[0]], 2))
    if len(entity_ids) == 2:
        return tuple(
            (left, right)
            for left, right in product(
                candidates[entity_ids[0]],
                candidates[entity_ids[1]],
            )
            if left != right
        )
    raise RoutePlanningError('between semantics require one pair or two entities')


def _regions_for_step(
    step: RouteStep,
    candidates: dict[str, tuple[str, ...]],
    objects: dict[str, OracleObject],
    config: OraclePlannerConfig,
    warnings: list[str],
) -> tuple[SemanticRegion, ...]:
    if step.action == 'pass_between':
        regions = []
        for left, right in _object_pairs(tuple(step.entity_ids), candidates):
            gate = make_between_gate(
                objects[left],
                objects[right],
                robot_diameter=2.0 * config.robot_radius,
                clearance=config.gate_clearance,
                gate_depth=config.gate_depth,
                region_id=f'step_{step.step_index}_between_{left}_{right}',
                allow_inferred_clearance=True,
            )
            if gate.valid:
                regions.append(gate.region)
                required_width = 2.0 * (
                    config.robot_radius + config.gate_clearance
                )
                if gate.gap_width < required_width:
                    warnings.append(
                        f'step {step.step_index}: inferred traversable gate for '
                        f'conservative boxes {left}/{right}'
                    )
            else:
                warnings.append(
                    f'step {step.step_index}: rejected between pair '
                    f'{left}/{right}: {gate.reason}'
                )
        return tuple(region for region in regions if region is not None)

    regions = []
    for entity_id in step.entity_ids:
        for object_id in candidates[entity_id]:
            obj = objects[object_id]
            if step.action in {'go_near', 'pass_by', 'pass_near', 'stop_near'}:
                region = make_near_region(
                    obj,
                    config.near_minimum_distance,
                    config.near_maximum_distance,
                    region_id=f'step_{step.step_index}_near_{object_id}',
                )
            else:
                region = make_approach_region(
                    obj,
                    config.approach_minimum_clearance,
                    config.approach_maximum_distance,
                    region_id=f'step_{step.step_index}_approach_{object_id}',
                )
            regions.append(region)
            support = _support_object(obj, objects)
            if support is not None:
                if step.action in {
                    'go_near',
                    'pass_by',
                    'pass_near',
                    'stop_near',
                }:
                    support_region = make_near_region(
                        support,
                        config.near_minimum_distance,
                        config.near_maximum_distance,
                        region_id=(
                            f'step_{step.step_index}_near_{object_id}_via_'
                            f'{support.object_id}'
                        ),
                    )
                else:
                    support_region = make_approach_region(
                        support,
                        config.approach_minimum_clearance,
                        config.approach_maximum_distance,
                        region_id=(
                            f'step_{step.step_index}_approach_{object_id}_via_'
                            f'{support.object_id}'
                        ),
                    )
                regions.append(
                    SemanticRegion(
                        region_id=support_region.region_id,
                        region_type=support_region.region_type,
                        polygon=support_region.polygon,
                        exclusions=support_region.exclusions,
                        source_object_ids=(object_id,),
                        required=True,
                    )
                )
    return tuple(regions)


def _support_object(
    target: OracleObject,
    objects: dict[str, OracleObject],
) -> OracleObject | None:
    target_area = target.dimensions_xyz[0] * target.dimensions_xyz[1]
    possible = []
    for candidate in objects.values():
        if candidate.object_id == target.object_id:
            continue
        if candidate.class_name in {'ceiling', 'floor', 'wall'}:
            continue
        candidate_area = candidate.dimensions_xyz[0] * candidate.dimensions_xyz[1]
        if candidate_area <= target_area * 1.2:
            continue
        if candidate.centre_xyz[2] > target.centre_xyz[2] + 0.1:
            continue
        if not object_footprint(candidate, 0.15).contains(target.centre_xyz[:2]):
            continue
        possible.append((candidate_area, candidate.object_id, candidate))
    return min(possible)[2] if possible else None


def _regions_for_forbidden_constraint(
    constraint: RouteConstraint,
    candidates: dict[str, tuple[str, ...]],
    objects: dict[str, OracleObject],
    config: OraclePlannerConfig,
    warnings: list[str],
) -> tuple[SemanticRegion, ...]:
    if constraint.constraint_type == 'avoid_between':
        regions = []
        for left, right in _object_pairs(tuple(constraint.entity_ids), candidates):
            gate = make_between_gate(
                objects[left],
                objects[right],
                robot_diameter=2.0 * config.robot_radius,
                clearance=config.gate_clearance,
                gate_depth=config.gate_depth,
                required=False,
                region_id=f'forbidden_between_{left}_{right}',
            )
            if gate.valid:
                regions.append(gate.region)
            else:
                warnings.append(
                    f'ignored invalid forbidden pair {left}/{right}: {gate.reason}'
                )
        return tuple(region for region in regions if region is not None)

    regions = []
    for entity_id in constraint.entity_ids:
        for object_id in candidates[entity_id]:
            regions.append(
                make_near_region(
                    objects[object_id],
                    min_distance=0.0,
                    max_distance=config.forbidden_near_distance,
                    required=False,
                    region_id=f'forbidden_near_{object_id}',
                )
            )
    return tuple(regions)


def _nearest_free_cell(grid: PlanningGrid, requested: GridCell) -> GridCell:
    if grid.is_free(requested):
        return requested
    free_cells = (
        (hypot(cell[0] - requested[0], cell[1] - requested[1]), cell)
        for x_index in range(grid.width)
        for y_index in range(grid.height)
        for cell in ((x_index, y_index),)
        if grid.is_free(cell)
    )
    try:
        return min(free_cells)[1]
    except ValueError as exc:
        raise RoutePlanningError('planning grid has no free cell') from exc


def plan_semantic_route(
    task: TaskSpecification,
    scene: OracleScene,
    start_xy: Point2D,
    *,
    grid: PlanningGrid | None = None,
    config: OraclePlannerConfig = OraclePlannerConfig(),
) -> OracleRoutePlan:
    """Resolve ground-truth landmarks and plan through route steps in order."""
    if task.task_type != 'instruction_following':
        raise RoutePlanningError(
            'semantic route planner requires an instruction-following task'
        )
    if not task.ordered_route_steps:
        raise RoutePlanningError('instruction has no ordered route steps')
    resolution = resolve_task_entities(task, scene, strict_relations=False)
    candidates = dict(resolution.candidates)
    objects = {obj.object_id: obj for obj in scene.objects}
    warnings = list(resolution.warnings)
    referenced_entity_ids = {
        entity_id
        for step in task.ordered_route_steps
        for entity_id in step.entity_ids
    }
    referenced_entity_ids.update(
        entity_id
        for constraint in task.forbidden_constraints
        for entity_id in constraint.entity_ids
    )
    missing = sorted(
        entity_id for entity_id in referenced_entity_ids if not candidates[entity_id]
    )
    if missing:
        raise RoutePlanningError(
            f'no oracle object candidates for route entities: {missing}'
        )

    forbidden = tuple(
        region
        for constraint in task.forbidden_constraints
        for region in _regions_for_forbidden_constraint(
            constraint,
            candidates,
            objects,
            config,
            warnings,
        )
    )
    base_grid = grid or build_planning_grid(scene, start_xy, config)
    planning_grid = base_grid.with_blocked_regions(forbidden)
    requested_start = planning_grid.point_to_cell(start_xy)
    if not planning_grid.in_bounds(requested_start):
        raise RoutePlanningError('start position lies outside the planning grid')
    current = _nearest_free_cell(planning_grid, requested_start)
    if current != requested_start:
        warnings.append('start position was occupied and snapped to nearest free cell')

    route_cells = [current]
    required_regions = []
    resolved_steps = []
    for step in task.ordered_route_steps:
        options = _regions_for_step(
            step,
            candidates,
            objects,
            config,
            warnings,
        )
        if not options:
            raise RoutePlanningError(
                f'route step {step.step_index} has no valid semantic region'
            )
        if step.action == 'pass_between':
            planning_grid = (
                planning_grid.with_cleared_regions(options)
                .with_blocked_regions(forbidden)
            )
            warnings.append(
                f'step {step.step_index}: semantic gate overrides conservative '
                'object-box occupancy inside the annotated corridor'
            )
        try:
            segment, selected_region = shortest_path_to_regions(
                planning_grid,
                current,
                options,
            )
        except GridPlanningError as exc:
            raise RoutePlanningError(
                f'route step {step.step_index}: {exc}'
            ) from exc
        route_cells.extend(segment[1:])
        current = segment[-1]
        required_regions.append(selected_region)
        resolved_steps.append(
            (step.step_index, selected_region.source_object_ids)
        )

    points = tuple(planning_grid.cell_centre(cell) for cell in route_cells)
    if requested_start == route_cells[0] and planning_grid.is_free(requested_start):
        points = (tuple(float(value) for value in start_xy),) + points[1:]
    return OracleRoutePlan(
        waypoints_xy=points,
        required_regions=tuple(required_regions),
        forbidden_regions=forbidden,
        resolved_step_object_ids=tuple(resolved_steps),
        grid=planning_grid,
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    'OraclePlannerConfig',
    'OracleRoutePlan',
    'PlanningGrid',
    'RoutePlanningError',
    'build_planning_grid',
    'plan_semantic_route',
]
