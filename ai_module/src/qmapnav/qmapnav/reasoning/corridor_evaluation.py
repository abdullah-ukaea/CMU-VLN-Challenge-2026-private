"""Physical traversability checks for path-level between-anchor gates."""

from collections import deque
from dataclasses import dataclass
from math import atan2, isfinite
from typing import Sequence

import cv2
import numpy as np

from qmapnav.reasoning.resolution_contracts import PairHypothesis
from qmapnav.reasoning.route_planner import PlanningGrid
from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_geometry import SupportGeometry


@dataclass(frozen=True)
class CorridorConfig:
    """Gate policy with robot width supplied by system configuration."""

    robot_width_m: float
    safety_clearance_m: float = 0.15
    minimum_depth_m: float = 0.60
    occupancy_free_fraction: float = 0.90
    maximum_anchor_separation_m: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            'robot_width_m',
            'safety_clearance_m',
            'minimum_depth_m',
            'maximum_anchor_separation_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not isfinite(self.occupancy_free_fraction) or not (
            0.0 <= self.occupancy_free_fraction <= 1.0
        ):
            raise ValueError('occupancy_free_fraction must lie in [0, 1]')


@dataclass(frozen=True)
class GateGeometry:
    """Explicit map-frame gap and crossing corridor geometry."""

    pair_ids: tuple[str, str]
    boundary_first_xy: tuple[float, float]
    boundary_second_xy: tuple[float, float]
    centre_xy: tuple[float, float]
    crossing_direction_xy: tuple[float, float]
    polygon_xy: tuple[tuple[float, float], ...]
    width_m: float
    depth_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if len(set(self.pair_ids)) != 2:
            raise ValueError('gate requires two distinct pair IDs')
        for point in (
            self.boundary_first_xy,
            self.boundary_second_xy,
            self.centre_xy,
            self.crossing_direction_xy,
        ):
            if len(point) != 2 or not all(isfinite(value) for value in point):
                raise ValueError('gate points must contain two finite values')
        if len(self.polygon_xy) != 4:
            raise ValueError('gate polygon must be a rectangle')
        if not all(isfinite(value) and value > 0.0 for value in (
            self.width_m, self.depth_m
        )):
            raise ValueError('gate extent must be finite and positive')


@dataclass(frozen=True)
class CorridorEvaluation:
    """Gate hypothesis with every physical rejection reason retained."""

    pair: PairHypothesis
    gate: GateGeometry | None
    reasons: tuple[str, ...]
    blocker_ids: tuple[str, ...]
    occupancy_free_fraction: float
    approach_reachable: bool
    exit_reachable: bool

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError('corridor evaluation requires a reason')

    def to_dict(self) -> dict[str, object]:
        """Return complete trace-ready corridor evidence."""
        return {
            'pair': self.pair.to_dict(),
            'gate': None if self.gate is None else {
                'pair_ids': list(self.gate.pair_ids),
                'boundary_first_xy': list(self.gate.boundary_first_xy),
                'boundary_second_xy': list(self.gate.boundary_second_xy),
                'centre_xy': list(self.gate.centre_xy),
                'crossing_direction_xy': list(
                    self.gate.crossing_direction_xy
                ),
                'polygon_xy': [list(item) for item in self.gate.polygon_xy],
                'width_m': self.gate.width_m,
                'depth_m': self.gate.depth_m,
                'yaw_rad': self.gate.yaw_rad,
            },
            'reasons': list(self.reasons),
            'blocker_ids': list(self.blocker_ids),
            'occupancy_free_fraction': self.occupancy_free_fraction,
            'approach_reachable': self.approach_reachable,
            'exit_reachable': self.exit_reachable,
        }


def build_gate(
    first: SupportGeometry,
    second: SupportGeometry,
    config: CorridorConfig,
) -> GateGeometry | None:
    """Build the physical gap from nearest footprint boundary points."""
    _validate_pair(first, second)
    connector = second.centre_xyz[:2] - first.centre_xyz[:2]
    separation = float(np.linalg.norm(connector))
    if separation <= 1.0e-9:
        return None
    across = connector / separation
    boundary_first, boundary_second, width = _closest_polygon_points(
        first.footprint_xy,
        second.footprint_xy,
        (first.centre_xyz[:2] + second.centre_xyz[:2]) / 2.0,
    )
    if width <= 1.0e-9:
        return None
    centre = (boundary_first + boundary_second) / 2.0
    crossing = np.array((-across[1], across[0]), dtype=np.float64)
    yaw = atan2(float(crossing[1]), float(crossing[0]))
    polygon = _rectangle(
        centre, crossing, config.minimum_depth_m, config.robot_width_m
    )
    return GateGeometry(
        tuple(sorted((first.entity_id, second.entity_id))),
        tuple(map(float, boundary_first)),
        tuple(map(float, boundary_second)),
        tuple(map(float, centre)),
        tuple(map(float, crossing)),
        tuple(tuple(map(float, point)) for point in polygon),
        width,
        config.minimum_depth_m,
        yaw,
    )


def evaluate_corridor(
    first: SupportGeometry,
    second: SupportGeometry,
    grid: PlanningGrid,
    config: CorridorConfig,
    *,
    blockers: Sequence[SupportGeometry] = (),
) -> CorridorEvaluation:
    """Validate width, raster occupancy, blockers, approach, and exit."""
    _validate_pair(first, second)
    if not isinstance(grid, PlanningGrid):
        raise TypeError('grid must be PlanningGrid')
    centre_distance = float(np.linalg.norm(
        first.centre_xyz[:2] - second.centre_xyz[:2]
    ))
    footprint_gap = footprint_metrics(first, second).edge_distance_m
    gate = build_gate(first, second, config)
    required_width = config.robot_width_m + 2.0 * config.safety_clearance_m
    reasons = []
    if centre_distance > config.maximum_anchor_separation_m:
        reasons.append('anchor_separation_exceeds_maximum')
    if gate is None or footprint_gap <= 1.0e-9:
        reasons.append('anchor_footprints_touch_or_overlap')
    elif footprint_gap < required_width:
        reasons.append('corridor_too_narrow_for_robot_clearance')

    blocker_ids = ()
    occupancy_fraction = 0.0
    approach_reachable = False
    exit_reachable = False
    if gate is not None:
        blocker_ids = tuple(sorted(
            item.entity_id for item in blockers
            if item.entity_id not in {first.entity_id, second.entity_id}
            and _polygons_intersect(item.footprint_xy, gate.polygon_xy)
        ))
        if blocker_ids:
            reasons.append('corridor_blocked_by_third_entity')
        cells = _polygon_cells(grid, gate.polygon_xy)
        free = sum(grid.is_free(cell) for cell in cells)
        occupancy_fraction = free / len(cells) if cells else 0.0
        if occupancy_fraction < config.occupancy_free_fraction:
            reasons.append('corridor_occupancy_below_free_threshold')
        direction = np.asarray(gate.crossing_direction_xy)
        centre = np.asarray(gate.centre_xy)
        offset = config.minimum_depth_m / 2.0 + grid.resolution
        approach_cell = grid.point_to_cell(tuple(centre - direction * offset))
        exit_cell = grid.point_to_cell(tuple(centre + direction * offset))
        approach_reachable = grid.is_free(approach_cell)
        exit_reachable = grid.is_free(exit_cell)
        if not approach_reachable:
            reasons.append('approach_region_unreachable')
        if not exit_reachable:
            reasons.append('exit_region_unreachable')
        if approach_reachable and exit_reachable and not _connected(
            grid, approach_cell, exit_cell
        ):
            approach_reachable = False
            exit_reachable = False
            reasons.append('approach_exit_regions_disconnected')

    traversable = not reasons
    if traversable:
        reasons.append('traversable_gate')
    width_score = min(1.0, footprint_gap / max(required_width, 1.0e-9))
    reachability_score = float(approach_reachable and exit_reachable)
    score = (
        0.35 * width_score
        + 0.35 * occupancy_fraction
        + 0.20 * reachability_score
        + 0.10 * min(first.confidence, second.confidence)
    )
    if not traversable:
        score *= 0.25
    pair = PairHypothesis(
        first.entity_id,
        second.entity_id,
        first.semantic_class if first.semantic_class == second.semantic_class
        else f'{first.semantic_class}|{second.semantic_class}',
        centre_distance,
        footprint_gap,
        traversable,
        None if gate is None else gate.width_m,
        score,
        min(first.confidence, second.confidence),
        {
            'minimum_traversable_width_m': required_width,
            'occupancy_free_fraction': occupancy_fraction,
            'approach_reachable': float(approach_reachable),
            'exit_reachable': float(exit_reachable),
            'blocker_count': float(len(blocker_ids)),
            'anchor_separation_m': centre_distance,
        },
    )
    return CorridorEvaluation(
        pair,
        gate,
        tuple(reasons),
        blocker_ids,
        occupancy_fraction,
        approach_reachable,
        exit_reachable,
    )


def rank_corridors(
    evaluations: Sequence[CorridorEvaluation],
) -> tuple[CorridorEvaluation, ...]:
    """Rank traversable gates first with deterministic canonical tie breaks."""
    return tuple(sorted(
        evaluations,
        key=lambda item: (
            not item.pair.traversable,
            -item.pair.score,
            item.pair.first_id,
            item.pair.second_id,
        ),
    ))


def _validate_pair(first, second):
    if not isinstance(first, SupportGeometry) or not isinstance(
        second, SupportGeometry
    ):
        raise TypeError('gate anchors must be SupportGeometry')
    if first.entity_id == second.entity_id:
        raise ValueError('gate anchors must be distinct')


def _rectangle(centre, axis, length, width):
    normal = np.array((-axis[1], axis[0]), dtype=np.float64)
    half_length = axis * length / 2.0
    half_width = normal * width / 2.0
    return np.array((
        centre - half_length - half_width,
        centre + half_length - half_width,
        centre + half_length + half_width,
        centre - half_length + half_width,
    ))


def _closest_polygon_points(first, second, preferred_midpoint):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    preferred = np.asarray(preferred_midpoint, dtype=np.float64)
    if _polygons_intersect(first, second):
        centre = (first.mean(axis=0) + second.mean(axis=0)) / 2.0
        return centre, centre, 0.0
    best = None

    def consider(point_first, point_second):
        nonlocal best
        distance = float(np.linalg.norm(point_first - point_second))
        midpoint_error = float(np.linalg.norm(
            (point_first + point_second) / 2.0 - preferred
        ))
        candidate = (
            (distance, midpoint_error), point_first.copy(),
            point_second.copy(),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    for point in first:
        for start, end in _edges(second):
            projected = _project_segment(point, start, end)
            consider(point, projected)
    for point in second:
        for start, end in _edges(first):
            projected = _project_segment(point, start, end)
            consider(projected, point)
    for first_start, first_end in _edges(first):
        for second_start, second_end in _edges(second):
            parallel = _parallel_overlap_midpoints(
                first_start, first_end, second_start, second_end
            )
            if parallel is not None:
                consider(*parallel)
    return best[1], best[2], best[0][0]


def _edges(polygon):
    return zip(polygon, np.roll(polygon, -1, axis=0))


def _project_segment(point, start, end):
    vector = end - start
    squared = float(np.dot(vector, vector))
    if squared <= 1.0e-12:
        return start
    position = float(np.dot(point - start, vector) / squared)
    return start + min(1.0, max(0.0, position)) * vector


def _parallel_overlap_midpoints(first_start, first_end, second_start,
                                second_end):
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    first_length = float(np.linalg.norm(first_vector))
    second_length = float(np.linalg.norm(second_vector))
    if first_length <= 1.0e-12 or second_length <= 1.0e-12:
        return None
    cross = (
        first_vector[0] * second_vector[1]
        - first_vector[1] * second_vector[0]
    )
    if abs(float(cross)) > 1.0e-9:
        return None
    axis = first_vector / first_length
    second_positions = sorted((
        float(np.dot(second_start - first_start, axis)),
        float(np.dot(second_end - first_start, axis)),
    ))
    overlap_start = max(0.0, second_positions[0])
    overlap_end = min(first_length, second_positions[1])
    if overlap_end < overlap_start:
        return None
    first_point = first_start + axis * (overlap_start + overlap_end) / 2.0
    second_point = _project_segment(first_point, second_start, second_end)
    return first_point, second_point


def _polygons_intersect(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    return cv2.intersectConvexConvex(first, second)[0] > 1.0e-9


def _polygon_cells(grid, polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    first = grid.point_to_cell(tuple(minimum))
    last = grid.point_to_cell(tuple(maximum))
    cells = []
    contour = polygon.astype(np.float32)
    for x_index in range(first[0] - 1, last[0] + 2):
        for y_index in range(first[1] - 1, last[1] + 2):
            cell = (x_index, y_index)
            if grid.in_bounds(cell) and cv2.pointPolygonTest(
                contour, grid.cell_centre(cell), False
            ) >= 0.0:
                cells.append(cell)
    return cells


def _connected(grid, start, goal):
    queue = deque((start,))
    visited = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (cell[0] + delta[0], cell[1] + delta[1])
            if neighbour not in visited and grid.is_free(neighbour):
                visited.add(neighbour)
                queue.append(neighbour)
    return False


__all__ = [
    'CorridorConfig',
    'CorridorEvaluation',
    'GateGeometry',
    'build_gate',
    'evaluate_corridor',
    'rank_corridors',
]
