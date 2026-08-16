"""
Deterministic free-space search over the three-state exploration grid.

The Day 3 planner owns its own binary ``PlanningGrid`` built from a perfect
``OracleScene``. Perceived exploration instead needs unknown-aware costs, so
this module provides the equivalent eight-connected search over
``OccupancyGrid2D`` and is shared by viewpoint scoring and the perceived
two-stage route.
"""

from dataclasses import dataclass
from heapq import heappop
from heapq import heappush
from math import floor
from math import isfinite
from math import sqrt

from qmapnav.mapping.occupancy_grid import GridCell
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.occupancy_grid import Point2D
from qmapnav.reasoning.semantic_geometry import SemanticRegion


_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class GridPlanningError(ValueError):
    """Indicate that a finite planning grid cannot satisfy a route request."""


@dataclass(frozen=True)
class PlanningGrid:
    """A finite row-major occupancy grid in the map frame."""

    resolution: float
    origin_xy: Point2D
    width: int
    height: int
    occupied: frozenset[GridCell]

    def __post_init__(self) -> None:
        if not isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError('resolution must be finite and positive')
        if len(self.origin_xy) != 2 or not all(
            isfinite(value) for value in self.origin_xy
        ):
            raise ValueError('origin_xy must contain two finite coordinates')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('grid dimensions must be positive')
        occupied = frozenset(self.occupied)
        if any(not self.in_bounds(cell) for cell in occupied):
            raise ValueError('occupied cells must lie inside the grid')
        object.__setattr__(self, 'occupied', occupied)

    def in_bounds(self, cell: GridCell) -> bool:
        """Return whether a cell index lies within this grid."""
        x_index, y_index = cell
        return 0 <= x_index < self.width and 0 <= y_index < self.height

    def is_free(self, cell: GridCell) -> bool:
        """Return whether a cell is in bounds and unoccupied."""
        return self.in_bounds(cell) and cell not in self.occupied

    def point_to_cell(self, point: Point2D) -> GridCell:
        """Convert a map-frame point to its containing grid cell."""
        return (
            floor((point[0] - self.origin_xy[0]) / self.resolution),
            floor((point[1] - self.origin_xy[1]) / self.resolution),
        )

    def cell_centre(self, cell: GridCell) -> Point2D:
        """Convert a grid cell to its map-frame centre point."""
        return (
            self.origin_xy[0] + (cell[0] + 0.5) * self.resolution,
            self.origin_xy[1] + (cell[1] + 0.5) * self.resolution,
        )

    def with_blocked_regions(
        self,
        regions: tuple[SemanticRegion, ...],
    ) -> 'PlanningGrid':
        """Return a copy with all usable cells inside regions marked blocked."""
        blocked = set(self.occupied)
        for region in regions:
            blocked.update(_cells_in_region(self, region, require_free=False))
        return PlanningGrid(
            resolution=self.resolution,
            origin_xy=self.origin_xy,
            width=self.width,
            height=self.height,
            occupied=frozenset(blocked),
        )

    def with_cleared_regions(
        self,
        regions: tuple[SemanticRegion, ...],
    ) -> 'PlanningGrid':
        """Clear cells in trusted semantic corridors from conservative boxes."""
        cleared = set(self.occupied)
        for region in regions:
            cleared.difference_update(
                _cells_in_region(self, region, require_free=False)
            )
        return PlanningGrid(
            resolution=self.resolution,
            origin_xy=self.origin_xy,
            width=self.width,
            height=self.height,
            occupied=frozenset(cleared),
        )


def _passable(
    grid: OccupancyGrid2D,
    cell: GridCell,
    clearance: float,
) -> bool:
    if not grid.in_bounds(cell):
        return False
    centre = grid.cell_centre(cell)
    return grid.is_free(centre[0], centre[1], clearance=clearance)


def cost_field(
    grid: OccupancyGrid2D,
    start_xy: Point2D,
    *,
    clearance: float = 0.0,
    maximum_distance_m: float | None = None,
) -> dict[GridCell, float]:
    """
    Return metric travel cost from ``start_xy`` to every reachable cell.

    Cells absent from the mapping are unreachable, which lets callers reject
    a viewpoint outright instead of merely penalizing it.
    """
    if maximum_distance_m is not None and (
        not isfinite(maximum_distance_m) or maximum_distance_m < 0.0
    ):
        raise ValueError('maximum_distance_m must be finite and non-negative')
    start = grid.point_to_cell(start_xy)
    if not _passable(grid, start, clearance):
        return {}
    resolution = grid.resolution
    costs: dict[GridCell, float] = {start: 0.0}
    queue: list[tuple[float, GridCell]] = [(0.0, start)]
    while queue:
        cost, cell = heappop(queue)
        if cost > costs.get(cell, float('inf')):
            continue
        for delta_x, delta_y in _NEIGHBOUR_OFFSETS:
            neighbour = (cell[0] + delta_x, cell[1] + delta_y)
            if not _passable(grid, neighbour, clearance):
                continue
            if delta_x and delta_y:
                # Refuse to cut a diagonal between two blocked cells.
                if not _passable(grid, (cell[0] + delta_x, cell[1]),
                                 clearance):
                    continue
                if not _passable(grid, (cell[0], cell[1] + delta_y),
                                 clearance):
                    continue
                step = sqrt(2.0) * resolution
            else:
                step = resolution
            new_cost = cost + step
            if maximum_distance_m is not None and (
                new_cost > maximum_distance_m
            ):
                continue
            if new_cost >= costs.get(neighbour, float('inf')):
                continue
            costs[neighbour] = new_cost
            heappush(queue, (new_cost, neighbour))
    return costs


def planned_distance(
    grid: OccupancyGrid2D,
    start_xy: Point2D,
    goal_xy: Point2D,
    *,
    clearance: float = 0.0,
    maximum_distance_m: float | None = None,
    costs: dict[GridCell, float] | None = None,
) -> float | None:
    """Return planned path length in metres, or ``None`` when unreachable."""
    field = (
        costs
        if costs is not None
        else cost_field(
            grid,
            start_xy,
            clearance=clearance,
            maximum_distance_m=maximum_distance_m,
        )
    )
    return field.get(grid.point_to_cell(goal_xy))


def shortest_path(
    grid: OccupancyGrid2D,
    start_xy: Point2D,
    goal_xy: Point2D,
    *,
    clearance: float = 0.0,
) -> tuple[Point2D, ...] | None:
    """Return map-frame cell centres from start to goal, or ``None``."""
    start = grid.point_to_cell(start_xy)
    goal = grid.point_to_cell(goal_xy)
    if not _passable(grid, start, clearance):
        return None
    if not _passable(grid, goal, clearance):
        return None
    if start == goal:
        return (grid.cell_centre(start),)
    resolution = grid.resolution
    costs: dict[GridCell, float] = {start: 0.0}
    parents: dict[GridCell, GridCell] = {}
    queue: list[tuple[float, GridCell]] = [(0.0, start)]
    while queue:
        cost, cell = heappop(queue)
        if cost > costs.get(cell, float('inf')):
            continue
        if cell == goal:
            break
        for delta_x, delta_y in _NEIGHBOUR_OFFSETS:
            neighbour = (cell[0] + delta_x, cell[1] + delta_y)
            if not _passable(grid, neighbour, clearance):
                continue
            if delta_x and delta_y:
                if not _passable(grid, (cell[0] + delta_x, cell[1]),
                                 clearance):
                    continue
                if not _passable(grid, (cell[0], cell[1] + delta_y),
                                 clearance):
                    continue
                step = sqrt(2.0) * resolution
            else:
                step = resolution
            new_cost = cost + step
            if new_cost >= costs.get(neighbour, float('inf')):
                continue
            costs[neighbour] = new_cost
            parents[neighbour] = cell
            heappush(queue, (new_cost, neighbour))
    if goal not in costs:
        return None
    path = [goal]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    path.reverse()
    return tuple(grid.cell_centre(cell) for cell in path)


def is_reachable(
    grid: OccupancyGrid2D,
    start_xy: Point2D,
    goal_xy: Point2D,
    *,
    clearance: float = 0.0,
    costs: dict[GridCell, float] | None = None,
) -> bool:
    """Return whether free space connects two map-frame points."""
    return planned_distance(
        grid, start_xy, goal_xy, clearance=clearance, costs=costs
    ) is not None


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


def _cells_in_region(
    grid: PlanningGrid,
    region: SemanticRegion,
    *,
    require_free: bool,
) -> tuple[GridCell, ...]:
    x_range, y_range = _cell_range_for_bounds(grid, region.polygon.bounds)
    cells = []
    for x_index in x_range:
        for y_index in y_range:
            cell = (x_index, y_index)
            if require_free and not grid.is_free(cell):
                continue
            if region.contains(grid.cell_centre(cell)):
                cells.append(cell)
    return tuple(cells)


def _planning_neighbours(grid: PlanningGrid, cell: GridCell):
    x_index, y_index = cell
    for delta_x, delta_y in _NEIGHBOUR_OFFSETS:
        neighbour = (x_index + delta_x, y_index + delta_y)
        if not grid.is_free(neighbour):
            continue
        if delta_x and delta_y:
            if not grid.is_free((x_index + delta_x, y_index)):
                continue
            if not grid.is_free((x_index, y_index + delta_y)):
                continue
            cost = sqrt(2.0)
        else:
            cost = 1.0
        yield neighbour, cost


def _reconstruct_planning_path(
    parents: dict[GridCell, GridCell],
    goal: GridCell,
) -> tuple[GridCell, ...]:
    path = [goal]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    path.reverse()
    return tuple(path)


def shortest_path_to_regions(
    grid: PlanningGrid,
    start: GridCell,
    regions: tuple[SemanticRegion, ...],
) -> tuple[tuple[GridCell, ...], SemanticRegion]:
    """Return an eight-connected A* path to the first reachable region."""
    goals: dict[GridCell, int] = {}
    for index, region in enumerate(regions):
        for cell in _cells_in_region(grid, region, require_free=True):
            goals.setdefault(cell, index)
    if not goals:
        raise GridPlanningError('semantic region contains no free grid cell')
    if start in goals:
        return (start,), regions[goals[start]]

    queue: list[tuple[float, GridCell]] = [(0.0, start)]
    costs = {start: 0.0}
    parents: dict[GridCell, GridCell] = {}
    while queue:
        cost, cell = heappop(queue)
        if cost != costs.get(cell):
            continue
        if cell in goals:
            return (
                _reconstruct_planning_path(parents, cell),
                regions[goals[cell]],
            )
        for neighbour, step_cost in _planning_neighbours(grid, cell):
            new_cost = cost + step_cost
            if new_cost >= costs.get(neighbour, float('inf')):
                continue
            costs[neighbour] = new_cost
            parents[neighbour] = cell
            heappush(queue, (new_cost, neighbour))
    raise GridPlanningError('no collision-free path reaches the semantic region')


__all__ = [
    'GridPlanningError',
    'PlanningGrid',
    'cost_field',
    'is_reachable',
    'planned_distance',
    'shortest_path',
    'shortest_path_to_regions',
]
