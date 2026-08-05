"""
Deterministic free-space search over the three-state exploration grid.

The Day 3 planner owns its own binary ``PlanningGrid`` built from a perfect
``OracleScene``. Perceived exploration instead needs unknown-aware costs, so
this module provides the equivalent eight-connected search over
``OccupancyGrid2D`` and is shared by viewpoint scoring and the perceived
two-stage route.
"""

from heapq import heappop
from heapq import heappush
from math import isfinite
from math import sqrt

from qmapnav.mapping.occupancy_grid import GridCell
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.occupancy_grid import Point2D


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


__all__ = [
    'cost_field',
    'is_reachable',
    'planned_distance',
    'shortest_path',
]
