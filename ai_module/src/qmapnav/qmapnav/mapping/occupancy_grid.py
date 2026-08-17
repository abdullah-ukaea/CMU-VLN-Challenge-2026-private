"""
Three-state planar occupancy used for visibility and frontier reasoning.

The persistent protocol scan accumulator proves cells *free* or *occupied*; every
other cell is genuinely *unknown*. Frontier extraction and unexplored-area
scoring both need that third state, so exploration rasterizes a bounded local
window rather than reusing the binary evaluation planning grid.
"""

from collections import deque
from dataclasses import dataclass
from math import ceil
from math import cos
from math import floor
from math import hypot
from math import isfinite
from math import pi
from math import sin

import numpy as np


CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OCCUPIED = 2

Point2D = tuple[float, float]
GridCell = tuple[int, int]


@dataclass(frozen=True)
class FrontierCluster:
    """A contiguous run of free cells bordering unknown space."""

    cluster_id: str
    cells: tuple[GridCell, ...]
    centroid_xy: Point2D
    unknown_neighbour_count: int

    @property
    def size(self) -> int:
        """Return how many frontier cells the cluster contains."""
        return len(self.cells)


class OccupancyGrid2D:
    """A bounded row-major grid holding unknown, free, and occupied cells."""

    def __init__(
        self,
        resolution: float,
        origin_xy: Point2D,
        width: int,
        height: int,
        cells: np.ndarray | None = None,
    ) -> None:
        """Build a grid, defaulting every cell to unknown."""
        if not isfinite(resolution) or resolution <= 0.0:
            raise ValueError('resolution must be finite and positive')
        origin = tuple(origin_xy)
        if len(origin) != 2 or not all(isfinite(item) for item in origin):
            raise ValueError('origin_xy must contain two finite coordinates')
        for name, value in (('width', width), ('height', height)):
            if isinstance(value, bool) or not isinstance(value, int) or (
                value <= 0
            ):
                raise ValueError(f'{name} must be a positive integer')
        self._resolution = float(resolution)
        self._origin = (float(origin[0]), float(origin[1]))
        self._width = width
        self._height = height
        if cells is None:
            self._cells = np.full((height, width), CELL_UNKNOWN, dtype=np.uint8)
        else:
            array = np.asarray(cells, dtype=np.uint8)
            if array.shape != (height, width):
                raise ValueError('cells must have shape (height, width)')
            if not np.all(np.isin(array, (CELL_UNKNOWN, CELL_FREE,
                                          CELL_OCCUPIED))):
                raise ValueError('cells must hold only known state values')
            self._cells = array.copy()
        # Clearance-inflated passability is recomputed lazily and cached per
        # clearance, because grid search queries it once per neighbour and
        # recomputing the disc test each time dominates planning runtime.
        self._passable_cache: dict[float, np.ndarray] = {}

    @property
    def resolution(self) -> float:
        """Return the metric size of one cell."""
        return self._resolution

    @property
    def origin_xy(self) -> Point2D:
        """Return the map-frame corner of cell ``(0, 0)``."""
        return self._origin

    @property
    def width(self) -> int:
        """Return the grid width in cells."""
        return self._width

    @property
    def height(self) -> int:
        """Return the grid height in cells."""
        return self._height

    def in_bounds(self, cell: GridCell) -> bool:
        """Return whether a cell index lies inside the grid."""
        x_index, y_index = cell
        return 0 <= x_index < self._width and 0 <= y_index < self._height

    def point_to_cell(self, point: Point2D) -> GridCell:
        """Convert a map-frame point to its containing cell."""
        return (
            floor((point[0] - self._origin[0]) / self._resolution),
            floor((point[1] - self._origin[1]) / self._resolution),
        )

    def cell_centre(self, cell: GridCell) -> Point2D:
        """Convert a cell index to its map-frame centre."""
        return (
            self._origin[0] + (cell[0] + 0.5) * self._resolution,
            self._origin[1] + (cell[1] + 0.5) * self._resolution,
        )

    def state(self, cell: GridCell) -> int:
        """Return the stored state, treating out-of-bounds as unknown."""
        if not self.in_bounds(cell):
            return CELL_UNKNOWN
        return int(self._cells[cell[1], cell[0]])

    def set_state(self, cell: GridCell, value: int) -> None:
        """Set one in-bounds cell state."""
        if value not in (CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED):
            raise ValueError('value must be a known cell state')
        if not self.in_bounds(cell):
            raise ValueError('cell must lie inside the grid')
        self._cells[cell[1], cell[0]] = value
        self._passable_cache.clear()

    def fill_rectangle(
        self,
        bounds: tuple[float, float, float, float],
        value: int,
    ) -> None:
        """Set every cell whose centre lies inside a metric rectangle."""
        min_x, min_y, max_x, max_y = bounds
        start = self.point_to_cell((min_x, min_y))
        stop = self.point_to_cell((max_x, max_y))
        for x_index in range(max(0, start[0]), min(self._width, stop[0] + 1)):
            for y_index in range(
                max(0, start[1]), min(self._height, stop[1] + 1)
            ):
                centre = self.cell_centre((x_index, y_index))
                if (
                    min_x <= centre[0] <= max_x
                    and min_y <= centre[1] <= max_y
                ):
                    self._cells[y_index, x_index] = value
        self._passable_cache.clear()

    def state_at_point(self, x: float, y: float) -> int:
        """Return the state of the cell containing a map-frame point."""
        return self.state(self.point_to_cell((x, y)))

    def passable_mask(self, clearance: float) -> np.ndarray:
        """Return a cached mask of cells that are free with ``clearance``."""
        key = float(clearance)
        cached = self._passable_cache.get(key)
        if cached is not None:
            return cached
        free = self._cells == CELL_FREE
        if key <= 0.0:
            mask = free
        else:
            occupied = self._cells == CELL_OCCUPIED
            blocked = occupied.copy()
            radius = int(ceil(key / self._resolution))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if hypot(dx, dy) * self._resolution > key:
                        continue
                    # Shift the occupied mask so any cell within the disc of
                    # an obstacle becomes blocked.
                    blocked |= _shift(occupied, dx, dy)
            mask = free & ~blocked
        mask.setflags(write=False)
        self._passable_cache[key] = mask
        return mask

    def is_free(self, x: float, y: float, *, clearance: float = 0.0) -> bool:
        """Return whether a disc of ``clearance`` around a point is free."""
        if not all(isfinite(value) for value in (x, y, clearance)):
            return False
        if clearance < 0.0:
            return False
        cell = self.point_to_cell((x, y))
        if not self.in_bounds(cell):
            return False
        return bool(self.passable_mask(clearance)[cell[1], cell[0]])

    def line_of_sight(self, start: Point2D, end: Point2D) -> bool:
        """Return whether no occupied cell blocks the segment."""
        distance = hypot(end[0] - start[0], end[1] - start[1])
        if distance <= 1e-9:
            return self.state_at_point(*start) != CELL_OCCUPIED
        steps = max(1, int(ceil(distance / (self._resolution * 0.5))))
        for index in range(steps + 1):
            ratio = index / steps
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            if self.state_at_point(x, y) == CELL_OCCUPIED:
                # Allow the final sample to terminate on the target surface.
                if index == steps:
                    return True
                return False
        return True

    def count_visible_unknown(
        self,
        pose_xy_yaw: tuple[float, float, float],
        *,
        horizontal_fov: float = pi / 2.0,
        max_range: float = 5.0,
        ray_count: int = 21,
    ) -> int:
        """Count distinct unknown cells a bounded sensor cone would reveal."""
        if not isfinite(max_range) or max_range <= 0.0:
            raise ValueError('max_range must be finite and positive')
        if isinstance(ray_count, bool) or ray_count < 1:
            raise ValueError('ray_count must be a positive integer')
        x, y, yaw = pose_xy_yaw
        revealed: set[GridCell] = set()
        step = self._resolution * 0.5
        samples = max(1, int(ceil(max_range / step)))
        for index in range(ray_count):
            ratio = 0.5 if ray_count == 1 else index / (ray_count - 1)
            angle = yaw - horizontal_fov / 2.0 + horizontal_fov * ratio
            direction = (cos(angle), sin(angle))
            for sample in range(1, samples + 1):
                distance = sample * step
                cell = self.point_to_cell(
                    (x + direction[0] * distance, y + direction[1] * distance)
                )
                if not self.in_bounds(cell):
                    # Stop at the window edge rather than claiming gain from
                    # space this bounded raster never modelled.
                    break
                state = self.state(cell)
                if state == CELL_OCCUPIED:
                    break
                if state == CELL_UNKNOWN:
                    revealed.add(cell)
        return len(revealed)

    def frontier_cells(self) -> tuple[GridCell, ...]:
        """Return free cells that touch unknown space."""
        output = []
        for y_index in range(self._height):
            for x_index in range(self._width):
                if self._cells[y_index, x_index] != CELL_FREE:
                    continue
                if self._unknown_neighbours((x_index, y_index)):
                    output.append((x_index, y_index))
        return tuple(output)

    def frontier_clusters(
        self,
        *,
        minimum_cells: int = 3,
    ) -> tuple[FrontierCluster, ...]:
        """Group frontier cells and drop tiny isolated boundaries."""
        if isinstance(minimum_cells, bool) or minimum_cells < 1:
            raise ValueError('minimum_cells must be a positive integer')
        remaining = set(self.frontier_cells())
        clusters = []
        while remaining:
            seed = min(remaining)
            group = []
            queue = deque([seed])
            remaining.discard(seed)
            while queue:
                cell = queue.popleft()
                group.append(cell)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbour = (cell[0] + dx, cell[1] + dy)
                        if neighbour in remaining:
                            remaining.discard(neighbour)
                            queue.append(neighbour)
            if len(group) < minimum_cells:
                continue
            ordered = tuple(sorted(group))
            centres = [self.cell_centre(cell) for cell in ordered]
            centroid = (
                sum(item[0] for item in centres) / len(centres),
                sum(item[1] for item in centres) / len(centres),
            )
            clusters.append(
                FrontierCluster(
                    cluster_id=f'frontier_{len(clusters)}',
                    cells=ordered,
                    centroid_xy=centroid,
                    unknown_neighbour_count=sum(
                        self._unknown_neighbours(cell) for cell in ordered
                    ),
                )
            )
        return tuple(
            sorted(clusters, key=lambda item: (-item.size, item.cluster_id))
        )

    def _unknown_neighbours(self, cell: GridCell) -> int:
        # Only in-bounds neighbours count. The window edge is an artifact of
        # bounded rasterization, not a real explored/unexplored boundary, and
        # treating it as one would ring every map in phantom frontiers.
        x_index, y_index = cell
        total = 0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (x_index + dx, y_index + dy)
            if not self.in_bounds(neighbour):
                continue
            if self.state(neighbour) == CELL_UNKNOWN:
                total += 1
        return total


def _shift(mask: np.ndarray, delta_x: int, delta_y: int) -> np.ndarray:
    """Return ``mask`` translated by a cell offset, padding with False."""
    output = np.zeros_like(mask)
    height, width = mask.shape
    src_y = slice(max(0, -delta_y), height - max(0, delta_y))
    dst_y = slice(max(0, delta_y), height - max(0, -delta_y))
    src_x = slice(max(0, -delta_x), width - max(0, delta_x))
    dst_x = slice(max(0, delta_x), width - max(0, -delta_x))
    output[dst_y, dst_x] = mask[src_y, src_x]
    return output


def occupancy_from_scan_accumulator(
    accumulator,
    *,
    centre_xy: Point2D,
    half_extent_m: float = 8.0,
    resolution: float = 0.25,
    clearance: float = 0.25,
    robot_footprint_radius_m: float = 0.35,
) -> OccupancyGrid2D:
    """
    Rasterize a bounded window of the persistent scan map.

    Only a local window is rasterized so exploration reasoning stays bounded
    in both memory and runtime regardless of how large the episode map grows.
    """
    if not isfinite(half_extent_m) or half_extent_m <= 0.0:
        raise ValueError('half_extent_m must be finite and positive')
    if not isfinite(robot_footprint_radius_m) or (
        robot_footprint_radius_m < 0.0
    ):
        raise ValueError('robot_footprint_radius_m must be non-negative')
    size = max(1, int(ceil(2.0 * half_extent_m / resolution)))
    origin = (centre_xy[0] - half_extent_m, centre_xy[1] - half_extent_m)
    grid = OccupancyGrid2D(resolution, origin, size, size)
    points = accumulator.snapshot_points()
    config = getattr(accumulator, 'config', None)
    min_z = getattr(config, 'navigation_min_z', 0.10)
    max_z = getattr(config, 'navigation_max_z', 1.80)
    if len(points):
        navigation = points[
            (points[:, 2] > min_z) & (points[:, 2] <= max_z)
        ]
        for point in navigation:
            cell = grid.point_to_cell((float(point[0]), float(point[1])))
            if grid.in_bounds(cell):
                grid.set_state(cell, CELL_OCCUPIED)
    for x_index in range(size):
        for y_index in range(size):
            cell = (x_index, y_index)
            if grid.state(cell) == CELL_OCCUPIED:
                continue
            centre = grid.cell_centre(cell)
            if accumulator.is_known_free(
                centre[0], centre[1], clearance=clearance
            ):
                grid.set_state(cell, CELL_FREE)
    # The current robot footprint is observed free by construction.  Clearing
    # it after voxel rasterization prevents quantized ground returns at the
    # lower navigation boundary from trapping every planner at startup.
    half_diagonal = resolution * (2.0 ** 0.5) / 2.0
    clear_radius = robot_footprint_radius_m + half_diagonal
    for x_index in range(size):
        for y_index in range(size):
            cell = (x_index, y_index)
            cell_centre = grid.cell_centre(cell)
            if hypot(
                cell_centre[0] - centre_xy[0],
                cell_centre[1] - centre_xy[1],
            ) <= clear_radius:
                grid.set_state(cell, CELL_FREE)
    return grid


__all__ = [
    'CELL_FREE',
    'CELL_OCCUPIED',
    'CELL_UNKNOWN',
    'FrontierCluster',
    'GridCell',
    'OccupancyGrid2D',
    'Point2D',
    'occupancy_from_scan_accumulator',
]
