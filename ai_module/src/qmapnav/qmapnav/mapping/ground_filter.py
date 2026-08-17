"""Local map-frame ground estimation and candidate filtering."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array


@dataclass(frozen=True)
class GroundPlane:
    """Normalized upward-facing map-frame plane n dot p + d = 0."""

    normal_xyz: np.ndarray
    offset: float
    frame_id: str = 'map'
    timestamp_ns: int = 0
    confidence: float = 1.0
    source: str = 'estimated'

    def __post_init__(self) -> None:
        normal = readonly_array('normal_xyz', self.normal_xyz, (3,))
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError('ground normal must be non-zero')
        normal = normal / norm
        offset = float(self.offset) / norm
        if normal[2] < 0.0:
            normal = -normal
            offset = -offset
        normal.setflags(write=False)
        if normal[2] < 0.75:
            raise ValueError('ground plane must be predominantly horizontal')
        if not isfinite(offset):
            raise ValueError('ground offset must be finite')
        if not self.frame_id or not self.source:
            raise ValueError('ground frame and source must be non-empty')
        if self.timestamp_ns < 0:
            raise ValueError('ground timestamp must be non-negative')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('ground confidence must lie in [0, 1]')
        object.__setattr__(self, 'normal_xyz', normal)
        object.__setattr__(self, 'offset', offset)

    def signed_distance(self, points_xyz: np.ndarray) -> np.ndarray:
        """Return positive distance above the upward-facing plane."""
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3)')
        return points @ self.normal_xyz + self.offset


@dataclass(frozen=True)
class GroundEstimate:
    """Ground estimation result with support diagnostics."""

    plane: GroundPlane | None
    candidate_count: int
    inlier_count: int
    residual_median_m: float | None
    reason: str


@dataclass(frozen=True)
class GroundFilterResult:
    """Indices kept and removed by one ground-plane filter."""

    kept_indices: np.ndarray
    removed_indices: np.ndarray
    clearance_m: float
    plane_available: bool
    warning: str | None

    def __post_init__(self) -> None:
        for name in ('kept_indices', 'removed_indices'):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (None,), dtype=np.int64),
            )


def estimate_local_ground_plane(
    points_map_xyz: np.ndarray,
    *,
    timestamp_ns: int,
    sensor_position_xyz: np.ndarray | None = None,
    max_radius_m: float = 8.0,
    grid_size_m: float = 0.35,
    lower_quantile: float = 0.35,
    inlier_threshold_m: float = 0.05,
    minimum_candidates: int = 20,
) -> GroundEstimate:
    """Estimate floor from the lowest point in local XY cells."""
    points = np.asarray(points_map_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_map_xyz must have shape (N, 3)')
    points = points[np.all(np.isfinite(points), axis=1)]
    if sensor_position_xyz is not None and points.size:
        sensor = np.asarray(sensor_position_xyz, dtype=np.float64)
        if sensor.shape != (3,) or not np.all(np.isfinite(sensor)):
            raise ValueError('sensor_position_xyz must be finite shape (3,)')
        points = points[
            np.linalg.norm(points[:, :2] - sensor[:2], axis=1) <= max_radius_m
        ]
    if points.shape[0] < minimum_candidates:
        return GroundEstimate(None, points.shape[0], 0, None, 'insufficient_points')
    cells = np.floor(points[:, :2] / grid_size_m).astype(np.int64)
    order = np.lexsort((points[:, 2], cells[:, 1], cells[:, 0]))
    ordered_cells = cells[order]
    first = np.concatenate(
        ([True], np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1))
    )
    lowest = points[order[first]]
    z_limit = np.quantile(lowest[:, 2], lower_quantile)
    candidates = lowest[lowest[:, 2] <= z_limit + inlier_threshold_m]
    if candidates.shape[0] < minimum_candidates:
        candidates = lowest[np.argsort(lowest[:, 2])[:minimum_candidates]]
    design = np.column_stack((candidates[:, 0], candidates[:, 1], np.ones(len(candidates))))
    coefficients, _, _, _ = np.linalg.lstsq(design, candidates[:, 2], rcond=None)
    for _ in range(3):
        residual = candidates[:, 2] - design @ coefficients
        inliers = np.abs(residual) <= inlier_threshold_m
        if np.count_nonzero(inliers) < max(3, minimum_candidates // 2):
            break
        coefficients, _, _, _ = np.linalg.lstsq(
            design[inliers], candidates[inliers, 2], rcond=None
        )
    residual = candidates[:, 2] - design @ coefficients
    inliers = np.abs(residual) <= inlier_threshold_m
    inlier_count = int(np.count_nonzero(inliers))
    median = float(np.median(np.abs(residual[inliers]))) if inlier_count else None
    a, b, c = coefficients
    normal = np.array([-a, -b, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    confidence = min(1.0, inlier_count / max(minimum_candidates * 2, 1))
    if median is not None:
        confidence *= float(np.exp(-median / max(inlier_threshold_m, 1e-6)))
    try:
        plane = GroundPlane(
            normal_xyz=normal,
            offset=-c / np.linalg.norm(np.array([-a, -b, 1.0])),
            timestamp_ns=timestamp_ns,
            confidence=confidence,
            source='local_lowest_cell_fit',
        )
    except ValueError:
        return GroundEstimate(None, len(candidates), inlier_count, median, 'implausible_plane')
    return GroundEstimate(plane, len(candidates), inlier_count, median, 'estimated')


def remove_ground_points(
    points_map_xyz: np.ndarray,
    plane: GroundPlane | None,
    *,
    clearance_m: float = 0.07,
    minimum_plane_confidence: float = 0.2,
) -> GroundFilterResult:
    """Remove points within clearance above a trusted ground plane."""
    points = np.asarray(points_map_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_map_xyz must have shape (N, 3)')
    if not isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError('clearance_m must be finite and non-negative')
    indices = np.arange(points.shape[0], dtype=np.int64)
    if plane is None or plane.confidence < minimum_plane_confidence:
        warning = 'ground_plane_unavailable' if plane is None else 'ground_plane_low_confidence'
        return GroundFilterResult(
            indices,
            np.empty(0, dtype=np.int64),
            clearance_m,
            False,
            warning,
        )
    above = plane.signed_distance(points) > clearance_m
    return GroundFilterResult(
        indices[above], indices[~above], clearance_m, True, None
    )


__all__ = [
    'estimate_local_ground_plane',
    'GroundEstimate',
    'GroundFilterResult',
    'GroundPlane',
    'remove_ground_points',
]
