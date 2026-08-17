"""Robust AABB and upright minimum-area OBB estimation."""

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array


@dataclass(frozen=True)
class BoxEstimationConfig:
    """Robust extent and plausibility policy."""

    lower_percentile: float = 2.5
    upper_percentile: float = 97.5
    min_dimension_m: float = 0.01
    max_dimension_m: float = 12.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError('box percentiles are invalid')
        if not 0.0 < self.min_dimension_m < self.max_dimension_m:
            raise ValueError('box dimension limits are invalid')


@dataclass(frozen=True)
class AxisAlignedBox:
    """Robust map-frame axis-aligned box."""

    minimum_xyz: np.ndarray
    maximum_xyz: np.ndarray

    def __post_init__(self) -> None:
        minimum = readonly_array('minimum_xyz', self.minimum_xyz, (3,))
        maximum = readonly_array('maximum_xyz', self.maximum_xyz, (3,))
        if np.any(maximum <= minimum):
            raise ValueError('AABB maximum must exceed minimum')
        object.__setattr__(self, 'minimum_xyz', minimum)
        object.__setattr__(self, 'maximum_xyz', maximum)

    @property
    def centre_xyz(self) -> np.ndarray:
        """Return the box midpoint."""
        return (self.minimum_xyz + self.maximum_xyz) / 2.0

    @property
    def dimensions_xyz(self) -> np.ndarray:
        """Return map-axis dimensions."""
        return self.maximum_xyz - self.minimum_xyz


@dataclass(frozen=True)
class UprightOrientedBox:
    """Upright map-frame box with yaw around positive Z."""

    centre_xyz: np.ndarray
    dimensions_xyz: np.ndarray
    yaw_rad: float
    pca_yaw_rad: float
    pca_eigenvalues: np.ndarray
    raw_dimensions_xyz: np.ndarray

    def __post_init__(self) -> None:
        centre = readonly_array('centre_xyz', self.centre_xyz, (3,))
        dimensions = readonly_array('dimensions_xyz', self.dimensions_xyz, (3,))
        raw = readonly_array('raw_dimensions_xyz', self.raw_dimensions_xyz, (3,))
        eigenvalues = readonly_array('pca_eigenvalues', self.pca_eigenvalues, (2,))
        if np.any(dimensions <= 0.0) or np.any(raw <= 0.0):
            raise ValueError('OBB dimensions must be positive')
        if dimensions[0] + 1e-12 < dimensions[1]:
            raise ValueError('canonical OBB length must be at least width')
        if not -pi / 2.0 <= self.yaw_rad < pi / 2.0:
            raise ValueError('canonical yaw must be in [-pi/2, pi/2)')
        if not isfinite(self.pca_yaw_rad):
            raise ValueError('PCA yaw must be finite')
        object.__setattr__(self, 'centre_xyz', centre)
        object.__setattr__(self, 'dimensions_xyz', dimensions)
        object.__setattr__(self, 'raw_dimensions_xyz', raw)
        object.__setattr__(self, 'pca_eigenvalues', eigenvalues)


def wrap_half_pi(yaw_rad: float) -> float:
    """Wrap a rectangle yaw to the canonical half-open pi interval."""
    yaw = (float(yaw_rad) + pi / 2.0) % pi - pi / 2.0
    if np.isclose(yaw, pi / 2.0, atol=1e-12):
        yaw = -pi / 2.0
    if np.isclose(yaw, 0.0, atol=1e-15):
        yaw = 0.0
    return float(yaw)


def canonicalize_box(
    length_m: float,
    width_m: float,
    yaw_rad: float,
) -> tuple[float, float, float]:
    """Return length >= width and the yaw of the length axis."""
    length = float(length_m)
    width = float(width_m)
    yaw = float(yaw_rad)
    if not all(isfinite(value) and value > 0.0 for value in (length, width)):
        raise ValueError('box dimensions must be finite and positive')
    if not isfinite(yaw):
        raise ValueError('yaw must be finite')
    if width > length:
        length, width = width, length
        yaw += pi / 2.0
    return length, width, wrap_half_pi(yaw)


def rectangle_yaw_difference(first: float, second: float) -> float:
    """Return the smallest yaw error under 180-degree box symmetry."""
    delta = abs(wrap_half_pi(float(first) - float(second)))
    return float(min(delta, pi - delta))


def robust_aabb(
    points_xyz: np.ndarray,
    config: BoxEstimationConfig | None = None,
) -> AxisAlignedBox:
    """Fit a percentile-trimmed AABB."""
    policy = config or BoxEstimationConfig()
    points = _valid_points(points_xyz, minimum=2)
    minimum = np.percentile(points, policy.lower_percentile, axis=0)
    maximum = np.percentile(points, policy.upper_percentile, axis=0)
    dimensions = maximum - minimum
    _validate_dimensions(dimensions, policy)
    return AxisAlignedBox(minimum, maximum)


def estimate_upright_obb(
    points_xyz: np.ndarray,
    config: BoxEstimationConfig | None = None,
) -> UprightOrientedBox:
    """Fit a robust upright minimum-area rectangle plus vertical extent."""
    policy = config or BoxEstimationConfig()
    points = _valid_points(points_xyz, minimum=3)
    trim_min = np.percentile(points, policy.lower_percentile, axis=0)
    trim_max = np.percentile(points, policy.upper_percentile, axis=0)
    trimmed = points[np.all((points >= trim_min) & (points <= trim_max), axis=1)]
    if trimmed.shape[0] < 3:
        trimmed = points
    yaw = _minimum_area_yaw(trimmed[:, :2])
    axis = np.array([cos(yaw), sin(yaw)], dtype=np.float64)
    perpendicular = np.array([-sin(yaw), cos(yaw)], dtype=np.float64)
    basis = np.column_stack((axis, perpendicular))
    local_xy = points[:, :2] @ basis
    lower = np.percentile(local_xy, policy.lower_percentile, axis=0)
    upper = np.percentile(local_xy, policy.upper_percentile, axis=0)
    centre_local = (lower + upper) / 2.0
    horizontal = upper - lower
    raw_min = np.min(local_xy, axis=0)
    raw_max = np.max(local_xy, axis=0)
    z_min, z_max = np.percentile(
        points[:, 2],
        (policy.lower_percentile, policy.upper_percentile),
    )
    raw_z_min, raw_z_max = np.min(points[:, 2]), np.max(points[:, 2])
    length, width, canonical_yaw = canonicalize_box(
        horizontal[0], horizontal[1], yaw
    )
    if not np.isclose(canonical_yaw, yaw, atol=1e-10):
        centre_local = np.array([centre_local[1], -centre_local[0]])
        basis = np.column_stack((perpendicular, -axis))
        raw_horizontal = (raw_max - raw_min)[::-1]
    else:
        raw_horizontal = raw_max - raw_min
    centre_xy = basis @ centre_local
    dimensions = np.array([length, width, z_max - z_min], dtype=np.float64)
    _validate_dimensions(dimensions, policy)
    pca_yaw, eigenvalues = pca_footprint_yaw(points[:, :2])
    return UprightOrientedBox(
        centre_xyz=np.array([centre_xy[0], centre_xy[1], (z_min + z_max) / 2.0]),
        dimensions_xyz=dimensions,
        yaw_rad=canonical_yaw,
        pca_yaw_rad=pca_yaw,
        pca_eigenvalues=eigenvalues,
        raw_dimensions_xyz=np.array(
            [raw_horizontal[0], raw_horizontal[1], raw_z_max - raw_z_min]
        ),
    )


def pca_footprint_yaw(points_xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Return canonical PCA major-axis yaw and ascending eigenvalues."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError('points_xy must have shape (N, 2), N >= 2')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_xy must be finite')
    covariance = np.cov(points - np.mean(points, axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    return wrap_half_pi(atan2(major[1], major[0])), eigenvalues


def _minimum_area_yaw(points_xy: np.ndarray) -> float:
    hull = _convex_hull(points_xy)
    if hull.shape[0] < 2:
        raise ValueError('footprint has no horizontal extent')
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    edges = edges[lengths > 1e-9]
    if not edges.size:
        raise ValueError('footprint is horizontally degenerate')
    angles = np.unique(
        np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), pi / 2.0), 12)
    )
    best = None
    for angle in angles:
        axis = np.array([cos(angle), sin(angle)])
        perpendicular = np.array([-sin(angle), cos(angle)])
        local = hull @ np.column_stack((axis, perpendicular))
        extents = np.ptp(local, axis=0)
        area = float(extents[0] * extents[1])
        candidate = (area, float(angle))
        if best is None or candidate < best:
            best = candidate
    return float(best[1])


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points_xy, dtype=np.float64), axis=0)
    if unique.shape[0] <= 2:
        return unique
    ordered = sorted(map(tuple, unique.tolist()))

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _valid_points(points_xyz: np.ndarray, *, minimum: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_xyz must have shape (N, 3)')
    if points.shape[0] < minimum:
        raise ValueError(f'at least {minimum} points are required')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_xyz must contain only finite values')
    return points


def _validate_dimensions(
    dimensions: np.ndarray,
    policy: BoxEstimationConfig,
) -> None:
    if np.any(~np.isfinite(dimensions)):
        raise ValueError('box dimensions must be finite')
    if np.any(dimensions < policy.min_dimension_m):
        raise ValueError('box dimensions are degenerate')
    if np.any(dimensions > policy.max_dimension_m):
        raise ValueError('box dimensions are implausibly large')


__all__ = [
    'AxisAlignedBox',
    'BoxEstimationConfig',
    'UprightOrientedBox',
    'canonicalize_box',
    'estimate_upright_obb',
    'pca_footprint_yaw',
    'rectangle_yaw_difference',
    'robust_aabb',
    'wrap_half_pi',
]
