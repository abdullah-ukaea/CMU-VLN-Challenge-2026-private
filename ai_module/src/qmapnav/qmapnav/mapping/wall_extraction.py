"""Deterministic vertical-support wall extraction and safe line merging."""

from dataclasses import dataclass
from math import atan2
from math import isfinite
from math import pi

import numpy as np

from qmapnav.mapping.bounding_boxes import wrap_half_pi
from qmapnav.mapping.object_candidate import readonly_array


@dataclass(frozen=True)
class WallExtractionConfig:
    """Measured starting policy for bounded deterministic wall extraction."""

    min_height_above_ground_m: float = 0.20
    max_height_above_ground_m: float = 3.50
    vertical_cell_size_m: float = 0.10
    minimum_vertical_coverage_m: float = 0.40
    minimum_support_points: int = 24
    min_segment_length_m: float = 0.50
    max_line_residual_m: float = 0.08
    merge_angle_deg: float = 5.0
    merge_perpendicular_distance_m: float = 0.12
    preserve_opening_width_m: float = 0.60
    maximum_segments: int = 32
    ransac_iterations: int = 256
    max_candidate_points: int = 50_000

    def __post_init__(self) -> None:
        for name in (
            'min_height_above_ground_m',
            'max_height_above_ground_m',
            'vertical_cell_size_m',
            'minimum_vertical_coverage_m',
            'min_segment_length_m',
            'max_line_residual_m',
            'merge_angle_deg',
            'merge_perpendicular_distance_m',
            'preserve_opening_width_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.max_height_above_ground_m <= self.min_height_above_ground_m:
            raise ValueError('wall height interval is invalid')
        for name in (
            'minimum_support_points', 'maximum_segments', 'ransac_iterations',
            'max_candidate_points',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class WallCandidate:
    """One map-frame vertical wall segment with fit evidence."""

    line_segment_xy: np.ndarray
    plane_parameters: np.ndarray
    vertical_extent_m: tuple[float, float]
    yaw_rad: float
    supporting_point_count: int
    fit_residual_m: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'line_segment_xy',
            readonly_array('line_segment_xy', self.line_segment_xy, (2, 2)),
        )
        object.__setattr__(
            self,
            'plane_parameters',
            readonly_array('plane_parameters', self.plane_parameters, (4,)),
        )
        if self.vertical_extent_m[1] <= self.vertical_extent_m[0]:
            raise ValueError('wall vertical extent must be increasing')
        if self.supporting_point_count <= 0:
            raise ValueError('wall support count must be positive')
        if not isfinite(self.fit_residual_m) or self.fit_residual_m < 0.0:
            raise ValueError('wall fit residual must be non-negative')
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('wall confidence must lie in [0, 1]')

    @property
    def length_m(self) -> float:
        """Return the map-XY segment length."""
        return float(np.linalg.norm(
            self.line_segment_xy[1] - self.line_segment_xy[0]
        ))


def extract_wall_candidates(
    points_map_xyz: np.ndarray,
    *,
    timestamp_ns: int,
    config: WallExtractionConfig | None = None,
) -> tuple[WallCandidate, ...]:
    """Extract bounded wall segments from vertically supported map points."""
    policy = config or WallExtractionConfig()
    points = np.asarray(points_map_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_map_xyz must have shape (N, 3)')
    if timestamp_ns < 0:
        raise ValueError('timestamp_ns must be non-negative')
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < policy.minimum_support_points:
        return ()
    ground_z = float(np.quantile(points[:, 2], 0.05))
    height = points[:, 2] - ground_z
    above = points[
        (height >= policy.min_height_above_ground_m)
        & (height <= policy.max_height_above_ground_m)
    ]
    supported = _vertical_support(above, policy)
    if supported.shape[0] < policy.minimum_support_points:
        return ()
    if supported.shape[0] > policy.max_candidate_points:
        indices = np.linspace(
            0,
            supported.shape[0] - 1,
            policy.max_candidate_points,
            dtype=np.int64,
        )
        supported = supported[indices]
    remaining = supported.copy()
    extracted = []
    random = np.random.default_rng(0)
    while (
        remaining.shape[0] >= policy.minimum_support_points
        and len(extracted) < policy.maximum_segments
    ):
        fit = _best_line_fit(remaining, policy, random)
        if fit is None:
            break
        origin, direction, inliers, residual = fit
        support = remaining[inliers]
        fragments = _split_support(origin, direction, support, policy)
        for fragment in fragments:
            candidate = _candidate_from_support(
                origin, direction, fragment, residual, timestamp_ns, policy
            )
            if candidate is not None:
                extracted.append(candidate)
        remaining = remaining[~inliers]
    return tuple(merge_wall_candidates(extracted, config=policy))


def merge_wall_candidates(
    candidates: list[WallCandidate] | tuple[WallCandidate, ...],
    *,
    config: WallExtractionConfig | None = None,
) -> list[WallCandidate]:
    """Merge compatible collinear segments without crossing doorway gaps."""
    policy = config or WallExtractionConfig()
    merged = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            round(item.yaw_rad, 6),
            tuple(item.line_segment_xy[0]),
        ),
    ):
        match_index = None
        for index, existing in enumerate(merged):
            if _segments_mergeable(existing, candidate, policy):
                match_index = index
                break
        if match_index is None:
            merged.append(candidate)
        else:
            merged[match_index] = _merge_pair(
                merged[match_index], candidate
            )
    return merged


def _vertical_support(
    points: np.ndarray,
    config: WallExtractionConfig,
) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    keys = np.floor(
        points[:, :2] / config.vertical_cell_size_m
    ).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    coverage = np.zeros(int(np.max(inverse)) + 1, dtype=np.float64)
    minimum = np.full(coverage.shape, np.inf)
    maximum = np.full(coverage.shape, -np.inf)
    np.minimum.at(minimum, inverse, points[:, 2])
    np.maximum.at(maximum, inverse, points[:, 2])
    coverage[:] = maximum - minimum
    return points[coverage[inverse] >= config.minimum_vertical_coverage_m]


def _best_line_fit(points, config, random):
    best = None
    count = points.shape[0]
    for _ in range(config.ransac_iterations):
        indices = random.choice(count, size=2, replace=False)
        first, second = points[indices, :2]
        delta = second - first
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            continue
        direction = delta / norm
        normal = np.array([-direction[1], direction[0]])
        distances = np.abs((points[:, :2] - first) @ normal)
        inliers = distances <= config.max_line_residual_m
        support_count = int(np.count_nonzero(inliers))
        if support_count < config.minimum_support_points:
            continue
        support = points[inliers, :2]
        length = float(np.ptp((support - first) @ direction))
        residual = float(np.median(distances[inliers]))
        key = (support_count, length, -residual)
        if best is None or key > best[0]:
            best = (key, first, direction, inliers, residual)
    if best is None:
        return None
    _, origin, direction, inliers, residual = best
    support_xy = points[inliers, :2]
    centred = support_xy - np.mean(support_xy, axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    if direction[0] < 0.0 or (
        np.isclose(direction[0], 0.0) and direction[1] < 0.0
    ):
        direction = -direction
    origin = np.mean(support_xy, axis=0)
    normal = np.array([-direction[1], direction[0]])
    distances = np.abs((points[:, :2] - origin) @ normal)
    inliers = distances <= config.max_line_residual_m
    residual = float(np.median(distances[inliers]))
    return origin, direction, inliers, residual


def _split_support(origin, direction, support, config):
    projection = (support[:, :2] - origin) @ direction
    ordered = np.argsort(projection)
    sorted_projection = projection[ordered]
    gaps = np.flatnonzero(
        np.diff(sorted_projection) >= config.preserve_opening_width_m
    ) + 1
    return [support[indices] for indices in np.split(ordered, gaps)]


def _candidate_from_support(
    origin,
    direction,
    support,
    residual,
    timestamp_ns,
    config,
):
    if support.shape[0] < config.minimum_support_points:
        return None
    projection = (support[:, :2] - origin) @ direction
    endpoints = np.vstack((
        origin + np.min(projection) * direction,
        origin + np.max(projection) * direction,
    ))
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    vertical = (float(np.min(support[:, 2])), float(np.max(support[:, 2])))
    vertical_coverage = vertical[1] - vertical[0]
    if (
        length < config.min_segment_length_m
        or vertical_coverage < config.minimum_vertical_coverage_m
    ):
        return None
    normal_xy = np.array([-direction[1], direction[0]])
    plane = np.array([
        normal_xy[0],
        normal_xy[1],
        0.0,
        -float(normal_xy @ np.mean(endpoints, axis=0)),
    ])
    support_score = min(1.0, support.shape[0] / 100.0)
    length_score = min(1.0, length / 2.0)
    residual_score = np.exp(
        -residual / max(config.max_line_residual_m, 1e-9)
    )
    vertical_score = min(1.0, vertical_coverage / 1.5)
    confidence = float(
        0.25 * support_score
        + 0.30 * length_score
        + 0.25 * residual_score
        + 0.20 * vertical_score
    )
    return WallCandidate(
        endpoints,
        plane,
        vertical,
        wrap_half_pi(atan2(direction[1], direction[0])),
        int(support.shape[0]),
        residual,
        confidence,
        timestamp_ns,
    )


def _segments_mergeable(first, second, config):
    angle = abs(first.yaw_rad - second.yaw_rad)
    angle = min(angle, pi - angle)
    if angle > np.deg2rad(config.merge_angle_deg):
        return False
    direction = first.line_segment_xy[1] - first.line_segment_xy[0]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    perpendicular = max(abs(
        (point - first.line_segment_xy[0]) @ normal
    ) for point in second.line_segment_xy)
    if perpendicular > config.merge_perpendicular_distance_m:
        return False
    first_t = sorted(
        (first.line_segment_xy - first.line_segment_xy[0]) @ direction
    )
    second_t = sorted(
        (second.line_segment_xy - first.line_segment_xy[0]) @ direction
    )
    gap = max(0.0, second_t[0] - first_t[1], first_t[0] - second_t[1])
    return gap < config.preserve_opening_width_m


def _merge_pair(first, second):
    direction = first.line_segment_xy[1] - first.line_segment_xy[0]
    direction /= np.linalg.norm(direction)
    points = np.vstack((first.line_segment_xy, second.line_segment_xy))
    origin = np.mean(points, axis=0)
    projection = (points - origin) @ direction
    endpoints = np.vstack((
        origin + np.min(projection) * direction,
        origin + np.max(projection) * direction,
    ))
    normal = np.array([-direction[1], direction[0]])
    plane = np.array([
        normal[0], normal[1], 0.0, -float(normal @ np.mean(endpoints, axis=0))
    ])
    count = first.supporting_point_count + second.supporting_point_count
    residual = (
        first.fit_residual_m * first.supporting_point_count
        + second.fit_residual_m * second.supporting_point_count
    ) / count
    return WallCandidate(
        endpoints,
        plane,
        (
            min(first.vertical_extent_m[0], second.vertical_extent_m[0]),
            max(first.vertical_extent_m[1], second.vertical_extent_m[1]),
        ),
        first.yaw_rad,
        count,
        residual,
        max(first.confidence, second.confidence),
        max(first.timestamp_ns, second.timestamp_ns),
    )


__all__ = [
    'extract_wall_candidates',
    'merge_wall_candidates',
    'WallCandidate',
    'WallExtractionConfig',
]
