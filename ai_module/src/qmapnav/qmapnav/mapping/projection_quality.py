"""Geometric support summaries inside wrap-aware Day 4 detections."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


@dataclass(frozen=True)
class ProjectionQualityConfig:
    """Thresholds for non-semantic projected-point support quality."""

    sparse_point_threshold: int = 8
    high_depth_iqr_m: float = 2.0
    timing_warning_ms: float = 100.0
    occupancy_columns: int = 8
    occupancy_rows: int = 6

    def __post_init__(self) -> None:
        for name in ('sparse_point_threshold', 'occupancy_columns', 'occupancy_rows'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('high_depth_iqr_m', 'timing_warning_ms'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class DetectionProjection:
    """Projected geometric support inside one 2D detection region."""

    detection_id: str
    projection_indices: np.ndarray
    point_count: int
    depth_min_m: float | None
    depth_median_m: float | None
    depth_max_m: float | None
    depth_iqr_m: float | None
    occupied_cell_fraction: float
    image_scan_delta_ms: float
    pose_before_delta_ms: float | None
    pose_after_delta_ms: float | None
    quality: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.projection_indices, dtype=np.int64).copy()
        if indices.ndim != 1 or indices.shape[0] != self.point_count:
            raise ValueError('projection_indices and point_count disagree')
        indices.setflags(write=False)
        object.__setattr__(self, 'projection_indices', indices)
        allowed = {
            'good',
            'sparse',
            'high_depth_spread',
            'timing_warning',
            'no_points',
        }
        if self.quality not in allowed:
            raise ValueError('unsupported projection quality')
        if not 0.0 <= self.occupied_cell_fraction <= 1.0:
            raise ValueError('occupied_cell_fraction must lie in [0, 1]')


def panorama_box_mask(uv: np.ndarray, box: PanoramaBox) -> np.ndarray:
    """Return point membership for ordinary or seam-split panorama boxes."""
    pixels = np.asarray(uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('uv must have shape (N, 2)')
    vertical = (pixels[:, 1] >= box.y_min) & (pixels[:, 1] <= box.y_max)
    horizontal = np.zeros(pixels.shape[0], dtype=np.bool_)
    for x_min, x_max in box.x_intervals:
        horizontal |= (pixels[:, 0] >= x_min) & (pixels[:, 0] <= x_max)
    return vertical & horizontal


def summarize_detection_projection(
    detection: Detection2D,
    projection: ProjectionResult,
    config: ProjectionQualityConfig | None = None,
) -> DetectionProjection:
    """Summarize map points whose projections fall inside one detection."""
    policy = config or ProjectionQualityConfig()
    mask = panorama_box_mask(projection.panorama_uv, detection.panorama_box)
    indices = np.flatnonzero(mask)
    count = int(indices.shape[0])
    delta = projection.diagnostics.image_scan_delta_ms
    warnings = []
    if abs(delta) > policy.timing_warning_ms:
        warnings.append('timing_warning')
    if count == 0:
        return DetectionProjection(
            detection_id=detection.detection_id,
            projection_indices=indices,
            point_count=0,
            depth_min_m=None,
            depth_median_m=None,
            depth_max_m=None,
            depth_iqr_m=None,
            occupied_cell_fraction=0.0,
            image_scan_delta_ms=delta,
            pose_before_delta_ms=projection.diagnostics.pose_before_delta_ms,
            pose_after_delta_ms=projection.diagnostics.pose_after_delta_ms,
            quality='no_points',
            warnings=tuple(warnings),
        )
    depths = projection.euclidean_range_m[indices]
    q25, median, q75 = np.percentile(depths, (25.0, 50.0, 75.0))
    iqr = float(q75 - q25)
    if count < policy.sparse_point_threshold:
        quality = 'sparse'
        warnings.append('sparse')
    elif iqr > policy.high_depth_iqr_m:
        quality = 'high_depth_spread'
        warnings.append('high_depth_spread')
    elif 'timing_warning' in warnings:
        quality = 'timing_warning'
    else:
        quality = 'good'
    occupied = _occupied_cell_fraction(
        projection.panorama_uv[indices],
        detection.panorama_box,
        policy.occupancy_columns,
        policy.occupancy_rows,
    )
    return DetectionProjection(
        detection_id=detection.detection_id,
        projection_indices=indices,
        point_count=count,
        depth_min_m=float(np.min(depths)),
        depth_median_m=float(median),
        depth_max_m=float(np.max(depths)),
        depth_iqr_m=iqr,
        occupied_cell_fraction=occupied,
        image_scan_delta_ms=delta,
        pose_before_delta_ms=projection.diagnostics.pose_before_delta_ms,
        pose_after_delta_ms=projection.diagnostics.pose_after_delta_ms,
        quality=quality,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def summarize_detections(
    detections: tuple[Detection2D, ...],
    projection: ProjectionResult,
    config: ProjectionQualityConfig | None = None,
) -> tuple[DetectionProjection, ...]:
    """Summarize projected support for every final Day 4 detection."""
    return tuple(
        summarize_detection_projection(detection, projection, config)
        for detection in detections
    )


def _occupied_cell_fraction(
    uv: np.ndarray,
    box: PanoramaBox,
    columns: int,
    rows: int,
) -> float:
    widths = [x_max - x_min for x_min, x_max in box.x_intervals]
    total_width = sum(widths)
    height = box.y_max - box.y_min
    if total_width <= 0.0 or height <= 0.0 or uv.shape[0] == 0:
        return 0.0
    local_x = np.empty(uv.shape[0], dtype=np.float64)
    offset = 0.0
    assigned = np.zeros(uv.shape[0], dtype=np.bool_)
    for (x_min, x_max), width in zip(box.x_intervals, widths):
        member = (uv[:, 0] >= x_min) & (uv[:, 0] <= x_max) & ~assigned
        local_x[member] = offset + (uv[member, 0] - x_min)
        assigned |= member
        offset += width
    x_cell = np.minimum(
        columns - 1,
        np.floor(local_x / total_width * columns).astype(np.int64),
    )
    y_cell = np.minimum(
        rows - 1,
        np.floor((uv[:, 1] - box.y_min) / height * rows).astype(np.int64),
    )
    occupied = np.unique(y_cell * columns + x_cell).shape[0]
    return float(occupied / (columns * rows))


__all__ = [
    'DetectionProjection',
    'ProjectionQualityConfig',
    'panorama_box_mask',
    'summarize_detection_projection',
    'summarize_detections',
]
