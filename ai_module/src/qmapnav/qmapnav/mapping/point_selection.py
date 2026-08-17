"""Wrap-aware box and optional segmentation-mask point selection."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

import numpy as np

from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import readonly_array
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


class SelectionMode(str, Enum):
    """Image-region evidence used to select projected points."""

    BOX = 'box'
    MASK = 'mask'
    BOX_FALLBACK = 'box_fallback'


@dataclass(frozen=True)
class PointSelectionConfig:
    """Box contraction, mask, and boundary diagnostic policy."""

    bbox_inner_margin_fraction: float = 0.05
    minimum_box_size_px: float = 2.0
    mask_threshold: float = 0.5
    boundary_band_fraction: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.bbox_inner_margin_fraction < 0.5:
            raise ValueError('bbox_inner_margin_fraction must lie in [0, 0.5)')
        if not isfinite(self.minimum_box_size_px) or self.minimum_box_size_px <= 0.0:
            raise ValueError('minimum_box_size_px must be positive')
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError('mask_threshold must lie in [0, 1]')
        if not 0.0 <= self.boundary_band_fraction < 0.5:
            raise ValueError('boundary_band_fraction must lie in [0, 0.5)')


@dataclass(frozen=True)
class PointSelectionResult:
    """Immutable indices retained by original box, contraction and mask."""

    mode: SelectionMode
    original_box_indices: np.ndarray
    contracted_box_indices: np.ndarray
    selected_projection_indices: np.ndarray
    boundary_fraction: float
    mask_available: bool
    reason: str

    def __post_init__(self) -> None:
        for name in (
            'original_box_indices',
            'contracted_box_indices',
            'selected_projection_indices',
        ):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (None,), dtype=np.int64),
            )
        if not 0.0 <= self.boundary_fraction <= 1.0:
            raise ValueError('boundary_fraction must lie in [0, 1]')
        if not self.reason:
            raise ValueError('reason must be non-empty')


def select_detection_points(
    detection: Detection2D,
    projection: ProjectionResult,
    *,
    use_mask: bool = True,
    mask: np.ndarray | None = None,
    mask_polygons_uv: Iterable[np.ndarray] | None = None,
    config: PointSelectionConfig | None = None,
) -> PointSelectionResult:
    """Select projected points using a mask when valid, else contracted box."""
    policy = config or PointSelectionConfig()
    uv = projection.panorama_uv
    original_mask = points_inside_panorama_box(uv, detection.panorama_box)
    contracted_mask = points_inside_panorama_box(
        uv,
        detection.panorama_box,
        margin_fraction=policy.bbox_inner_margin_fraction,
        minimum_size_px=policy.minimum_box_size_px,
    )
    original_indices = np.flatnonzero(original_mask)
    box_indices = np.flatnonzero(contracted_mask)
    supplied_polygons = tuple(mask_polygons_uv or ())
    if not supplied_polygons:
        supplied_polygons = _metadata_polygons(detection)
    mask_available = mask is not None or bool(supplied_polygons)
    selected = box_indices
    if use_mask and mask is not None and _valid_mask(mask, detection.panorama_box):
        image_mask = np.asarray(mask, dtype=np.float64)
        pixels = np.rint(uv).astype(np.int64)
        x = np.mod(pixels[:, 0], detection.panorama_box.panorama_width)
        y = np.clip(pixels[:, 1], 0, detection.panorama_box.panorama_height - 1)
        inside_mask = image_mask[y, x] >= policy.mask_threshold
        selected = np.flatnonzero(original_mask & inside_mask)
        mode = SelectionMode.MASK
        reason = 'valid_panorama_mask'
    elif use_mask and supplied_polygons and all(
        _valid_polygon(value) for value in supplied_polygons
    ):
        inside_mask = points_inside_panorama_polygons(
            uv,
            supplied_polygons,
            panorama_width=detection.panorama_box.panorama_width,
            centre_u=detection.centre_panorama_uv[0],
        )
        selected = np.flatnonzero(original_mask & inside_mask)
        mode = SelectionMode.MASK
        reason = 'valid_detector_mask_polygon'
    elif use_mask and mask_available:
        mode = SelectionMode.BOX_FALLBACK
        reason = 'invalid_mask_used_contracted_box'
    else:
        mode = SelectionMode.BOX
        reason = 'contracted_box'
    boundary_fraction = _boundary_fraction(
        uv[selected], detection.panorama_box, policy.boundary_band_fraction
    )
    return PointSelectionResult(
        mode=mode,
        original_box_indices=original_indices,
        contracted_box_indices=box_indices,
        selected_projection_indices=selected,
        boundary_fraction=boundary_fraction,
        mask_available=mask_available,
        reason=reason,
    )


def points_inside_panorama_box(
    uv: np.ndarray,
    box: PanoramaBox,
    *,
    margin_fraction: float = 0.0,
    minimum_size_px: float = 1.0,
) -> np.ndarray:
    """Return vectorized membership in an optionally contracted circular box."""
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('uv must have shape (N, 2)')
    if not np.all(np.isfinite(points)):
        raise ValueError('uv must contain finite values')
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError('margin_fraction must lie in [0, 0.5)')
    start, width = _circular_start_width(box)
    horizontal_margin = min(
        width * margin_fraction,
        max(0.0, (width - minimum_size_px) / 2.0),
    )
    height = box.y_max - box.y_min
    vertical_margin = min(
        height * margin_fraction,
        max(0.0, (height - minimum_size_px) / 2.0),
    )
    offsets = np.mod(points[:, 0] - start, box.panorama_width)
    horizontal = (
        (offsets >= horizontal_margin)
        & (offsets <= width - horizontal_margin)
    )
    vertical = (
        (points[:, 1] >= box.y_min + vertical_margin)
        & (points[:, 1] <= box.y_max - vertical_margin)
    )
    return horizontal & vertical


def points_inside_panorama_polygons(
    uv: np.ndarray,
    polygons_uv: Iterable[np.ndarray],
    *,
    panorama_width: int,
    centre_u: float,
) -> np.ndarray:
    """Return union membership in seam-safe detector-mask polygons."""
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('uv must have shape (N, 2)')
    unwrapped_points = points.copy()
    unwrapped_points[:, 0] = _unwrap_u(points[:, 0], centre_u, panorama_width)
    inside = np.zeros(points.shape[0], dtype=np.bool_)
    for polygon in polygons_uv:
        vertices = np.asarray(polygon, dtype=np.float64)
        if not _valid_polygon(vertices):
            raise ValueError('mask polygons must have shape (N, 2), N >= 3')
        vertices = vertices.copy()
        vertices[:, 0] = _unwrap_u(vertices[:, 0], centre_u, panorama_width)
        inside |= _points_in_polygon(unwrapped_points, vertices)
    return inside


def _circular_start_width(box: PanoramaBox) -> tuple[float, float]:
    if len(box.x_intervals) == 1:
        start, end = box.x_intervals[0]
        return start, end - start
    left, right = box.x_intervals
    return right[0], (box.panorama_width - right[0]) + left[1]


def _boundary_fraction(
    uv: np.ndarray,
    box: PanoramaBox,
    band_fraction: float,
) -> float:
    if uv.shape[0] == 0:
        return 0.0
    start, width = _circular_start_width(box)
    height = box.y_max - box.y_min
    offsets = np.mod(uv[:, 0] - start, box.panorama_width)
    x_band = width * band_fraction
    y_band = height * band_fraction
    boundary = (
        (offsets <= x_band)
        | (offsets >= width - x_band)
        | (uv[:, 1] <= box.y_min + y_band)
        | (uv[:, 1] >= box.y_max - y_band)
    )
    return float(np.mean(boundary))


def _metadata_polygons(detection: Detection2D) -> tuple[np.ndarray, ...]:
    values = detection.metadata.get('mask_polygons_panorama_uv', ())
    try:
        return tuple(np.asarray(value, dtype=np.float64) for value in values)
    except (TypeError, ValueError):
        return ()


def _valid_mask(mask: np.ndarray, box: PanoramaBox) -> bool:
    array = np.asarray(mask)
    return (
        array.ndim == 2
        and array.shape == (box.panorama_height, box.panorama_width)
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )


def _valid_polygon(polygon: np.ndarray) -> bool:
    array = np.asarray(polygon)
    return (
        array.ndim == 2
        and array.shape[0] >= 3
        and array.shape[1] == 2
        and np.all(np.isfinite(array))
    )


def _unwrap_u(values: np.ndarray, centre: float, width: int) -> np.ndarray:
    return centre + np.mod(values - centre + width / 2.0, width) - width / 2.0


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=np.bool_)
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crossing = ((y1 > y) != (y2 > y))
        denominator = y2 - y1
        x_crossing = (x2 - x1) * (y - y1) / (
            denominator if abs(denominator) > 1e-12 else 1e-12
        ) + x1
        inside ^= crossing & (x <= x_crossing)
        previous = current
    return inside


__all__ = [
    'PointSelectionConfig',
    'PointSelectionResult',
    'points_inside_panorama_box',
    'points_inside_panorama_polygons',
    'select_detection_points',
    'SelectionMode',
]
