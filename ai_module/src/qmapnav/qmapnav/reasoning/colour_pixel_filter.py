"""Mask, geometry, exposure, and contamination handling for object colours."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ColourSelectionConfig:
    """Measured starting thresholds for object-pixel selection."""

    min_valid_pixels: int = 50
    mask_erosion_px: int = 3
    small_object_mask_erosion_px: int = 1
    geometry_support_dilation_px: int = 3
    contracted_box_margin_fraction: float = 0.08
    low_saturation_threshold: float = 0.15
    shadow_lower_percentile: float = 5.0
    highlight_value_threshold: float = 0.92
    highlight_saturation_threshold: float = 0.12
    contamination_lab_distance: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            'min_valid_pixels',
            'mask_erosion_px',
            'small_object_mask_erosion_px',
            'geometry_support_dilation_px',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        for name in (
            'contracted_box_margin_fraction',
            'low_saturation_threshold',
            'highlight_value_threshold',
            'highlight_saturation_threshold',
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        if not 0.0 <= self.shadow_lower_percentile <= 50.0:
            raise ValueError('shadow percentile must lie in [0, 50]')
        if self.contamination_lab_distance <= 0.0:
            raise ValueError('contamination distance must be positive')


@dataclass(frozen=True)
class PixelSelectionResult:
    """Spatial object-pixel mask and auditable selection stages."""

    crop_rgb: np.ndarray
    selected_mask: np.ndarray
    spatial_weights: np.ndarray
    source: str
    status: str
    contamination_score: float
    stage_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        image = np.asarray(self.crop_rgb)
        mask = np.asarray(self.selected_mask, dtype=np.bool_)
        weights = np.asarray(self.spatial_weights, dtype=np.float64)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError('crop_rgb must have shape (H, W, 3)')
        if mask.shape != image.shape[:2] or weights.shape != mask.shape:
            raise ValueError('pixel-selection arrays must share image height/width')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError('spatial weights must be finite and non-negative')
        if not isfinite(self.contamination_score) or not (
            0.0 <= self.contamination_score <= 1.0
        ):
            raise ValueError('contamination_score must lie in [0, 1]')
        image = np.ascontiguousarray(image).copy()
        mask = np.ascontiguousarray(mask).copy()
        weights = np.ascontiguousarray(weights).copy()
        image.setflags(write=False)
        mask.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, 'crop_rgb', image)
        object.__setattr__(self, 'selected_mask', mask)
        object.__setattr__(self, 'spatial_weights', weights)
        object.__setattr__(
            self, 'stage_counts', MappingProxyType(dict(self.stage_counts))
        )


@dataclass(frozen=True)
class ReliablePixelResult:
    """HSV/Lab pixels and explicit rejected-pixel diagnostics."""

    rgb: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    weights: np.ndarray
    reliable_mask: np.ndarray
    shadow_mask: np.ndarray
    highlight_mask: np.ndarray
    status: str
    stage_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        count = np.asarray(self.rgb).shape[0]
        if np.asarray(self.rgb).shape != (count, 3):
            raise ValueError('rgb must have shape (N, 3)')
        for name in ('hsv', 'lab'):
            if np.asarray(getattr(self, name)).shape != (count, 3):
                raise ValueError(f'{name} must have shape (N, 3)')
        if np.asarray(self.weights).shape != (count,):
            raise ValueError('weights must have shape (N,)')
        for name in ('reliable_mask', 'shadow_mask', 'highlight_mask'):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.dtype != np.bool_:
                raise ValueError(f'{name} must be a 2D boolean mask')
        for name in ('rgb', 'hsv', 'lab', 'weights'):
            value = np.ascontiguousarray(getattr(self, name)).copy()
            if not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be finite')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ('reliable_mask', 'shadow_mask', 'highlight_mask'):
            value = np.ascontiguousarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, 'stage_counts', MappingProxyType(dict(self.stage_counts))
        )


def select_object_pixels(
    crop_rgb: np.ndarray | None,
    *,
    segmentation_mask: np.ndarray | None = None,
    geometry_support_uv: np.ndarray | None = None,
    config: ColourSelectionConfig | None = None,
) -> PixelSelectionResult | None:
    """Select pixels by mask, then projected support, then contracted box."""
    policy = config or ColourSelectionConfig()
    if crop_rgb is None:
        return None
    image = np.asarray(crop_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError('crop_rgb must have shape (H, W, 3)')
    height, width = image.shape[:2]
    counts = {'crop': height * width}
    source = 'contracted_box'
    mask = None
    if segmentation_mask is not None:
        candidate = np.asarray(segmentation_mask, dtype=np.bool_)
        if candidate.shape != (height, width):
            raise ValueError('segmentation_mask shape must match crop')
        candidate = _central_component(candidate)
        counts['mask_component'] = int(np.count_nonzero(candidate))
        if np.any(candidate):
            erosion = (
                policy.small_object_mask_erosion_px
                if np.count_nonzero(candidate) < 400
                else policy.mask_erosion_px
            )
            eroded = _erode_preserving(candidate, erosion)
            counts['mask_eroded'] = int(np.count_nonzero(eroded))
            if np.any(eroded):
                mask = eroded
                source = 'segmentation_mask'
    support_mask = _geometry_mask(
        geometry_support_uv,
        width,
        height,
        policy.geometry_support_dilation_px,
    )
    counts['geometry_supported'] = int(np.count_nonzero(support_mask))
    if mask is None and np.any(support_mask):
        mask = support_mask
        source = 'geometry_support'
    elif mask is not None and np.any(support_mask):
        intersection = mask & support_mask
        if np.count_nonzero(intersection) >= max(
            5, policy.min_valid_pixels // 4
        ):
            mask = intersection
            source = 'segmentation_mask+geometry_support'
    if mask is None:
        mask = _contracted_box_mask(
            width, height, policy.contracted_box_margin_fraction
        )
    weights = _spatial_weights(mask)
    if np.any(support_mask):
        weights = np.where(support_mask & mask, np.maximum(weights, 0.9), weights)
    selected_count = int(np.count_nonzero(mask))
    counts['selected'] = selected_count
    contamination = _contamination_score(image, mask)
    if selected_count < policy.min_valid_pixels:
        status = 'too_few_pixels'
    elif source == 'contracted_box' and contamination >= 0.80:
        status = 'mask_contaminated'
    else:
        status = 'good'
    return PixelSelectionResult(
        image,
        mask,
        weights,
        source,
        status,
        contamination,
        counts,
    )


def filter_reliable_pixels(
    selection: PixelSelectionResult,
    config: ColourSelectionConfig | None = None,
) -> ReliablePixelResult:
    """Downweight shadows/highlights while retaining true neutral objects."""
    if not isinstance(selection, PixelSelectionResult):
        raise TypeError('selection must be PixelSelectionResult')
    policy = config or ColourSelectionConfig()
    rgb_image = selection.crop_rgb.astype(np.float32) / 255.0
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    selected = selection.selected_mask
    selected_hsv = hsv_image[selected]
    selected_lab = lab_image[selected]
    selected_rgb = rgb_image[selected]
    selected_weights = selection.spatial_weights[selected].copy()
    shadow_full = np.zeros(selected.shape, dtype=np.bool_)
    highlight_full = np.zeros(selected.shape, dtype=np.bool_)
    if selected_rgb.shape[0] == 0:
        return ReliablePixelResult(
            selected_rgb,
            selected_hsv,
            selected_lab,
            selected_weights,
            np.zeros_like(selected),
            shadow_full,
            highlight_full,
            'too_few_pixels',
            {'selected': 0, 'reliable': 0},
        )
    saturation = selected_hsv[:, 1]
    value = selected_hsv[:, 2]
    median_saturation = float(np.median(saturation))
    median_value = float(np.median(value))
    low_tail = float(np.percentile(value, policy.shadow_lower_percentile))
    # Include a repeated dark tail only when it is separated from the object
    # median. Otherwise use a strict comparison so a large flat-colour mode is
    # not mistaken for shadow when the percentile lands on that mode.
    shadow = (
        value <= low_tail
        if low_tail < median_value - 0.03
        else value < low_tail
    )
    genuinely_dark = float(np.percentile(value, 90.0)) < 0.18
    if not genuinely_dark:
        selected_weights[shadow] *= 0.25
    else:
        shadow[:] = False
    highlight = (
        (value >= policy.highlight_value_threshold)
        & (saturation <= policy.highlight_saturation_threshold)
    )
    highlight_fraction = float(np.mean(highlight))
    if highlight_fraction < 0.60:
        selected_weights[highlight] *= 0.10
    else:
        highlight[:] = False
    coordinates = np.flatnonzero(selected)
    shadow_full.flat[coordinates[shadow]] = True
    highlight_full.flat[coordinates[highlight]] = True
    reliable = selected.copy()
    reliable.flat[coordinates[selected_weights <= 0.01]] = False
    counts = {
        **selection.stage_counts,
        'shadow_downweighted': int(np.count_nonzero(shadow)),
        'highlight_downweighted': int(np.count_nonzero(highlight)),
        'reliable': int(np.count_nonzero(selected_weights > 0.01)),
    }
    if selection.status != 'good':
        status = selection.status
    elif genuinely_dark and median_value < 0.12:
        status = 'underexposed'
    elif highlight_fraction >= 0.85:
        status = 'overexposed'
    elif median_saturation < policy.low_saturation_threshold:
        status = 'low_saturation'
    else:
        status = 'good'
    return ReliablePixelResult(
        selected_rgb,
        selected_hsv,
        selected_lab,
        selected_weights,
        reliable,
        shadow_full,
        highlight_full,
        status,
        counts,
    )


def _central_component(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    centre = np.array([(mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0])
    choices = []
    for label in range(1, count):
        coordinates = np.argwhere(labels == label)
        if coordinates.size == 0:
            continue
        distance = float(np.linalg.norm(np.mean(coordinates, axis=0) - centre))
        choices.append((distance, -coordinates.shape[0], label))
    selected_label = min(choices)[2]
    return labels == selected_label


def _erode_preserving(mask, pixels):
    if pixels <= 0:
        return mask
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel) != 0
    minimum = max(5, int(np.count_nonzero(mask) * 0.20))
    return eroded if np.count_nonzero(eroded) >= minimum else mask


def _geometry_mask(points, width, height, dilation):
    mask = np.zeros((height, width), dtype=np.uint8)
    if points is None:
        return mask.astype(np.bool_)
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('geometry_support_uv must have shape (N, 2)')
    finite = values[np.all(np.isfinite(values), axis=1)]
    for u, v in np.rint(finite).astype(np.int64):
        if 0 <= u < width and 0 <= v < height:
            mask[v, u] = 1
    if dilation > 0 and np.any(mask):
        size = 2 * dilation + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    return mask != 0


def _contracted_box_mask(width, height, fraction):
    margin_x = min(width // 4, int(round(width * fraction)))
    margin_y = min(height // 4, int(round(height * fraction)))
    if width < 20:
        margin_x = min(margin_x, 1)
    if height < 20:
        margin_y = min(margin_y, 1)
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[margin_y:max(margin_y + 1, height - margin_y),
         margin_x:max(margin_x + 1, width - margin_x)] = True
    return mask


def _spatial_weights(mask):
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    maximum = float(np.max(distance))
    if maximum <= 0.0:
        return np.zeros(mask.shape, dtype=np.float64)
    return np.where(mask, 0.25 + 0.75 * distance / maximum, 0.0)


def _contamination_score(image, mask):
    border = ~mask
    if not np.any(mask) or not np.any(border):
        return 0.0
    rgb = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    difference = float(np.linalg.norm(
        np.median(lab[mask], axis=0) - np.median(lab[border], axis=0)
    ))
    return float(np.exp(-difference / 8.0))


__all__ = [
    'ColourSelectionConfig',
    'filter_reliable_pixels',
    'PixelSelectionResult',
    'ReliablePixelResult',
    'select_object_pixels',
]
