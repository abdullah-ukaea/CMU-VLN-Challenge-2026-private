"""Robust HSV/Lab object-level features and circular hue statistics."""

from dataclasses import dataclass
from math import atan2
from math import pi

import numpy as np

from qmapnav.reasoning.colour_pixel_filter import ReliablePixelResult


@dataclass(frozen=True)
class ColourFeatures:
    """Robust photometric summary of one selected object observation."""

    hsv_median: np.ndarray
    lab_median: np.ndarray
    lab_covariance: np.ndarray
    hue_centre_rad: float | None
    hue_spread: float | None
    saturation_percentiles: tuple[float, float, float]
    value_percentiles: tuple[float, float, float]
    lightness_percentiles: tuple[float, float, float]
    valid_pixel_count: int
    effective_pixel_count: float

    def __post_init__(self) -> None:
        for name, shape in (
            ('hsv_median', (3,)),
            ('lab_median', (3,)),
            ('lab_covariance', (3, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be finite shape {shape}')
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.valid_pixel_count <= 0 or self.effective_pixel_count <= 0.0:
            raise ValueError('colour feature pixel counts must be positive')


def extract_colour_features(pixels: ReliablePixelResult) -> ColourFeatures:
    """Extract weighted medians, robust covariance, and circular hue."""
    if not isinstance(pixels, ReliablePixelResult):
        raise TypeError('pixels must be ReliablePixelResult')
    if pixels.rgb.shape[0] == 0 or float(np.sum(pixels.weights)) <= 0.0:
        raise ValueError('at least one positively weighted pixel is required')
    weights = np.maximum(pixels.weights.astype(np.float64), 0.0)
    hsv_median = np.array([
        _weighted_median(pixels.hsv[:, index], weights) for index in range(3)
    ])
    lab_median = np.array([
        _weighted_median(pixels.lab[:, index], weights) for index in range(3)
    ])
    deviations = np.abs(pixels.lab - lab_median)
    limits = np.percentile(deviations, 90.0, axis=0)
    retained = np.all(deviations <= np.maximum(limits, 1.0e-6), axis=1)
    if np.count_nonzero(retained) < 4:
        retained[:] = True
    covariance = _weighted_covariance(
        pixels.lab[retained], weights[retained], regularization=4.0
    )
    hue_reliability = weights * np.clip(pixels.hsv[:, 1], 0.0, 1.0)
    if float(np.sum(hue_reliability)) <= 1.0e-9:
        hue_centre = None
        hue_spread = None
    else:
        angles = pixels.hsv[:, 0] * (2.0 * pi / 360.0)
        cosine = float(np.sum(hue_reliability * np.cos(angles)))
        sine = float(np.sum(hue_reliability * np.sin(angles)))
        hue_centre = atan2(sine, cosine) % (2.0 * pi)
        resultant = np.hypot(cosine, sine) / float(np.sum(hue_reliability))
        hue_spread = float(np.clip(1.0 - resultant, 0.0, 1.0))
    return ColourFeatures(
        hsv_median,
        lab_median,
        covariance,
        hue_centre,
        hue_spread,
        _percentiles(pixels.hsv[:, 1]),
        _percentiles(pixels.hsv[:, 2]),
        _percentiles(pixels.lab[:, 0]),
        pixels.rgb.shape[0],
        float(np.sum(weights)),
    )


def circular_hue_distance(first_rad: float, second_rad: float) -> float:
    """Return shortest distance between two hue angles in radians."""
    difference = abs(float(first_rad) - float(second_rad)) % (2.0 * pi)
    return min(difference, 2.0 * pi - difference)


def _weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    midpoint = float(np.sum(sorted_weights)) / 2.0
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint, side='left'))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _weighted_covariance(values, weights, regularization):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    mean = np.sum(values * weights[:, None], axis=0) / max(total, 1.0e-9)
    centred = values - mean
    covariance = (
        centred.T @ (centred * weights[:, None]) / max(total, 1.0)
    )
    return covariance + np.eye(3) * regularization


def _percentiles(values):
    result = np.percentile(values, (10.0, 50.0, 90.0))
    return tuple(float(item) for item in result)


__all__ = ['circular_hue_distance', 'ColourFeatures', 'extract_colour_features']
