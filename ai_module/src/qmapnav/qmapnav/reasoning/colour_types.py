"""Validated colour colour contracts."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np


COLOUR_STATUSES = frozenset({
    'good',
    'low_saturation',
    'too_few_pixels',
    'mask_contaminated',
    'overexposed',
    'underexposed',
    'ambiguous',
    'no_crop',
})


def _optional_vector(value, name, length):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite shape ({length},)')
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ColourEstimate:
    """One calibrated object-colour observation or explicit failure."""

    probabilities: Mapping[str, float]
    dominant_colour: str | None
    confidence: float
    valid_pixel_count: int
    hsv_median: np.ndarray | None
    lab_median: np.ndarray | None
    source_viewpoint_id: str | None
    source_detection_id: str | None
    status: str

    def __post_init__(self) -> None:
        if self.status not in COLOUR_STATUSES:
            raise ValueError(f'unsupported colour status {self.status!r}')
        checked = {str(name): float(value)
                   for name, value in self.probabilities.items()}
        if any(not isfinite(value) or value < 0.0 for value in checked.values()):
            raise ValueError('colour probabilities must be finite and non-negative')
        total = sum(checked.values())
        if checked and not np.isclose(total, 1.0, atol=1.0e-6):
            raise ValueError('colour probabilities must sum to one')
        if self.dominant_colour is not None:
            if self.dominant_colour not in checked:
                raise ValueError('dominant colour must be in probabilities')
            expected = max(sorted(checked), key=lambda key: checked[key])
            if self.dominant_colour != expected:
                raise ValueError('dominant colour must maximize probability')
        elif checked:
            raise ValueError('non-empty probabilities require a dominant colour')
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('colour confidence must lie in [0, 1]')
        if self.valid_pixel_count < 0:
            raise ValueError('valid_pixel_count must be non-negative')
        object.__setattr__(self, 'probabilities', MappingProxyType(checked))
        object.__setattr__(
            self, 'hsv_median', _optional_vector(self.hsv_median, 'hsv_median', 3)
        )
        object.__setattr__(
            self, 'lab_median', _optional_vector(self.lab_median, 'lab_median', 3)
        )


@dataclass(frozen=True)
class ColourPrototype:
    """Robust HSV/Lab prototype fitted from released development metadata."""

    class_name: str
    lab_centre: np.ndarray
    lab_covariance: np.ndarray
    hue_centre: float | None
    hue_spread: float | None
    saturation_range: tuple[float, float]
    lightness_range: tuple[float, float]
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError('prototype class_name must be non-empty')
        centre = np.asarray(self.lab_centre, dtype=np.float64)
        covariance = np.asarray(self.lab_covariance, dtype=np.float64)
        if centre.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError('prototype Lab geometry has invalid shape')
        if not np.all(np.isfinite(centre)) or not np.all(np.isfinite(covariance)):
            raise ValueError('prototype Lab values must be finite')
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError('prototype covariance must be positive definite')
        for value in (self.hue_centre, self.hue_spread):
            if value is not None and not isfinite(value):
                raise ValueError('prototype hue values must be finite')
        for name, limits in (
            ('saturation_range', self.saturation_range),
            ('lightness_range', self.lightness_range),
        ):
            if len(limits) != 2 or not all(isfinite(item) for item in limits):
                raise ValueError(f'{name} must contain two finite values')
            if limits[1] < limits[0]:
                raise ValueError(f'{name} must be ordered')
        if self.sample_count <= 0:
            raise ValueError('prototype sample_count must be positive')
        centre = centre.copy()
        covariance = covariance.copy()
        centre.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, 'lab_centre', centre)
        object.__setattr__(self, 'lab_covariance', covariance)
        object.__setattr__(self, 'saturation_range', tuple(self.saturation_range))
        object.__setattr__(self, 'lightness_range', tuple(self.lightness_range))


__all__ = ['COLOUR_STATUSES', 'ColourEstimate', 'ColourPrototype']
