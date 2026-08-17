"""Uncertainty-aware HSV/Lab colour probability classification."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.reasoning.colour_features import circular_hue_distance
from qmapnav.reasoning.colour_features import ColourFeatures
from qmapnav.reasoning.colour_pixel_filter import ReliablePixelResult
from qmapnav.reasoning.colour_types import ColourEstimate
from qmapnav.reasoning.colour_types import ColourPrototype


NEUTRAL_CLASSES = frozenset({'black', 'grey', 'white'})


@dataclass(frozen=True)
class ColourClassifierConfig:
    """Probability temperature and confidence gates."""

    probability_temperature: float = 1.0
    ambiguous_margin: float = 0.12
    min_valid_pixels: int = 50
    low_saturation_threshold: float = 0.15

    def __post_init__(self) -> None:
        if not isfinite(self.probability_temperature) or (
            self.probability_temperature <= 0.0
        ):
            raise ValueError('probability temperature must be positive')
        if not 0.0 <= self.ambiguous_margin <= 1.0:
            raise ValueError('ambiguous_margin must lie in [0, 1]')
        if self.min_valid_pixels <= 0:
            raise ValueError('min_valid_pixels must be positive')


def classify_colour(
    features: ColourFeatures,
    pixels: ReliablePixelResult,
    prototypes: dict[str, ColourPrototype],
    *,
    source_viewpoint_id: str | None = None,
    source_detection_id: str | None = None,
    config: ColourClassifierConfig | None = None,
) -> ColourEstimate:
    """Classify one feature vector into a full normalized distribution."""
    policy = config or ColourClassifierConfig()
    if set(prototypes) != set(COLOUR_CLASSES):
        raise ValueError('one prototype is required for every colour class')
    if features.valid_pixel_count < policy.min_valid_pixels:
        return _failure(
            'too_few_pixels', features, source_viewpoint_id, source_detection_id
        )
    if pixels.status == 'mask_contaminated':
        return _failure(
            'mask_contaminated', features,
            source_viewpoint_id, source_detection_id,
        )
    saturation = float(features.hsv_median[1])
    neutral = saturation < policy.low_saturation_threshold
    distances = {}
    for name, prototype in prototypes.items():
        delta = features.lab_median - prototype.lab_centre
        try:
            lab_distance = float(np.sqrt(max(
                0.0, delta @ np.linalg.solve(prototype.lab_covariance, delta)
            )))
        except np.linalg.LinAlgError:
            lab_distance = float(np.linalg.norm(delta) / 20.0)
        penalty = 0.0
        if neutral and name not in NEUTRAL_CLASSES:
            penalty += 4.0 * (1.0 - saturation)
        elif not neutral and name in NEUTRAL_CLASSES:
            penalty += 1.5 * saturation
        hue_distance = 0.0
        if (
            not neutral
            and name not in NEUTRAL_CLASSES
            and features.hue_centre_rad is not None
            and prototype.hue_centre is not None
        ):
            hue_distance = circular_hue_distance(
                features.hue_centre_rad, prototype.hue_centre
            ) / np.pi
        distances[name] = lab_distance + 2.0 * hue_distance + penalty
    values = np.asarray([distances[name] for name in COLOUR_CLASSES])
    shifted = values - float(np.min(values))
    logits = -shifted / policy.probability_temperature
    logits -= float(np.max(logits))
    probabilities_array = np.exp(logits)
    probabilities_array /= float(np.sum(probabilities_array))
    probabilities = {
        name: float(value)
        for name, value in zip(COLOUR_CLASSES, probabilities_array)
    }
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    dominant = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1]
    pixel_quality = min(1.0, features.effective_pixel_count / 250.0)
    consistency = 1.0 / (1.0 + float(np.trace(features.lab_covariance)) / 300.0)
    exposure_quality = 0.55 if pixels.status in {
        'underexposed', 'overexposed'
    } else 1.0
    confidence = float(np.clip(
        ordered[0][1] * (0.4 + 0.6 * margin)
        * (0.4 + 0.6 * pixel_quality)
        * (0.5 + 0.5 * consistency)
        * exposure_quality,
        0.0,
        1.0,
    ))
    if margin < policy.ambiguous_margin:
        status = 'ambiguous'
    elif pixels.status in {'underexposed', 'overexposed'}:
        status = pixels.status
    elif neutral:
        status = 'low_saturation'
    else:
        status = 'good'
    return ColourEstimate(
        probabilities,
        dominant,
        confidence,
        features.valid_pixel_count,
        features.hsv_median,
        features.lab_median,
        source_viewpoint_id,
        source_detection_id,
        status,
    )


def _failure(status, features, viewpoint_id, detection_id):
    return ColourEstimate(
        {},
        None,
        0.0,
        features.valid_pixel_count,
        features.hsv_median,
        features.lab_median,
        viewpoint_id,
        detection_id,
        status,
    )


__all__ = ['classify_colour', 'ColourClassifierConfig', 'NEUTRAL_CLASSES']
