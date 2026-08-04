"""Fit and serialize Day 8 colour prototypes from released RGB metadata."""

from collections import defaultdict
import csv
import json
from math import atan2
from math import pi
from pathlib import Path

import cv2
import numpy as np

from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.reasoning.colour_types import ColourPrototype


MISSING_VALUES = frozenset({'', '_', '-1', 'n/a', 'none', 'null'})
NEUTRAL_CLASSES = frozenset({'black', 'grey', 'white'})
WHITE_LIGHTNESS_TARGET = 82.0
WHITE_LIGHTNESS_QUANTILE = 95.0


def fit_colour_prototypes(
    vla_root: Path,
    fit_scenes: list[str] | tuple[str, ...],
) -> dict[str, ColourPrototype]:
    """Fit prototypes only from the explicitly supplied scene names."""
    samples = defaultdict(list)
    selected_scenes = set(fit_scenes)
    for scene in sorted(selected_scenes):
        path = vla_root / scene / f'{scene}_object_result.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                for index in (1, 2, 3):
                    raw = str(row.get(
                        f'object_color_scheme{index}', ''
                    )).strip()
                    if raw.casefold() in MISSING_VALUES:
                        continue
                    rgb_values = []
                    for channel in ('r', 'g', 'b'):
                        value = str(row.get(
                            f'object_color_{channel}{index}', ''
                        )).strip()
                        if value.casefold() in MISSING_VALUES:
                            break
                        rgb_values.append(int(value))
                    if len(rgb_values) != 3:
                        continue
                    canonical = normalize_colour_name(raw)
                    samples[canonical].append(rgb_values)
    if 'grey' in samples:
        grey = np.asarray(samples['grey'], dtype=np.uint8)
        lab = _rgb_to_lab(grey)
        # The released vocabulary has no explicit white label.  Preserve a
        # deterministic neutral policy by assigning only the brightest grey
        # tail to white; the absolute target prevents relabelling ordinary
        # light grey when the development palette already reaches white.
        lightness_cutoff = min(
            WHITE_LIGHTNESS_TARGET,
            float(np.percentile(lab[:, 0], WHITE_LIGHTNESS_QUANTILE)),
        )
        white = grey[lab[:, 0] >= lightness_cutoff]
        retained_grey = grey[lab[:, 0] < lightness_cutoff]
        if white.shape[0] >= 5 and retained_grey.shape[0] >= 5:
            samples['white'] = white.tolist()
            samples['grey'] = retained_grey.tolist()
    missing = sorted(set(COLOUR_CLASSES) - set(samples))
    if missing:
        raise ValueError(f'fit data cannot represent colours: {missing}')
    return fit_colour_prototypes_from_rgb(samples)


def fit_colour_prototypes_from_rgb(
    samples_by_class: dict[str, list[list[int]] | np.ndarray],
) -> dict[str, ColourPrototype]:
    """Fit robust regularized prototypes from labelled RGB samples."""
    output = {}
    for class_name in sorted(samples_by_class):
        rgb = np.asarray(samples_by_class[class_name], dtype=np.uint8)
        if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] == 0:
            raise ValueError(f'{class_name} RGB samples must have shape (N, 3)')
        lab = _rgb_to_lab(rgb)
        hsv = _rgb_to_hsv(rgb)
        centre = np.median(lab, axis=0)
        deviations = np.abs(lab - centre)
        limits = np.percentile(deviations, 90.0, axis=0)
        retained = np.all(deviations <= np.maximum(limits, 1.0e-6), axis=1)
        robust = lab[retained] if np.count_nonzero(retained) >= 3 else lab
        if robust.shape[0] <= 1:
            covariance = np.eye(3) * 36.0
        else:
            covariance = np.cov(robust, rowvar=False)
            covariance = np.atleast_2d(covariance)
            if covariance.shape != (3, 3):
                covariance = np.eye(3) * 36.0
            covariance += np.eye(3) * 16.0
        if class_name in NEUTRAL_CLASSES:
            hue_centre = None
            hue_spread = None
        else:
            angles = hsv[:, 0] * (2.0 * pi / 360.0)
            saturation = np.maximum(hsv[:, 1], 0.05)
            cosine = float(np.sum(saturation * np.cos(angles)))
            sine = float(np.sum(saturation * np.sin(angles)))
            hue_centre = atan2(sine, cosine) % (2.0 * pi)
            resultant = np.hypot(cosine, sine) / float(np.sum(saturation))
            hue_spread = float(np.clip(1.0 - resultant, 0.0, 1.0))
        output[class_name] = ColourPrototype(
            class_name,
            centre,
            covariance,
            hue_centre,
            hue_spread,
            tuple(float(item) for item in np.percentile(hsv[:, 1], (5, 95))),
            tuple(float(item) for item in np.percentile(lab[:, 0], (5, 95))),
            rgb.shape[0],
        )
    return output


def prototypes_to_json(prototypes: dict[str, ColourPrototype]) -> dict[str, object]:
    """Return finite JSON-safe fitted prototype metadata."""
    return {
        'schema_version': 1,
        'prototypes': {
            name: {
                'class_name': item.class_name,
                'lab_centre': item.lab_centre.tolist(),
                'lab_covariance': item.lab_covariance.tolist(),
                'hue_centre': item.hue_centre,
                'hue_spread': item.hue_spread,
                'saturation_range': list(item.saturation_range),
                'lightness_range': list(item.lightness_range),
                'sample_count': item.sample_count,
            }
            for name, item in sorted(prototypes.items())
        },
    }


def load_colour_prototypes(path: Path) -> dict[str, ColourPrototype]:
    """Load validated prototypes from a persisted JSON artifact."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        name: ColourPrototype(
            item['class_name'],
            np.asarray(item['lab_centre']),
            np.asarray(item['lab_covariance']),
            item['hue_centre'],
            item['hue_spread'],
            tuple(item['saturation_range']),
            tuple(item['lightness_range']),
            int(item['sample_count']),
        )
        for name, item in payload['prototypes'].items()
    }


def _rgb_to_lab(rgb):
    image = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def _rgb_to_hsv(rgb):
    image = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)


__all__ = [
    'fit_colour_prototypes',
    'fit_colour_prototypes_from_rgb',
    'load_colour_prototypes',
    'prototypes_to_json',
]
