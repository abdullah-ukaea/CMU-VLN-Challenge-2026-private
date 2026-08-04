"""Saved Day 8 object-pixel and probability diagnostics."""

import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.reasoning.colour_features import ColourFeatures
from qmapnav.reasoning.colour_pixel_filter import PixelSelectionResult
from qmapnav.reasoning.colour_pixel_filter import ReliablePixelResult
from qmapnav.reasoning.colour_types import ColourEstimate


def save_colour_diagnostic(
    output_directory: Path,
    case_id: str,
    selection: PixelSelectionResult,
    pixels: ReliablePixelResult,
    features: ColourFeatures,
    estimate: ColourEstimate,
    *,
    expected_colour: str | None = None,
) -> tuple[Path, Path]:
    """Save a composite pixel audit PNG and detailed JSON metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)
    original = selection.crop_rgb.copy()
    selected = _overlay(original, selection.selected_mask, (0, 220, 0))
    rejected = original.copy()
    rejected = _overlay(rejected, pixels.shadow_mask, (20, 20, 255))
    rejected = _overlay(rejected, pixels.highlight_mask, (255, 220, 20))
    height, width = original.shape[:2]
    panels = [
        cv2.resize(panel[..., ::-1], (width, height))
        for panel in (original, selected, rejected)
    ]
    composite = np.concatenate(panels, axis=1)
    png_path = output_directory / f'{case_id}_colour.png'
    if not cv2.imwrite(str(png_path), composite):
        raise OSError(f'failed to write {png_path}')
    hue_histogram, _ = np.histogram(
        pixels.hsv[:, 0], bins=36, range=(0.0, 360.0),
        weights=pixels.weights,
    )
    payload = {
        'case_id': case_id,
        'expected_colour': expected_colour,
        'selection_source': selection.source,
        'selection_status': selection.status,
        'colour_status': estimate.status,
        'valid_pixel_count': estimate.valid_pixel_count,
        'stage_counts': dict(pixels.stage_counts),
        'contamination_score': selection.contamination_score,
        'hue_histogram_10deg': hue_histogram.tolist(),
        'hsv_median': features.hsv_median.tolist(),
        'lab_median': features.lab_median.tolist(),
        'lab_covariance': features.lab_covariance.tolist(),
        'probabilities': dict(estimate.probabilities),
        'confidence': estimate.confidence,
    }
    json_path = output_directory / f'{case_id}_colour.json'
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    return png_path, json_path


def _overlay(image, mask, colour):
    result = image.astype(np.float32).copy()
    mask = np.asarray(mask, dtype=np.bool_)
    result[mask] = 0.55 * result[mask] + 0.45 * np.asarray(colour)
    return np.clip(result, 0, 255).astype(np.uint8)


__all__ = ['save_colour_diagnostic']
