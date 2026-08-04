"""Evaluate fitted Day 8 colour prototypes on fixed held-out scenes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.evaluation.colour_metrics import ColourEvaluationCase
from qmapnav.evaluation.colour_metrics import evaluate_colour_cases
from qmapnav.evaluation.colour_metrics import expected_calibration_error
from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_prototypes import load_colour_prototypes


MISSING_VALUES = frozenset({'', '_', '-1', 'n/a', 'none', 'null'})


def evaluate_heldout(
    vla_root: Path,
    split_path: Path,
    prototype_path: Path,
) -> dict[str, object]:
    """Classify held-out released RGB metadata without touching fit scenes."""
    split = json.loads(split_path.read_text(encoding='utf-8'))
    prototypes = load_colour_prototypes(prototype_path)
    cases = []
    scene_counts = {}
    for scene in split['held_out_scenes']:
        path = vla_root / scene / f'{scene}_object_result.csv'
        count_before = len(cases)
        with path.open(newline='', encoding='utf-8') as stream:
            for row_index, row in enumerate(csv.DictReader(stream)):
                for colour_index in (1, 2, 3):
                    raw = str(row.get(
                        f'object_color_scheme{colour_index}', ''
                    ) or '').strip()
                    if raw.casefold() in MISSING_VALUES:
                        continue
                    rgb = _row_rgb(row, colour_index)
                    if rgb is None:
                        continue
                    crop = np.tile(rgb, (24, 24, 1)).astype(np.uint8)
                    selection = select_object_pixels(
                        crop,
                        segmentation_mask=np.ones(
                            crop.shape[:2], dtype=np.bool_
                        ),
                    )
                    pixels = filter_reliable_pixels(selection)
                    features = extract_colour_features(pixels)
                    estimate = classify_colour(
                        features,
                        pixels,
                        prototypes,
                        source_viewpoint_id=scene,
                        source_detection_id=f'{row_index}:{colour_index}',
                    )
                    cases.append(ColourEvaluationCase(
                        f'{scene}:{row_index}:{colour_index}',
                        normalize_colour_name(raw),
                        estimate,
                    ))
        scene_counts[scene] = len(cases) - count_before
    report = evaluate_colour_cases(cases)
    report['expected_calibration_error'] = expected_calibration_error(report)
    report['held_out_scenes'] = sorted(split['held_out_scenes'])
    report['fit_scenes'] = sorted(split['fit_scenes'])
    report['scene_case_counts'] = scene_counts
    report['evaluation_source'] = 'released per-object RGB metadata'
    return report


def _row_rgb(row, index):
    values = []
    for channel in ('r', 'g', 'b'):
        value = str(row.get(f'object_color_{channel}{index}', '') or '').strip()
        if value.casefold() in MISSING_VALUES:
            return None
        values.append(int(value))
    return np.asarray(values, dtype=np.uint8)


def main() -> None:
    """Print deterministic held-out evaluation JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vla_root', type=Path)
    parser.add_argument('split_path', type=Path)
    parser.add_argument('prototype_path', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(evaluate_heldout(
        arguments.vla_root,
        arguments.split_path,
        arguments.prototype_path,
    ), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
