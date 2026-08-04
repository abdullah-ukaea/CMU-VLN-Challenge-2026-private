"""Tests for Day 8 colour metrics and saved visual diagnostics."""

import json

import numpy as np

from qmapnav.evaluation.colour_metrics import ColourEvaluationCase
from qmapnav.evaluation.colour_metrics import evaluate_colour_cases
from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_prototypes import fit_colour_prototypes_from_rgb
from qmapnav.reasoning.colour_types import ColourEstimate
from qmapnav.reasoning.colour_visualisation import save_colour_diagnostic
from qmapnav.reasoning.relation_visualisation import save_relation_diagnostic
from qmapnav.reasoning.support_relations import on_evidence
from test_day8_relations import _geometry


def _estimate(name, probability, status='good'):
    probabilities = {name: probability, 'grey': 1.0 - probability}
    return ColourEstimate(
        probabilities, name, probability, 100, np.ones(3), np.ones(3),
        'view', 'detection', status,
    )


def test_colour_metrics_report_top_k_confusion_calibration_and_coverage() -> None:
    failed = ColourEstimate(
        {}, None, 0.0, 0, None, None, 'view', 'failed', 'too_few_pixels'
    )
    report = evaluate_colour_cases([
        ColourEvaluationCase('a', 'blue', _estimate('blue', 0.8)),
        ColourEvaluationCase('b', 'red', _estimate('red', 0.6)),
        ColourEvaluationCase('c', 'white', failed),
    ])

    assert report['top1_accuracy'] == 2 / 3
    assert report['coverage'] == 2 / 3
    assert report['confusion_matrix']['white']['<failed>'] == 1
    assert sum(item['count'] for item in report['calibration']) == 2


def test_colour_and_relation_diagnostics_save_required_evidence(tmp_path) -> None:
    crop = np.full((30, 30, 3), [30, 80, 210], dtype=np.uint8)
    selection = select_object_pixels(
        crop, segmentation_mask=np.ones(crop.shape[:2], dtype=np.bool_)
    )
    pixels = filter_reliable_pixels(selection)
    features = extract_colour_features(pixels)
    samples = {
        name: np.tile(rgb, (3, 1))
        for name, rgb in {
            'black': [0, 0, 0], 'blue': [30, 80, 210],
            'brown': [100, 60, 20], 'green': [20, 150, 60],
            'grey': [110, 110, 110], 'orange': [240, 120, 20],
            'pink': [240, 140, 170], 'purple': [110, 40, 160],
            'red': [210, 30, 30], 'white': [245, 245, 245],
            'yellow': [230, 210, 30],
        }.items()
    }
    estimate = classify_colour(
        features, pixels, fit_colour_prototypes_from_rgb(samples)
    )
    png_path, colour_json = save_colour_diagnostic(
        tmp_path, 'blue_chair', selection, pixels, features, estimate,
        expected_colour='blue',
    )
    table = _geometry('table', 'table', (0.0, 0.0, 0.4), (1.5, 1.0, 0.8))
    book = _geometry('book', 'book', (0.0, 0.0, 0.87), (0.2, 0.2, 0.1))
    side, top, relation_json = save_relation_diagnostic(
        tmp_path, 'book_table', book, table, on_evidence(book, table)
    )

    assert all(path.stat().st_size > 0
               for path in (png_path, colour_json, side, top, relation_json))
    colour_payload = json.loads(colour_json.read_text(encoding='utf-8'))
    relation_payload = json.loads(relation_json.read_text(encoding='utf-8'))
    assert len(colour_payload['hue_histogram_10deg']) == 36
    assert colour_payload['probabilities']
    assert relation_payload['subject_support_overlap'] > 0.9
