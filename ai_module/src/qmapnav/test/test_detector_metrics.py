"""Tests for detector recall and false-positive measurements."""

import numpy as np

from qmapnav.evaluation import empty_metric_counts
from qmapnav.evaluation import score_panorama_detections
from qmapnav.evaluation import VisibleInstance
from qmapnav.perception import Detection2D
from qmapnav.perception import PanoramaBox


def _box(start: float, end: float) -> PanoramaBox:
    boundary = np.array(
        ((start, 10.0), (end, 10.0), (end, 30.0), (start, 30.0)),
    )
    return PanoramaBox(100, 50, ((start, end),), 10.0, 30.0, boundary)


def _truth(
    instance_id: str,
    class_name: str,
    start: float,
    end: float,
    *,
    target: bool = False,
    anchor: bool = False,
    rare: bool = False,
    size: str = 'medium',
) -> VisibleInstance:
    return VisibleInstance(
        instance_id,
        class_name,
        _box(start, end),
        target,
        anchor,
        rare,
        size,
    )


def _prediction(
    detection_id: str,
    class_name: str,
    start: float,
    end: float,
    confidence: float,
) -> Detection2D:
    return Detection2D(
        detection_id,
        class_name,
        class_name,
        confidence,
        _box(start, end),
        (0,),
        ((1.0, 1.0, 10.0, 10.0),),
        ((start + end) / 2.0, 20.0),
        np.array((1.0, 0.0, 0.0)),
    )


def test_metrics_match_once_and_count_unmatched_predictions() -> None:
    truth = (
        _truth('chair_1', 'chair', 10.0, 30.0, target=True, size='small'),
        _truth('window_1', 'window', 50.0, 70.0, anchor=True, rare=True),
    )
    predictions = (
        _prediction('chair_good', 'chair', 11.0, 29.0, 0.9),
        _prediction('chair_duplicate', 'chair', 12.0, 28.0, 0.8),
        _prediction('wrong_class', 'table', 50.0, 70.0, 0.7),
    )

    metrics = score_panorama_detections(truth, predictions)

    assert metrics.matched_instances == 1
    assert metrics.target_recall == 1.0
    assert metrics.anchor_recall == 0.0
    assert metrics.rare_recall == 0.0
    assert metrics.small_recall == 1.0
    assert metrics.false_positives == 2


def test_metric_counts_are_additive() -> None:
    truth = (_truth('chair_1', 'chair', 10.0, 30.0, target=True),)
    prediction = (_prediction('chair_good', 'chair', 10.0, 30.0, 0.9),)
    measured = score_panorama_detections(truth, prediction)

    combined = empty_metric_counts() + measured + measured

    assert combined.visible_instances == 2
    assert combined.target_matched == 2
    assert combined.false_positives == 0
