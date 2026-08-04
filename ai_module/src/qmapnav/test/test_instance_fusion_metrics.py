"""Quantitative Day 7 identity, geometry, and anchor metrics."""

import numpy as np
import pytest

from qmapnav.common import ObjectInstance
from qmapnav.evaluation.instance_fusion import evaluate_anchor_stability
from qmapnav.evaluation.instance_fusion import evaluate_fusion_geometry
from qmapnav.evaluation.instance_fusion import evaluate_identity_assignments
from qmapnav.evaluation.instance_fusion import IdentityAssignment
from qmapnav.mapping.structural_map import StructuralAnchor


def _instance() -> ObjectInstance:
    return ObjectInstance(
        0,
        {'chair': 1.0},
        {},
        np.array([1.01, 2.0, 0.5]),
        np.array([0.7, 1.75, 0.0]),
        np.array([1.3, 2.25, 1.0]),
        np.array([0.6, 0.5, 1.0]),
        0.0,
        0.8,
        3,
        0.85,
    )


def _anchor(anchor_id: str, position: tuple[float, float, float]):
    return StructuralAnchor(
        anchor_id,
        'window',
        'window',
        np.asarray(position),
        None,
        None,
        np.array([0.0, 1.0, 0.0, -2.0]),
        np.array([1.0, 0.05, 1.0]),
        0.0,
        'wall_0000',
        0.8,
        1,
        2,
        ('a', 'b'),
        (anchor_id,),
    )


def test_identity_metrics_detect_duplicates_and_false_merges() -> None:
    metrics = evaluate_identity_assignments([
        IdentityAssignment('chair_left', 0),
        IdentityAssignment('chair_left', 0),
        IdentityAssignment('chair_left', 1),
        IdentityAssignment('chair_right', 0),
    ])

    assert metrics.physical_object_count == 2
    assert metrics.extra_instance_count == 1
    assert metrics.duplicate_rate == 0.5
    assert metrics.false_merge_count == 1
    assert metrics.maximum_ids_per_physical_object == 2


def test_fusion_metric_reports_improved_centre_without_assuming_it() -> None:
    metrics = evaluate_fusion_geometry(
        np.array([1.2, 2.0, 0.5]),
        np.array([0.8, 0.5, 1.0]),
        _instance(),
        np.array([1.0, 2.0, 0.5]),
        np.array([0.6, 0.5, 1.0]),
        first_yaw_rad=0.2,
        reference_yaw_rad=0.0,
    )

    assert metrics.centre_error_improvement_m > 0.18
    assert metrics.dimension_error_improvement_m == pytest.approx(0.2)
    assert metrics.oriented_iou_improvement > 0.0
    assert metrics.yaw_error_improvement_rad == pytest.approx(0.2)


def test_anchor_stability_measures_variance_and_wall_correctness() -> None:
    metrics = evaluate_anchor_stability([
        _anchor('a', (0.0, 2.0, 1.0)),
        _anchor('b', (0.02, 2.0, 1.01)),
        _anchor('c', (-0.01, 2.0, 0.99)),
    ])

    assert metrics.observation_count == 3
    assert metrics.maximum_position_error_m < 0.03
    assert metrics.supporting_wall_consistent
