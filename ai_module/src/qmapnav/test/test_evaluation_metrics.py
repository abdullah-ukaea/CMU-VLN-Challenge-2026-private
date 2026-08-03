"""Tests for ROS-independent oracle proxy metrics."""

import pytest

from qmapnav.evaluation.metrics import count_accuracy_metric
from qmapnav.evaluation.metrics import forbidden_region_metrics
from qmapnav.evaluation.metrics import object_selection_metric
from qmapnav.evaluation.metrics import relation_metrics
from qmapnav.evaluation.metrics import semantic_route_metric
from qmapnav.evaluation.metrics import terminal_goal_distance
from qmapnav.evaluation.metrics import TimingMetric
from qmapnav.reasoning import Polygon2D
from qmapnav.reasoning import SemanticRegion


def _square_region(
    region_id: str,
    centre_x: float,
    centre_y: float = 0.0,
    *,
    required: bool = True,
) -> SemanticRegion:
    half_size = 0.2
    return SemanticRegion(
        region_id=region_id,
        region_type='test',
        polygon=Polygon2D(
            (
                (centre_x - half_size, centre_y - half_size),
                (centre_x + half_size, centre_y - half_size),
                (centre_x + half_size, centre_y + half_size),
                (centre_x - half_size, centre_y + half_size),
            )
        ),
        source_object_ids=(region_id,),
        required=required,
    )


def test_object_selection_distinguishes_wrong_and_unavailable_labels() -> None:
    wrong = object_selection_metric('chair_2', 'chair_1')
    unavailable = object_selection_metric('chair_2', None)

    assert wrong.label_available
    assert wrong.correct is False
    assert not unavailable.label_available
    assert unavailable.correct is None


def test_count_accuracy_reports_exact_match_and_absolute_error() -> None:
    exact = count_accuracy_metric(3, 3)
    wrong = count_accuracy_metric(1, 3)
    unavailable = count_accuracy_metric(2, None)

    assert exact.exact_match is True
    assert exact.absolute_error == 0
    assert wrong.exact_match is False
    assert wrong.absolute_error == 2
    assert unavailable.exact_match is None
    assert unavailable.absolute_error is None


def test_relation_metric_reports_complete_confusion_matrix() -> None:
    expected = {
        ('near', 'chair_1', ('table_1',)),
        ('near', 'chair_2', ('table_1',)),
    }
    predicted = {
        ('near', 'chair_1', ('table_1',)),
        ('near', 'chair_3', ('table_1',)),
    }

    metric = relation_metrics(
        expected,
        predicted,
        relation='near',
        negatives_evaluated=2,
    )

    assert metric.true_positive == 1
    assert metric.false_positive == 1
    assert metric.false_negative == 1
    assert metric.true_negative == 1
    assert metric.precision == pytest.approx(0.5)
    assert metric.recall == pytest.approx(0.5)
    assert metric.f1 == pytest.approx(0.5)


def test_semantic_route_metric_scores_order_avoidance_and_terminal_goal() -> None:
    required = (
        _square_region('plant', 1.0),
        _square_region('gate', 2.0),
        _square_region('window', 3.0),
    )
    forbidden = (_square_region('sofa', 2.0, 2.0, required=False),)
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))

    metric = semantic_route_metric(trajectory, required, forbidden)

    assert metric.required_intersection_fraction == 1.0
    assert metric.ordered_constraints_completed == 3
    assert all(item is not None for item in metric.ordered_hit_indices)
    assert metric.order_correct
    assert metric.forbidden_violation_count == 0
    assert metric.terminal_goal_distance == 0.0
    assert metric.terminal_goal_reached
    assert metric.proxy_score == 6.0
    assert metric.success


def test_semantic_route_metric_detects_wrong_order_and_forbidden_entry() -> None:
    required = (
        _square_region('first', 2.0),
        _square_region('second', 1.0),
    )
    forbidden = (_square_region('forbidden', 1.5, required=False),)
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    metric = semantic_route_metric(trajectory, required, forbidden)

    assert metric.required_intersection_fraction == 1.0
    assert not metric.order_correct
    assert metric.ordered_hit_indices[1] is None
    assert metric.forbidden_violation_count == 1
    assert not metric.success


def test_forbidden_length_and_terminal_region_distance_are_geometric() -> None:
    forbidden = _square_region('forbidden', 1.0, required=False)
    result = forbidden_region_metrics(
        ((0.0, 0.0), (2.0, 0.0)),
        (forbidden,),
        sampling_resolution=0.01,
    )[0]

    assert result.violated
    assert result.approximate_length_inside == pytest.approx(0.4, abs=0.02)
    assert terminal_goal_distance((2.0, 0.0), forbidden) == pytest.approx(0.8)


def test_timing_rejects_negative_stage_values() -> None:
    with pytest.raises(ValueError, match='parser_seconds'):
        TimingMetric(-0.1, 0.0, 0.0, None, 0.0)
