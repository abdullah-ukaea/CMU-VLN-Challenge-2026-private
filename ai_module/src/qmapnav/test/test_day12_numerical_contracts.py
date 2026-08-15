"""Validation tests for Day 12 numerical and anchor contracts."""

import pytest

from qmapnav.counting import AnchorCountHypothesis
from qmapnav.counting import assess_anchor_counts
from qmapnav.counting import CountDiagnostic
from qmapnav.counting import NumericalResult


def test_anchor_ambiguity_compares_counts_and_underlying_ids() -> None:
    assessment = assess_anchor_counts((
        AnchorCountHypothesis((('sofa', '2'),), (1,), 0.9, 0.8),
        AnchorCountHypothesis((('sofa', '4'),), (7,), 0.8, 0.8),
    ))
    assert assessment.count_consistent is True
    assert assessment.id_set_consistent is False
    assert (assessment.minimum_count, assessment.maximum_count) == (1, 1)


def test_numerical_result_accepts_zero_as_a_real_answer() -> None:
    result = NumericalResult(
        'cup', (), (), (), (), 0, 0.8, True, 'supports_exhausted', (),
        assess_anchor_counts(()),
    )
    assert result.count == 0
    assert result.qualifying_instance_ids == ()


def test_numerical_result_requires_disjoint_partitions() -> None:
    with pytest.raises(ValueError, match='disjoint'):
        NumericalResult(
            'chair', (1,), (1,), (), (), 1, 0.8, False, 'collecting', (),
            assess_anchor_counts(()),
        )


def test_count_diagnostic_is_json_safe_and_read_only() -> None:
    diagnostic = CountDiagnostic(
        4, 'definite', 0.9, 0.8, 0.7, 0.75, 0.8,
        ('qualifying_definite',), {'target': '4'}, {'class': 0.9},
    )
    assert diagnostic.to_dict()['instance_id'] == 4
    with pytest.raises(TypeError):
        diagnostic.evidence['class'] = 0.1
