"""Tests for reasoning contracts, map candidates, and cardinality."""

from math import nan

import numpy as np
import pytest

from qmapnav.common import EntityReference, ObjectInstance
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.reasoning.candidate_generation import generate_candidates
from qmapnav.reasoning.resolution_contracts import CandidateHypothesis
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation
from qmapnav.reasoning.resolution_contracts import PairHypothesis
from qmapnav.reasoning.resolution_contracts import ReferenceResolution
from qmapnav.reasoning.resolution_contracts import SetResolution


def _object(instance_id, class_scores, colour_scores=None, confidence=0.8):
    return ObjectInstance(
        instance_id,
        class_scores,
        colour_scores or {},
        np.array((float(instance_id), 0.0, 0.5)),
        np.array((float(instance_id) - 0.25, -0.25, 0.0)),
        np.array((float(instance_id) + 0.25, 0.25, 1.0)),
        np.array((0.5, 0.5, 1.0)),
        0.0,
        0.8,
        2,
        confidence,
    )


def test_resolution_contracts_serialize_without_forcing_selection():
    evaluation = ConstraintEvaluation(
        'colour', 0.4, False, None, 0.3, {'probability': 0.4}
    )
    hypothesis = CandidateHypothesis(
        ('chair_7',), 0.4, 0.3, (), (), ('colour',), {'colour': 0.4}
    )
    resolution = ReferenceResolution(
        'target', (hypothesis,), None, 0.0, 'low_confidence', ('colour',)
    )
    assert evaluation.to_dict()['satisfied'] is None
    assert resolution.to_dict()['selected_candidate_ids'] is None
    assert resolution.to_dict()['ranked_hypotheses'][0][
        'unresolved_constraints'
    ] == ['colour']


def test_contracts_reject_nonfinite_and_duplicate_ids():
    with pytest.raises(ValueError):
        ConstraintEvaluation('near', nan, False, True, 1.0)
    with pytest.raises(ValueError):
        CandidateHypothesis(('same', 'same'), 0.5, 0.5)


def test_pair_is_canonical_and_set_partitions_are_disjoint():
    pair = PairHypothesis(
        'table_5', 'table_2', 'table', 2.0, 1.0, True, 1.0, 0.8,
        0.9,
    )
    assert pair.candidate_ids == ('table_2', 'table_5')
    with pytest.raises(ValueError):
        SetResolution(('chair_1',), (), (), ('chair_1',))


def test_candidate_generation_uses_class_distribution_and_soft_colour():
    reference = EntityReference(
        'chair_ref', 'chair', {'colour': 'orange'}, 1, 'orange chair'
    )
    objects = (
        _object(1, {'stool': 0.7, 'chair': 0.25}, {'orange': 0.05}),
        _object(2, {'table': 0.95, 'chair': 0.05}, {'orange': 0.95}),
    )
    result = generate_candidates(reference, objects)
    assert tuple(item.candidate_id for item in result.retained) == ('1',)
    assert 'colour_probability_below_minimum' in result.retained[0].reasons
    assert result.cardinality_satisfied


def test_candidate_generation_keeps_weak_geometry_as_fallback():
    reference = EntityReference('chair_ref', 'chair')
    result = generate_candidates(
        reference, (_object(1, {'chair': 0.9}, confidence=0.1),)
    )
    assert result.retained
    assert 'weak_geometry_retained_for_later_constraints' in (
        result.retained[0].reasons
    )


def test_candidate_generation_searches_structural_anchors():
    reference = EntityReference('window_ref', 'window')
    anchor = StructuralAnchor(
        'window_1',
        'architectural',
        'window',
        np.array((1.0, 2.0, 1.5)),
        None,
        None,
        None,
        np.array((1.0, 0.1, 1.0)),
        0.0,
        'wall_1',
        0.8,
        0,
        1,
        ('view',),
        ('detection',),
    )
    result = generate_candidates(reference, (), (anchor,))
    assert result.retained[0].candidate_id == 'window_1'
    assert result.retained[0].source_type == 'structural'
