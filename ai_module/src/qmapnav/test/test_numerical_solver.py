"""Persistent-map class, colour, relation, and ambiguity counting tests."""

from fixtures import add_object

import pytest

from qmapnav.counting import AnchorCountHypothesis
from qmapnav.counting import assess_anchor_counts
from qmapnav.counting import CountDiagnostic
from qmapnav.counting import NumericalResult

from qmapnav.counting import resolve_numerical_from_maps
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap


def _maps():
    return ObjectMap(), StructuralMap()


def test_counts_unique_persistent_ids_after_support_filtering() -> None:
    object_map, structural_map = _maps()
    add_object(object_map, 'bed', 'bed', (0.0, 0.0, 0.4), (2.0, 2.0, 0.8))
    first = add_object(
        object_map, 'pillow_a', 'pillow', (-0.4, 0.0, 0.9),
        (0.5, 0.4, 0.2),
    )
    second = add_object(
        object_map, 'pillow_b', 'pillow', (0.4, 0.0, 0.9),
        (0.5, 0.4, 0.2),
    )
    add_object(
        object_map, 'pillow_floor', 'pillow', (4.0, 0.0, 0.1),
        (0.5, 0.4, 0.2),
    )

    result = resolve_numerical_from_maps(
        parse_question('How many pillows are on the bed?'),
        object_map,
        structural_map,
    )

    assert result.count == 2
    assert result.qualifying_instance_ids == (first, second)
    assert len(result.rejected_instance_ids) >= 1


def test_colour_constraint_is_applied_before_counting() -> None:
    object_map, structural_map = _maps()
    add_object(object_map, 'sofa', 'sofa', (0.0, 0.0, 0.5), (2.5, 1.0, 1.0))
    red = add_object(
        object_map, 'red', 'pillow', (-0.4, 0.0, 0.9),
        (0.4, 0.4, 0.2), colours={'red': 0.9, 'blue': 0.1},
    )
    add_object(
        object_map, 'blue', 'pillow', (0.4, 0.0, 0.9),
        (0.4, 0.4, 0.2), colours={'red': 0.02, 'blue': 0.98},
    )

    result = resolve_numerical_from_maps(
        parse_question('How many red pillows are on the sofa?'),
        object_map,
        structural_map,
    )

    assert result.qualifying_instance_ids == (red,)
    assert result.count == 1


def test_counted_anchor_role_uses_existential_subject_witness() -> None:
    object_map, structural_map = _maps()
    first_chair = add_object(
        object_map, 'chair_a', 'chair', (-2.0, 0.0, 0.5),
        (0.8, 0.8, 1.0),
    )
    second_chair = add_object(
        object_map, 'chair_b', 'chair', (2.0, 0.0, 0.5),
        (0.8, 0.8, 1.0),
    )
    add_object(
        object_map, 'pillow_a', 'pillow', (-2.0, 0.0, 1.05),
        (0.4, 0.4, 0.2),
    )
    add_object(
        object_map, 'pillow_b', 'pillow', (2.0, 0.0, 1.05),
        (0.4, 0.4, 0.2),
    )

    result = resolve_numerical_from_maps(
        parse_question('Count the number of chairs with pillows on them.'),
        object_map,
        structural_map,
    )

    assert result.qualifying_instance_ids == (first_chair, second_chair)
    assert result.count == 2


def test_anchor_hypotheses_are_compared_instead_of_selecting_first() -> None:
    object_map, structural_map = _maps()
    add_object(object_map, 'sofa_a', 'sofa', (-3.0, 0.0, 0.5), (2.0, 1.0, 1.0))
    add_object(object_map, 'sofa_b', 'sofa', (3.0, 0.0, 0.5), (2.0, 1.0, 1.0))
    add_object(
        object_map, 'pillow_a', 'pillow', (-3.0, 0.0, 1.0),
        (0.4, 0.4, 0.2),
    )
    add_object(
        object_map, 'pillow_b', 'pillow', (2.7, 0.0, 1.0),
        (0.4, 0.4, 0.2),
    )
    add_object(
        object_map, 'pillow_c', 'pillow', (3.3, 0.0, 1.0),
        (0.4, 0.4, 0.2),
    )

    result = resolve_numerical_from_maps(
        parse_question('How many pillows are on a sofa?'),
        object_map,
        structural_map,
    )

    assert result.count == 3
    assert result.anchor_ambiguity.count_consistent is False
    assert result.anchor_ambiguity.minimum_count == 1
    assert result.anchor_ambiguity.maximum_count == 2


def test_missing_anchor_preserves_target_as_unresolved() -> None:
    object_map, structural_map = _maps()
    target = add_object(
        object_map, 'cup', 'cup', (0.0, 0.0, 0.8),
        (0.1, 0.1, 0.2),
    )
    result = resolve_numerical_from_maps(
        parse_question('How many cups are on the coffee table?'),
        object_map,
        structural_map,
    )
    assert result.count == 0
    assert result.unresolved_instance_ids == (target,)


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
