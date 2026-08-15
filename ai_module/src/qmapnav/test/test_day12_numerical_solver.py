"""Persistent-map class, colour, relation, and ambiguity counting tests."""

from day12_helpers import add_object

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
