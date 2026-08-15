"""Repeated observations must fuse before numerical counting."""

from day11_helpers import make_candidate
from day11_helpers import make_observation

from qmapnav.counting import resolve_numerical_from_maps
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap


def test_same_physical_object_seen_twice_counts_once() -> None:
    object_map = ObjectMap()
    first = make_candidate(
        'cup_view_a', (0.0, 0.0, 0.8), class_name='cup',
        confidence=1.0, timestamp_ns=1,
    )
    second = make_candidate(
        'cup_view_b', (0.02, 0.01, 0.8), class_name='cup',
        confidence=1.0, timestamp_ns=2,
    )
    first_id = object_map.add_or_update(first, make_observation(first, 'a'))
    second_id = object_map.add_or_update(second, make_observation(second, 'b'))
    assert first_id == second_id

    result = resolve_numerical_from_maps(
        parse_question('How many cups are there?'),
        object_map,
        StructuralMap(),
    )
    assert result.count == 1
    assert result.qualifying_instance_ids == (first_id,)
