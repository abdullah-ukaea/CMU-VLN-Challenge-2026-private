"""Day 11 support search reused for stable low and zero counts."""

from day12_helpers import add_object
from day12_helpers import numerical_result

from qmapnav.counting.support_counting import assess_counting_supports
from qmapnav.counting.support_counting import strengthen_zero_with_support_evidence
from qmapnav.exploration import SupportSearchHistory
from qmapnav.mapping import ObjectMap


def test_strong_negatives_on_all_supports_strengthen_zero() -> None:
    object_map = ObjectMap()
    support_id = add_object(
        object_map, 'table', 'table', (1.0, 0.0, 0.4),
        (1.5, 1.0, 0.8),
    )
    history = SupportSearchHistory()
    history.note_observation(
        str(support_id), target_class='cup', viewpoint_id='support_front',
        found=False, visible_surface_fraction=0.95, distance_m=1.2,
    )
    assessment = assess_counting_supports('cup', object_map, history)
    strengthened = strengthen_zero_with_support_evidence(
        numerical_result((), confidence=0.4), assessment
    )
    assert assessment.all_plausible_supports_exhausted
    assert strengthened.count == 0
    assert strengthened.count_confidence == 0.9


def test_weak_negative_does_not_claim_stable_zero() -> None:
    object_map = ObjectMap()
    support_id = add_object(
        object_map, 'table', 'table', (1.0, 0.0, 0.4),
        (1.5, 1.0, 0.8),
    )
    history = SupportSearchHistory()
    history.note_observation(
        str(support_id), target_class='cup', viewpoint_id='far',
        found=False, visible_surface_fraction=0.2, distance_m=5.0,
        occluded=True,
    )
    assessment = assess_counting_supports('cup', object_map, history)
    assert assessment.all_plausible_supports_exhausted is False
    assert assessment.negative_evidence_confidence == 0.0
