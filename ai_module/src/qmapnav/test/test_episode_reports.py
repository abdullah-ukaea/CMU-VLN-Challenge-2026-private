"""Tests for episode manifests, records, failures, and fix ranking."""

from dataclasses import replace
from pathlib import Path

import pytest

from qmapnav.evaluation import build_object_reference_manifest
from qmapnav.evaluation import classify_primary_failure
from qmapnav.evaluation import FixCandidate
from qmapnav.evaluation import load_questions
from qmapnav.evaluation import manifest_digest
from qmapnav.evaluation import ObjectReferenceEpisodeResult
from qmapnav.evaluation import rank_fix_candidates
from qmapnav.evaluation import StageEvidence
from qmapnav.language import parse_question


FIXTURE = Path(__file__).parent / 'fixtures' / 'released_questions.json'


def _manifest():
    return build_object_reference_manifest(load_questions(FIXTURE), parse_question)


def test_released_manifest_has_stable_thirty_case_distribution() -> None:
    cases = _manifest()

    assert len(cases) == 30
    assert len({item.scene_id for item in cases}) == 15
    assert all(
        sum(other.scene_id == item.scene_id for other in cases) == 2
        for item in cases
    )
    assert all(item.expected_target_class for item in cases)
    assert len(manifest_digest(cases)) == 64
    assert manifest_digest(cases) == manifest_digest(cases)


def test_manifest_tags_released_relation_and_colour_cases() -> None:
    by_id = {item.case_id: item for item in _manifest()}

    assert 'between' in by_id[
        'japanese_room_object_reference_01'
    ].tags
    assert 'colour' in by_id[
        'japanese_room_object_reference_02'
    ].tags
    assert 'structural_anchor' in by_id[
        'hotel_room_1_object_reference_01'
    ].tags


@pytest.mark.parametrize(
    ('changes', 'category'),
    (
        ({'parser_correct': False}, 'parsing'),
        ({'parser_correct': True, 'target_observed': False}, 'missed_target'),
        ({'target_detected': False}, 'missed_target'),
        ({'anchors_available': False}, 'missed_anchor'),
        ({'target_lifted': False}, 'bad_lifting'),
        ({'identity_correct': False}, 'duplicate_instance'),
        ({'colour_correct': False}, 'incorrect_colour'),
        ({'relation_correct': False}, 'bad_relation'),
        ({'target_selected_correctly': False}, 'bad_relation'),
        ({'obb_acceptable': False}, 'incorrect_obb'),
        ({'protocol_valid': False}, 'protocol_failure'),
    ),
)
def test_failure_classifier_uses_earliest_cause(changes, category) -> None:
    base = StageEvidence(
        parser_correct=True,
        target_observed=True,
        target_detected=True,
        anchors_available=True,
        target_lifted=True,
        identity_correct=True,
        colour_correct=True,
        relation_correct=True,
        target_selected_correctly=True,
        obb_acceptable=True,
        protocol_valid=True,
    )

    result = classify_primary_failure(replace(base, **changes))

    assert result.category == category


def test_failure_classifier_does_not_hide_earlier_anchor_failure() -> None:
    evidence = StageEvidence(
        parser_correct=True,
        target_observed=True,
        target_detected=True,
        anchors_available=False,
        relation_correct=False,
        protocol_valid=False,
        detail={'anchor_subtype': 'anchor_structural_mapping_failure'},
    )

    result = classify_primary_failure(evidence)

    assert result.category == 'missed_anchor'
    assert result.subtype == 'anchor_structural_mapping_failure'


def test_episode_result_enforces_single_marker_publication() -> None:
    common = {
        'run_id': 'run',
        'case_id': 'case',
        'scene_id': 'scene',
        'question': 'Find the chair.',
        'pipeline_mode': 'synthetic',
        'episode_status': 'completed',
        'parser_mode': 'full',
        'task_specification': {},
        'requested_classes': ('chair',),
        'stage_evidence': StageEvidence(protocol_valid=True),
    }

    with pytest.raises(ValueError, match='at most once'):
        ObjectReferenceEpisodeResult(
            **common,
            marker_published=True,
            marker_publish_count=2,
        )
    with pytest.raises(ValueError, match='match publish count'):
        ObjectReferenceEpisodeResult(
            **common,
            marker_published=False,
            marker_publish_count=1,
        )


def test_fix_ranking_uses_expected_score_effort_and_risk() -> None:
    broad = FixCandidate('anchor', 5, 'improve association', 0.8, 2.0, 1.0)
    narrow = FixCandidate('rare miss', 1, 'fallback detector', 0.5, 4.0, 2.0)

    ranked = rank_fix_candidates((narrow, broad))

    assert ranked[0] is broad
    assert broad.expected_recovered_score == 8.0
    assert broad.priority == 4.0
