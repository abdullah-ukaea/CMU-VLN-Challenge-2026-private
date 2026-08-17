"""Tests for complete persistent-map object-reference resolution."""

from fixtures import candidate
from fixtures import geometry

from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.reasoning.candidate_generation import CandidateGenerationResult
from qmapnav.reasoning.object_reference_solver import (
    resolve_object_reference_from_maps,
)


def _result(reference_id, *candidates):
    return CandidateGenerationResult(reference_id, tuple(candidates), None, True)


def test_complete_relation_set_beats_closest_only_distractor(monkeypatch) -> None:
    task = parse_question(
        'Find the chair between the table and sink that is closest to the window.'
    )
    entities = {item.class_name: item.entity_id for item in task.entities}
    good = candidate(
        '7', geometry('7', 2.0, 0.1, semantic_class='chair'), colour=0.8
    )
    distractor = candidate(
        '4', geometry('4', 0.0, 4.5, semantic_class='chair'), colour=0.8
    )
    table = candidate(
        '10', geometry('10', 0.0, 0.0, semantic_class='table')
    )
    sink = candidate(
        '11', geometry('11', 4.0, 0.0, semantic_class='sink')
    )
    window = candidate(
        'anchor_window',
        geometry(
            'anchor_window', 2.0, 5.0, semantic_class='window'
        ),
    )
    generated = {
        entities['chair']: _result(entities['chair'], good, distractor),
        entities['table']: _result(entities['table'], table),
        entities['sink']: _result(entities['sink'], sink),
        entities['window']: _result(entities['window'], window),
    }

    monkeypatch.setattr(
        'qmapnav.reasoning.object_reference_solver.'
        'generate_candidates_from_maps',
        lambda reference, *args, **kwargs: generated[reference.entity_id],
    )

    resolution = resolve_object_reference_from_maps(
        task, ObjectMap(), StructuralMap()
    )

    assert resolution.selected_target_id == '7'
    assert resolution.ranked_hypotheses[0].target_id == '7'
    assert any(
        '.between.' in item
        for item in resolution.ranked_hypotheses[0].satisfied_constraints
    )


def test_missing_anchor_returns_best_target_fallback(monkeypatch) -> None:
    task = parse_question('Find the flowers near the window.')
    target, anchor = task.entities
    flowers = candidate(
        '3', geometry('3', 1.0, 1.0, semantic_class='flowers')
    )
    generated = {
        target.entity_id: _result(target.entity_id, flowers),
        anchor.entity_id: _result(anchor.entity_id),
    }
    monkeypatch.setattr(
        'qmapnav.reasoning.object_reference_solver.'
        'generate_candidates_from_maps',
        lambda reference, *args, **kwargs: generated[reference.entity_id],
    )

    resolution = resolve_object_reference_from_maps(
        task, ObjectMap(), StructuralMap()
    )

    assert resolution.selected_target_id == '3'
    assert resolution.used_fallback
    assert resolution.resolution_status == 'low_confidence'
    assert resolution.unresolved_constraints == (
        f'{anchor.entity_id}.candidate_missing',
    )


def test_no_target_candidates_returns_bounded_no_candidate_result(
    monkeypatch,
) -> None:
    task = parse_question('Find the picture closest to the bench.')
    generated = {
        item.entity_id: _result(item.entity_id) for item in task.entities
    }
    monkeypatch.setattr(
        'qmapnav.reasoning.object_reference_solver.'
        'generate_candidates_from_maps',
        lambda reference, *args, **kwargs: generated[reference.entity_id],
    )

    resolution = resolve_object_reference_from_maps(
        task, ObjectMap(), StructuralMap()
    )

    assert resolution.selected_target_id is None
    assert resolution.resolution_status == 'no_candidates'
    assert resolution.used_fallback


def test_repeated_anchor_mentions_may_share_one_physical_instance(
    monkeypatch,
) -> None:
    task = parse_question(
        'Find the speaker on the TV cabinet closest to the potted plant '
        'on the TV cabinet.'
    )
    speaker, first_cabinet, plant, second_cabinet = task.entities
    speaker_candidate = candidate(
        'speaker', geometry('speaker', 0.0, 0.0, semantic_class='speaker')
    )
    cabinet_candidate = candidate(
        'cabinet', geometry('cabinet', 0.0, 0.0,
                            semantic_class='tv_cabinet')
    )
    plant_candidate = candidate(
        'plant', geometry('plant', 0.8, 0.0,
                          semantic_class='potted_plant')
    )
    generated = {
        speaker.entity_id: _result(speaker.entity_id, speaker_candidate),
        first_cabinet.entity_id: _result(
            first_cabinet.entity_id, cabinet_candidate
        ),
        plant.entity_id: _result(plant.entity_id, plant_candidate),
        second_cabinet.entity_id: _result(
            second_cabinet.entity_id, cabinet_candidate
        ),
    }
    monkeypatch.setattr(
        'qmapnav.reasoning.object_reference_solver.'
        'generate_candidates_from_maps',
        lambda reference, *args, **kwargs: generated[reference.entity_id],
    )

    resolution = resolve_object_reference_from_maps(
        task, ObjectMap(), StructuralMap()
    )

    assert resolution.selected_target_id == 'speaker'
    assert resolution.ranked_hypotheses
    roles = resolution.ranked_hypotheses[0].role_ids
    assert roles[first_cabinet.entity_id] == 'cabinet'
    assert roles[second_cabinet.entity_id] == 'cabinet'


def test_released_query_vocabulary_matches_annotated_map_labels() -> None:
    assert canonicalize_class_name('flowers') == 'flower'
    assert canonicalize_class_name('nightstand') == 'night_stand'
    assert canonicalize_class_name('zen stone decoration') == 'stone_decoration'
