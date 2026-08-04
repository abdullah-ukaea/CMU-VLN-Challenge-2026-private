"""Tests for the bounded object-reference episode coordinator."""

import numpy as np
import pytest

from qmapnav.common import ObjectInstance
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mission import ObjectReferenceEpisodeCoordinator
from qmapnav.mission import ObjectReferenceEpisodeState
from qmapnav.reasoning.candidate_generation import CandidateGenerationResult
from qmapnav.reasoning.object_reference_solver import (
    PerceivedObjectReferenceResolution,
)


def _resolution(task, selected=None, margin=0.0):
    generated = {
        item.entity_id: CandidateGenerationResult(
            item.entity_id, (), item.cardinality, False
        )
        for item in task.entities
    }
    return PerceivedObjectReferenceResolution(
        task.entities[0].entity_id,
        generated,
        (),
        selected,
        margin,
        margin,
        'no_candidates' if selected is None else 'low_confidence',
        ('target_candidate_missing',) if selected is None else (),
        True,
    )


def _instance():
    return ObjectInstance(
        7,
        {'flowers': 1.0},
        {},
        np.array([2.0, 0.0, 0.5]),
        np.array([1.8, -0.2, 0.0]),
        np.array([2.2, 0.2, 1.0]),
        np.array([0.4, 0.4, 1.0]),
        0.0,
        0.8,
        2,
        0.9,
    )


def test_no_target_requests_one_viewpoint_then_commits_once(monkeypatch) -> None:
    task = parse_question('Find the flowers near the window.')
    outputs = iter((_resolution(task), _resolution(task, selected='7')))
    coordinator = ObjectReferenceEpisodeCoordinator(
        resolver=lambda *args: next(outputs)
    )
    object_map = ObjectMap()
    structural_map = StructuralMap()
    monkeypatch.setattr(
        'qmapnav.mission.object_reference_episode._selected_instance',
        lambda resolution, object_map: (
            _instance() if resolution.selected_target_id else None
        ),
    )
    coordinator.start(task)

    initial = coordinator.evaluate_initial(
        object_map,
        structural_map,
        current_pose_xy_heading=(0.0, 0.0, 0.0),
        time_remaining_sec=100.0,
        safe_pose=lambda x, y: True,
    )

    assert initial.action == 'viewpoint'
    assert coordinator.viewpoint_attempted
    coordinator.notify_viewpoint_arrived()
    final = coordinator.evaluate_after_reobservation(
        object_map, structural_map
    )
    assert final.action == 'commit'
    assert final.selected_instance.instance_id == 7
    assert coordinator.state is ObjectReferenceEpisodeState.COMMITTED
    with pytest.raises(RuntimeError, match='exactly once'):
        coordinator.evaluate_after_reobservation(object_map, structural_map)


def test_no_safe_viewpoint_terminates_with_no_candidate() -> None:
    task = parse_question('Find the flowers near the window.')
    coordinator = ObjectReferenceEpisodeCoordinator(
        resolver=lambda *args: _resolution(task)
    )
    coordinator.start(task)

    action = coordinator.evaluate_initial(
        ObjectMap(),
        StructuralMap(),
        current_pose_xy_heading=(0.0, 0.0, 0.0),
        time_remaining_sec=100.0,
        safe_pose=lambda x, y: False,
    )

    assert action.action == 'no_candidate'
    assert action.reason == 'no_safe_targeted_viewpoint'
    assert coordinator.state is ObjectReferenceEpisodeState.COMMITTED


def test_late_episode_skips_viewpoint_and_commits_best(monkeypatch) -> None:
    task = parse_question('Find the flowers near the window.')
    coordinator = ObjectReferenceEpisodeCoordinator(
        resolver=lambda *args: _resolution(task, selected='7')
    )
    monkeypatch.setattr(
        'qmapnav.mission.object_reference_episode._selected_instance',
        lambda resolution, object_map: _instance(),
    )
    coordinator.start(task)

    action = coordinator.evaluate_initial(
        ObjectMap(),
        StructuralMap(),
        current_pose_xy_heading=(0.0, 0.0, 0.0),
        time_remaining_sec=10.0,
        safe_pose=lambda x, y: True,
    )

    assert action.action == 'commit'
    assert action.reason == 'insufficient_time_reserve'
    assert not coordinator.viewpoint_attempted
