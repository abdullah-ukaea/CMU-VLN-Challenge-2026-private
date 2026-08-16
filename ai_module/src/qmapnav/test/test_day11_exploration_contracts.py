"""Tests for the Day 11 exploration contracts and budget."""

import pytest

from qmapnav.exploration import ExplorationBudget
from qmapnav.exploration import ExplorationBudgetTracker
from qmapnav.exploration import ExplorationNeed
from qmapnav.exploration import ViewpointCandidate
from qmapnav.exploration import ViewpointScoreTerms
from qmapnav.exploration import ViewpointSelection


def _need(need_type: str = 'missing_target') -> ExplorationNeed:
    return ExplorationNeed(
        need_type=need_type,
        target_reference_id='book_1',
        missing_classes=('book',),
        ambiguity_score=1.0,
        urgency=0.8,
        expected_task_value=2.0,
        reason='target likely lies on a detected support surface',
    )


def _candidate(identifier: str = 'vp_0', score: float = 1.0):
    return ViewpointCandidate(
        viewpoint_id=identifier,
        pose_xy_yaw=(1.0, 2.0, 0.5),
        source='support_surface',
        score_terms=ViewpointScoreTerms(target_visibility=0.7),
        score=score,
        reasons=('best visible angle onto shelf_2',),
    )


def test_need_rejects_unknown_type_and_blank_reason() -> None:
    with pytest.raises(ValueError, match='need_type'):
        ExplorationNeed(need_type='wander', reason='because')
    with pytest.raises(ValueError, match='reason'):
        ExplorationNeed(need_type='missing_target', reason='  ')


def test_need_rejects_out_of_range_scores_and_duplicates() -> None:
    with pytest.raises(ValueError, match='ambiguity_score'):
        ExplorationNeed(
            need_type='missing_target', reason='r', ambiguity_score=1.5
        )
    with pytest.raises(ValueError, match='expected_task_value'):
        ExplorationNeed(
            need_type='missing_target', reason='r', expected_task_value=-1.0
        )
    with pytest.raises(ValueError, match='duplicates'):
        ExplorationNeed(
            need_type='missing_target',
            reason='r',
            missing_classes=('book', 'book'),
        )


def test_need_reports_absent_entity_and_serializes() -> None:
    need = _need('small_object_search')
    assert need.seeks_absent_entity is True
    assert _need('ambiguous_target').seeks_absent_entity is False
    payload = need.to_dict()
    assert payload['need_type'] == 'small_object_search'
    assert payload['missing_classes'] == ['book']


def test_score_terms_reject_values_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match='travel_cost'):
        ViewpointScoreTerms(travel_cost=2.0)
    terms = ViewpointScoreTerms(target_visibility=0.72, support_visibility=0.89)
    assert terms.to_dict()['support_visibility'] == 0.89


def test_candidate_validates_pose_and_source() -> None:
    with pytest.raises(ValueError, match='source'):
        ViewpointCandidate(
            viewpoint_id='vp', pose_xy_yaw=(0.0, 0.0, 0.0), source='guess'
        )
    with pytest.raises(ValueError, match='pose_xy_yaw'):
        ViewpointCandidate(
            viewpoint_id='vp', pose_xy_yaw=(0.0, 0.0), source='frontier'
        )
    with pytest.raises(ValueError, match='travel_cost_m'):
        ViewpointCandidate(
            viewpoint_id='vp',
            pose_xy_yaw=(0.0, 0.0, 0.0),
            source='frontier',
            travel_cost_m=-1.0,
        )


def test_selection_requires_status_to_agree_with_selected_id() -> None:
    need = _need()
    candidate = _candidate()
    with pytest.raises(ValueError, match='selected_viewpoint_id'):
        ViewpointSelection(
            ranked_candidates=(candidate,),
            selected_viewpoint_id=None,
            selection_status='selected',
            unresolved_need=need,
        )
    with pytest.raises(ValueError, match='selected_viewpoint_id'):
        ViewpointSelection(
            ranked_candidates=(candidate,),
            selected_viewpoint_id='vp_0',
            selection_status='gain_too_low',
            unresolved_need=need,
        )
    with pytest.raises(ValueError, match='must name a candidate'):
        ViewpointSelection(
            ranked_candidates=(candidate,),
            selected_viewpoint_id='vp_missing',
            selection_status='selected',
            unresolved_need=need,
        )


def test_selection_exposes_selected_candidate() -> None:
    selection = ViewpointSelection(
        ranked_candidates=(_candidate('vp_0'), _candidate('vp_1', 0.4)),
        selected_viewpoint_id='vp_0',
        selection_status='selected',
        unresolved_need=_need(),
        expected_gain=0.68,
    )
    assert selection.selected.viewpoint_id == 'vp_0'
    assert selection.to_dict()['selection_status'] == 'selected'


def test_selection_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match='unique'):
        ViewpointSelection(
            ranked_candidates=(_candidate('vp_0'), _candidate('vp_0')),
            selected_viewpoint_id=None,
            selection_status='gain_too_low',
            unresolved_need=_need(),
        )


def test_budget_is_more_conservative_for_instruction_tasks() -> None:
    instruction = ExplorationBudget.for_task_type('instruction_following')
    reference = ExplorationBudget.for_task_type('object_reference')
    assert (
        instruction.max_total_exploration_distance_m
        < reference.max_total_exploration_distance_m
    )
    assert (
        instruction.max_targeted_viewpoints
        <= reference.max_targeted_viewpoints
    )
    assert (
        instruction.minimum_time_remaining_sec
        >= reference.minimum_time_remaining_sec
    )
    with pytest.raises(ValueError, match='unsupported task type'):
        ExplorationBudget.for_task_type('freestyle')


def test_budget_rejects_single_viewpoint_exceeding_total() -> None:
    with pytest.raises(ValueError, match='total travel budget'):
        ExplorationBudget(
            max_single_viewpoint_distance_m=10.0,
            max_total_exploration_distance_m=5.0,
        )


def test_tracker_reports_each_exhaustion_reason() -> None:
    tracker = ExplorationBudgetTracker(
        ExplorationBudget(
            max_targeted_viewpoints=2,
            max_single_viewpoint_distance_m=3.0,
            max_total_exploration_distance_m=5.0,
            max_exploration_time_sec=30.0,
            minimum_time_remaining_sec=100.0,
        )
    )
    assert tracker.allows(500.0) is True
    assert tracker.status(50.0) == 'time_budget_exhausted'

    assert tracker.permits_travel(2.5) is True
    assert tracker.permits_travel(4.0) is False

    tracker.consume(distance_m=3.0, duration_sec=10.0)
    assert tracker.viewpoints_used == 1
    assert tracker.remaining_distance_m == pytest.approx(2.0)
    assert tracker.permits_travel(2.5) is False

    tracker.consume(distance_m=2.0, duration_sec=10.0)
    assert tracker.status(500.0) == 'viewpoints_exhausted'
    with pytest.raises(RuntimeError, match='budget already exhausted'):
        tracker.consume(distance_m=0.5, duration_sec=1.0)


def test_tracker_reports_distance_exhaustion_before_viewpoint_limit() -> None:
    tracker = ExplorationBudgetTracker(
        ExplorationBudget(
            max_targeted_viewpoints=3,
            max_single_viewpoint_distance_m=4.0,
            max_total_exploration_distance_m=4.0,
        )
    )
    tracker.consume(distance_m=4.0, duration_sec=5.0)
    assert tracker.status(500.0) == 'distance_exhausted'
