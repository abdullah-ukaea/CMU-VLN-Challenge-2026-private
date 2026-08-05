"""Tests for the six-term viewpoint score and ranked selection."""

from math import atan2
from math import pi

from day11_helpers import add_unknown
from day11_helpers import add_wall
from day11_helpers import open_grid
import pytest

from qmapnav.exploration import ExplorationNeed
from qmapnav.exploration import score_candidate
from qmapnav.exploration import select_viewpoint
from qmapnav.exploration import ViewpointCandidate
from qmapnav.exploration import ViewpointScoringConfig
from qmapnav.exploration.viewpoint_scoring import distance_quality


def _need(need_type: str = 'missing_target') -> ExplorationNeed:
    return ExplorationNeed(
        need_type=need_type,
        target_reference_id='target_1',
        reason='unit test need',
    )


def _facing(identifier, position, target, *, travel=1.0, source='object_annulus'):
    yaw = atan2(target[1] - position[1], target[0] - position[0])
    return ViewpointCandidate(
        viewpoint_id=identifier,
        pose_xy_yaw=(position[0], position[1], yaw),
        source=source,
        travel_cost_m=travel,
    )


def test_distance_quality_peaks_near_the_preferred_standoff() -> None:
    config = ViewpointScoringConfig()
    preferred = config.preferred_observation_distance_m
    assert distance_quality(preferred, config) == pytest.approx(1.0)
    # Pressed against the object the surface leaves the field of view.
    assert distance_quality(0.2, config) < 0.3
    # Beyond the useful range nothing is legible.
    assert distance_quality(config.maximum_useful_distance_m, config) == 0.0
    assert distance_quality(float('nan'), config) == 0.0


def test_target_behind_a_wall_scores_zero_visibility() -> None:
    grid = open_grid(half_extent=6.0)
    add_wall(grid, (-0.3, -4.0, 0.3, 4.0))
    target = (2.0, 0.0)
    blocked = score_candidate(
        _facing('blocked', (-2.0, 0.0), target),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    clear = score_candidate(
        _facing('clear', (3.8, 0.0), target),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    assert blocked.score_terms.target_visibility == 0.0
    assert clear.score_terms.target_visibility > 0.5
    assert clear.score > blocked.score


def test_longer_planned_paths_receive_a_larger_penalty() -> None:
    grid = open_grid(half_extent=8.0)
    target = (2.0, 0.0)
    near = score_candidate(
        _facing('near', (0.2, 0.0), target, travel=1.0),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    far = score_candidate(
        _facing('far', (0.2, 0.0), target, travel=5.0),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    assert far.score_terms.travel_cost > near.score_terms.travel_cost
    # Identical evidence, so only travel can separate them.
    assert near.score > far.score


def test_a_bearing_separating_view_beats_a_redundant_one() -> None:
    grid = open_grid(half_extent=8.0)
    first = (2.0, 0.6)
    second = (2.0, -0.6)
    midpoint = (2.0, 0.0)
    # Far away both candidates collapse onto nearly the same bearing.
    redundant = score_candidate(
        _facing('redundant', (-5.0, 0.0), midpoint),
        grid=grid,
        need=_need('ambiguous_target'),
        hypothesis_xy=(first, second),
    )
    # Close and to the side they separate widely.
    separating = score_candidate(
        _facing('separating', (1.2, 0.0), midpoint),
        grid=grid,
        need=_need('ambiguous_target'),
        hypothesis_xy=(first, second),
    )
    assert (
        separating.score_terms.ambiguity_reduction
        > redundant.score_terms.ambiguity_reduction
    )
    assert separating.score > redundant.score


def test_ambiguity_term_is_zero_without_two_hypotheses() -> None:
    grid = open_grid(half_extent=6.0)
    scored = score_candidate(
        _facing('single', (0.0, 0.0), (2.0, 0.0)),
        grid=grid,
        need=_need('ambiguous_target'),
        hypothesis_xy=((2.0, 0.0),),
    )
    assert scored.score_terms.ambiguity_reduction == 0.0


def test_frontal_shelf_view_beats_a_rear_blocked_view() -> None:
    grid = open_grid(half_extent=6.0)
    shelf = (0.0, 2.0)
    # A solid wall directly behind the shelf blocks any rear approach.
    add_wall(grid, (-2.0, 2.6, 2.0, 3.2))
    frontal = score_candidate(
        _facing('frontal', (0.0, 0.4), shelf, source='support_surface'),
        grid=grid,
        need=_need('support_surface_search'),
        support_xy=shelf,
    )
    rear = score_candidate(
        _facing('rear', (0.0, 4.0), shelf, source='support_surface'),
        grid=grid,
        need=_need('support_surface_search'),
        support_xy=shelf,
    )
    assert rear.score_terms.support_visibility == 0.0
    assert frontal.score_terms.support_visibility > 0.5
    assert frontal.score > rear.score


def test_unexplored_gain_rewards_facing_unknown_space() -> None:
    grid = open_grid(half_extent=6.0)
    add_unknown(grid, (1.0, -5.0, 6.0, 5.0))
    toward = score_candidate(
        _facing('toward', (0.0, 0.0), (5.0, 0.0), source='frontier'),
        grid=grid,
        need=_need('unexplored_region'),
    )
    away = score_candidate(
        _facing('away', (0.0, 0.0), (-5.0, 0.0), source='frontier'),
        grid=grid,
        need=_need('unexplored_region'),
    )
    assert toward.score_terms.unexplored_gain > away.score_terms.unexplored_gain


def test_query_relevance_outweighs_a_large_empty_frontier() -> None:
    grid = open_grid(half_extent=8.0)
    add_unknown(grid, (2.0, -8.0, 8.0, 8.0))
    target = (-2.0, 0.0)
    frontier_view = score_candidate(
        _facing('frontier', (1.0, 0.0), (6.0, 0.0), source='frontier'),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    target_view = score_candidate(
        _facing('target', (-0.2, 0.0), target),
        grid=grid,
        need=_need(),
        target_xy=target,
    )
    assert target_view.score > frontier_view.score


def test_anchor_visibility_scores_the_best_visible_anchor() -> None:
    grid = open_grid(half_extent=6.0)
    anchors = ((0.0, 2.0), (4.5, 0.0))
    scored = score_candidate(
        _facing('anchor', (0.0, 0.3), (0.0, 2.0)),
        grid=grid,
        need=_need('missing_anchor'),
        anchor_targets=anchors,
    )
    assert scored.score_terms.anchor_visibility > 0.0
    empty = score_candidate(
        _facing('none', (0.0, 0.3), (0.0, 2.0)),
        grid=grid,
        need=_need('missing_anchor'),
        anchor_targets=(),
    )
    assert empty.score_terms.anchor_visibility == 0.0


def test_traversal_risk_reduces_the_final_score() -> None:
    grid = open_grid(half_extent=6.0)
    target = (2.0, 0.0)
    safe = score_candidate(
        _facing('safe', (0.2, 0.0), target),
        grid=grid,
        need=_need(),
        target_xy=target,
        traversal_risk=0.0,
    )
    risky = score_candidate(
        _facing('risky', (0.2, 0.0), target),
        grid=grid,
        need=_need(),
        target_xy=target,
        traversal_risk=0.8,
    )
    assert risky.score < safe.score


def test_every_term_is_individually_inspectable_in_the_trace() -> None:
    grid = open_grid(half_extent=6.0)
    scored = score_candidate(
        _facing('trace', (0.2, 0.0), (2.0, 0.0)),
        grid=grid,
        need=_need(),
        target_xy=(2.0, 0.0),
    )
    payload = scored.to_dict()['score_terms']
    assert set(payload) == {
        'target_visibility',
        'anchor_visibility',
        'unexplored_gain',
        'ambiguity_reduction',
        'support_visibility',
        'travel_cost',
        'traversal_risk',
    }


def test_selection_ranks_by_score_and_reports_selected() -> None:
    need = _need()
    weak = ViewpointCandidate(
        viewpoint_id='weak', pose_xy_yaw=(0.0, 0.0, 0.0),
        source='frontier', score=0.2,
    )
    strong = ViewpointCandidate(
        viewpoint_id='strong', pose_xy_yaw=(1.0, 0.0, 0.0),
        source='object_annulus', score=0.9,
    )
    selection = select_viewpoint((weak, strong), need)
    assert selection.selection_status == 'selected'
    assert selection.selected_viewpoint_id == 'strong'
    assert [item.viewpoint_id for item in selection.ranked_candidates] == [
        'strong', 'weak',
    ]


def test_selection_refuses_when_gain_is_too_low() -> None:
    need = _need()
    poor = ViewpointCandidate(
        viewpoint_id='poor', pose_xy_yaw=(0.0, 0.0, 0.0),
        source='frontier', score=0.01,
    )
    selection = select_viewpoint((poor,), need)
    assert selection.selection_status == 'gain_too_low'
    assert selection.selected_viewpoint_id is None
    # The rejected candidate is still reported for the trace.
    assert selection.ranked_candidates


def test_selection_reports_no_reachable_viewpoint_and_budget_status() -> None:
    need = _need()
    assert select_viewpoint((), need).selection_status == (
        'no_reachable_viewpoint'
    )
    good = ViewpointCandidate(
        viewpoint_id='good', pose_xy_yaw=(0.0, 0.0, 0.0),
        source='frontier', score=0.9,
    )
    exhausted = select_viewpoint(
        (good,), need, budget_status='time_budget_exhausted'
    )
    assert exhausted.selection_status == 'time_budget_exhausted'
    assert exhausted.selected_viewpoint_id is None
    unknown = select_viewpoint((good,), need, budget_status='viewpoints_exhausted')
    assert unknown.selection_status == 'budget_exhausted'


def test_selection_is_deterministic_for_tied_scores() -> None:
    need = _need()
    first = ViewpointCandidate(
        viewpoint_id='b', pose_xy_yaw=(0.0, 0.0, 0.0),
        source='frontier', score=0.5,
    )
    second = ViewpointCandidate(
        viewpoint_id='a', pose_xy_yaw=(1.0, 0.0, 0.0),
        source='frontier', score=0.5,
    )
    selection = select_viewpoint((first, second), need)
    assert selection.selected_viewpoint_id == 'a'


def test_scoring_config_rejects_inconsistent_distances() -> None:
    with pytest.raises(ValueError, match='preferred observation distance'):
        ViewpointScoringConfig(
            preferred_observation_distance_m=6.0,
            maximum_useful_distance_m=3.0,
        )
    with pytest.raises(ValueError, match='target_weight'):
        ViewpointScoringConfig(target_weight=0.0)


def test_fov_centring_prefers_a_target_near_image_centre() -> None:
    grid = open_grid(half_extent=6.0)
    target = (2.0, 0.0)
    centred = score_candidate(
        ViewpointCandidate(
            viewpoint_id='centred', pose_xy_yaw=(0.2, 0.0, 0.0),
            source='object_annulus', travel_cost_m=1.0,
        ),
        grid=grid, need=_need(), target_xy=target,
    )
    turned_away = score_candidate(
        ViewpointCandidate(
            viewpoint_id='turned', pose_xy_yaw=(0.2, 0.0, pi / 2.0),
            source='object_annulus', travel_cost_m=1.0,
        ),
        grid=grid, need=_need(), target_xy=target,
    )
    assert (
        centred.score_terms.target_visibility
        > turned_away.score_terms.target_visibility
    )
