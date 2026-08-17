"""Count-integer, identity-set, viewpoint, and deadline stability tests."""

from fixtures import numerical_result

from qmapnav.counting import CountStabilityConfig
from qmapnav.counting import CountStabilityMachine
from qmapnav.counting import CountStabilityStatus


def test_same_integer_with_churning_ids_never_finalises() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=2,
        required_independent_viewpoints=2,
    ))
    first = machine.update(
        numerical_result((1, 2)), viewpoint_id='a',
        time_remaining_sec=500.0, episode_time_sec=10.0,
    )
    second = machine.update(
        numerical_result((3, 4)), viewpoint_id='b',
        time_remaining_sec=490.0, episode_time_sec=20.0,
    )
    assert first.current_count == second.current_count == 2
    assert second.consecutive_stable_updates == 1
    assert second.status is CountStabilityStatus.CANDIDATE_COUNT
    assert second.stable is False


def test_independent_viewpoint_and_id_set_stability_finalises() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=3,
        required_independent_viewpoints=2,
    ))
    for index, viewpoint in enumerate(('a', 'a', 'b')):
        state = machine.update(
            numerical_result((1, 2)), viewpoint_id=viewpoint,
            time_remaining_sec=500.0, episode_time_sec=float(index),
        )
    assert state.status is CountStabilityStatus.STABLE
    assert state.should_publish
    assert state.result.stable


def test_stationary_repeated_frames_are_not_independent_views() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=2,
        required_independent_viewpoints=2,
    ))
    for index in range(3):
        state = machine.update(
            numerical_result((1,)), viewpoint_id='same_pose',
            time_remaining_sec=500.0, episode_time_sec=float(index),
        )
    assert state.independent_viewpoints == 1
    assert state.stable is False


def test_time_budget_exhaustion_publishes_best_available() -> None:
    machine = CountStabilityMachine()
    state = machine.update(
        numerical_result((1,), confidence=0.4, unresolved=(2,)),
        viewpoint_id='a', time_remaining_sec=20.0, episode_time_sec=580.0,
    )
    assert state.status is CountStabilityStatus.BEST_AVAILABLE_COUNT
    assert state.should_publish
    assert 'time_budget_low' in state.reason
    assert CountStabilityStatus.TIME_BUDGET_LOW in machine.transition_history


def test_new_discovery_resets_previous_stability_progress() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=3,
        required_independent_viewpoints=2,
    ))
    machine.update(
        numerical_result((1,)), viewpoint_id='a',
        time_remaining_sec=500.0, episode_time_sec=1.0,
    )
    machine.update(
        numerical_result((1,)), viewpoint_id='b',
        time_remaining_sec=499.0, episode_time_sec=2.0,
    )
    state = machine.update(
        numerical_result((1, 2)), viewpoint_id='c',
        time_remaining_sec=498.0, episode_time_sec=3.0,
    )
    assert state.consecutive_stable_updates == 1
    assert state.independent_viewpoint_ids == ('c',)
