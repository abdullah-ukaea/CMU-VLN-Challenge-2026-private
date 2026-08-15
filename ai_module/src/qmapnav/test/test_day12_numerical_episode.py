"""Bounded ROS-independent numerical episode coordinator tests."""

from day12_helpers import numerical_result

from qmapnav.counting import CountStabilityConfig
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mission.numerical_episode import NumericalEpisodeCoordinator
from qmapnav.mission.numerical_episode import NumericalEpisodeState


def test_episode_commits_after_stable_independent_views() -> None:
    coordinator = NumericalEpisodeCoordinator(
        stability_config=CountStabilityConfig(
            required_consecutive_updates=2,
            required_independent_viewpoints=2,
        ),
        resolver=lambda *args: numerical_result((1, 2)),
    )
    coordinator.start(parse_question('How many chairs are there?'))
    first = coordinator.evaluate(
        ObjectMap(), StructuralMap(), viewpoint_id='a',
        time_remaining_sec=500.0, episode_time_sec=10.0,
    )
    second = coordinator.evaluate(
        ObjectMap(), StructuralMap(), viewpoint_id='b',
        time_remaining_sec=490.0, episode_time_sec=20.0,
    )
    assert first.action == 'observe'
    assert second.action == 'commit'
    assert second.result.stable
    assert coordinator.state is NumericalEpisodeState.COMMITTED


def test_episode_force_commit_never_returns_silence() -> None:
    coordinator = NumericalEpisodeCoordinator(
        resolver=lambda *args: numerical_result((), confidence=0.3)
    )
    coordinator.start(parse_question('How many cups are there?'))
    action = coordinator.force_commit(
        ObjectMap(), StructuralMap(), reason='watchdog_reserve'
    )
    assert action.action == 'commit'
    assert action.result.count == 0
    assert 'best_available' in action.reason


def test_episode_marks_transport_publication_terminal() -> None:
    coordinator = NumericalEpisodeCoordinator(
        resolver=lambda *args: numerical_result((1,))
    )
    coordinator.start(parse_question('How many chairs are there?'))
    coordinator.force_commit(
        ObjectMap(), StructuralMap(), reason='test_commit'
    )
    coordinator.notify_published()
    assert coordinator.stability.state.status.value == 'published'
