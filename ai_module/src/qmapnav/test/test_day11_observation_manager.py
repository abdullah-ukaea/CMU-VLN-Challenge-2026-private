"""Tests for the bounded settle-accumulate-select observation manager."""

import pytest

from qmapnav.exploration.observation_manager import ObservationConfig
from qmapnav.exploration.observation_manager import ObservationManager
from qmapnav.exploration.observation_manager import ObservationState
from qmapnav.exploration.observation_manager import PanoramaOffer


def _manager() -> ObservationManager:
    return ObservationManager(
        ObservationConfig(
            settle_time_sec=0.75,
            scan_accumulation_time_sec=2.5,
            panoramas_to_consider=3,
        )
    )


def test_phases_advance_settle_then_accumulate_then_complete() -> None:
    manager = _manager()
    assert manager.state is ObservationState.IDLE
    manager.begin(100.0)
    assert manager.state is ObservationState.SETTLING

    assert manager.update(100.5) is ObservationState.SETTLING
    assert manager.update(100.8) is ObservationState.ACCUMULATING
    assert manager.update(102.0) is ObservationState.ACCUMULATING
    assert manager.update(103.3) is ObservationState.COMPLETE

    result = manager.result()
    assert result.status == 'complete'
    assert result.duration_sec == pytest.approx(3.3)


def test_panoramas_are_only_collected_while_accumulating() -> None:
    manager = _manager()
    manager.begin(0.0)
    assert manager.offer_panorama(PanoramaOffer('early', 1.0, 0.1)) is False
    manager.update(0.8)
    assert manager.offer_panorama(PanoramaOffer('good', 1.0, 1.0)) is True
    manager.update(3.3)
    assert manager.offer_panorama(PanoramaOffer('late', 9.0, 3.4)) is False
    assert manager.result().selected_panorama_id == 'good'


def test_sharpest_panorama_wins_within_the_considered_window() -> None:
    manager = _manager()
    manager.begin(0.0)
    manager.update(0.8)
    manager.offer_panorama(PanoramaOffer('blurry', 0.2, 1.0))
    manager.offer_panorama(PanoramaOffer('sharp', 0.9, 1.5))
    manager.offer_panorama(PanoramaOffer('medium', 0.5, 2.0))
    manager.update(3.3)
    result = manager.result()
    assert result.selected_panorama_id == 'sharp'
    assert result.panoramas_considered == 3


def test_only_the_newest_configured_panoramas_are_retained() -> None:
    manager = ObservationManager(
        ObservationConfig(panoramas_to_consider=2)
    )
    manager.begin(0.0)
    manager.update(0.8)
    # The sharpest is also the oldest, and falls out of the window.
    manager.offer_panorama(PanoramaOffer('oldest', 5.0, 1.0))
    manager.offer_panorama(PanoramaOffer('middle', 0.4, 1.5))
    manager.offer_panorama(PanoramaOffer('newest', 0.6, 2.0))
    manager.update(3.3)
    result = manager.result()
    assert result.panoramas_considered == 2
    assert result.selected_panorama_id == 'newest'


def test_most_recent_breaks_a_sharpness_tie() -> None:
    manager = _manager()
    manager.begin(0.0)
    manager.update(0.8)
    manager.offer_panorama(PanoramaOffer('first', 0.7, 1.0))
    manager.offer_panorama(PanoramaOffer('second', 0.7, 2.0))
    manager.update(3.3)
    assert manager.result().selected_panorama_id == 'second'


def test_observation_is_hard_capped_and_never_waits_forever() -> None:
    manager = ObservationManager(
        ObservationConfig(max_observation_time_sec=4.0)
    )
    manager.begin(0.0)
    manager.update(0.8)
    manager.offer_panorama(PanoramaOffer('only', 0.5, 1.0))
    assert manager.update(50.0) is ObservationState.TIMED_OUT
    result = manager.result()
    assert result.status == 'timed_out'
    # A timeout still yields the best evidence gathered so far.
    assert result.selected_panorama_id == 'only'


def test_scan_points_accumulate_into_the_result() -> None:
    manager = _manager()
    manager.begin(0.0)
    manager.update(0.8)
    manager.note_scan_points(20_000)
    manager.note_scan_points(28_213)
    manager.update(3.3)
    assert manager.result().scan_points_added == 48_213


def test_result_before_a_terminal_state_is_refused() -> None:
    manager = _manager()
    manager.begin(0.0)
    with pytest.raises(RuntimeError, match='terminal state'):
        manager.result()


def test_double_begin_is_refused() -> None:
    manager = _manager()
    manager.begin(0.0)
    with pytest.raises(RuntimeError, match='already started'):
        manager.begin(1.0)


def test_terminal_states_ignore_further_updates() -> None:
    manager = _manager()
    manager.begin(0.0)
    manager.update(0.8)
    manager.update(3.3)
    assert manager.state is ObservationState.COMPLETE
    assert manager.update(99.0) is ObservationState.COMPLETE


def test_config_rejects_a_cap_below_its_own_phases() -> None:
    with pytest.raises(ValueError, match='max_observation_time_sec'):
        ObservationConfig(
            settle_time_sec=2.0,
            scan_accumulation_time_sec=3.0,
            max_observation_time_sec=4.0,
        )
    with pytest.raises(ValueError, match='panoramas_to_consider'):
        ObservationConfig(panoramas_to_consider=0)


def test_no_panorama_still_produces_a_terminal_result() -> None:
    manager = _manager()
    manager.begin(0.0)
    manager.update(0.8)
    manager.update(3.3)
    result = manager.result()
    assert result.selected_panorama_id is None
    assert result.panoramas_considered == 0
    assert result.to_dict()['status'] == 'complete'
