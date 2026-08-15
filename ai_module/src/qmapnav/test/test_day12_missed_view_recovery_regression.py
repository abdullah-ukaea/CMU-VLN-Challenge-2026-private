"""A missed first view must recover without locking an early low count."""

from day12_helpers import numerical_result
from qmapnav.counting import CountStabilityConfig
from qmapnav.counting import CountStabilityMachine


def test_late_second_instance_resets_then_reaches_stability() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=2,
        required_independent_viewpoints=2,
    ))
    first = machine.update(
        numerical_result((10,)),
        viewpoint_id='near_side',
        time_remaining_sec=500.0,
        episode_time_sec=1.0,
    )
    recovered = machine.update(
        numerical_result((10, 11)),
        viewpoint_id='far_side',
        time_remaining_sec=490.0,
        episode_time_sec=2.0,
    )
    verified = machine.update(
        numerical_result((10, 11)),
        viewpoint_id='doorway',
        time_remaining_sec=480.0,
        episode_time_sec=3.0,
    )
    assert not first.should_publish
    assert recovered.consecutive_stable_updates == 1
    assert recovered.current_instance_ids == (10, 11)
    assert verified.stable
    assert verified.result.count == 2
