"""Zero is stable evidence and not a sentinel for an absent answer."""

from day12_helpers import numerical_result
from qmapnav.counting import CountStabilityConfig
from qmapnav.counting import CountStabilityMachine


def test_zero_id_set_stabilises_across_independent_views() -> None:
    machine = CountStabilityMachine(CountStabilityConfig(
        required_consecutive_updates=2,
        required_independent_viewpoints=2,
    ))
    machine.update(
        numerical_result(()),
        viewpoint_id='support_left',
        time_remaining_sec=500.0,
        episode_time_sec=1.0,
    )
    stable = machine.update(
        numerical_result(()),
        viewpoint_id='support_right',
        time_remaining_sec=490.0,
        episode_time_sec=2.0,
    )
    assert stable.stable
    assert stable.should_publish
    assert stable.current_count == 0
    assert stable.current_instance_ids == ()
