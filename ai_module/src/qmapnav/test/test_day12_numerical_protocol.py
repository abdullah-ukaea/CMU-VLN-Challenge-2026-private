"""Official topic, Int32, zero, and single-publication protocol tests."""

from collections.abc import Iterator

from day12_helpers import numerical_result
import pytest
from qmapnav.evaluation import InMemoryTraceRecorder
from qmapnav.mission.numerical_output_adapter import NumericalOutputAdapter
from qmapnav.mission.numerical_output_adapter import OFFICIAL_NUMERICAL_TOPIC


class _Publisher:
    """Record official messages without requiring a DDS subscriber."""

    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        """Record one message."""
        self.messages.append(message)


@pytest.fixture
def node() -> Iterator[object]:
    """Create and destroy one numerical-capable ROS node."""
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    rclpy.init()
    instance = QMapNavNode(trace_recorder=InMemoryTraceRecorder())
    try:
        yield instance
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _question(text):
    from std_msgs.msg import String

    message = String()
    message.data = text
    return message


def test_official_numerical_contract_is_int32_and_exact_topic(node) -> None:
    assert OFFICIAL_NUMERICAL_TOPIC == '/numerical_response'
    assert node._numerical_publisher.topic_name == '/numerical_response'
    assert node._numerical_publisher.msg_type.__name__ == 'Int32'


def test_adapter_publishes_zero_once() -> None:
    publisher = _Publisher()
    adapter = NumericalOutputAdapter(publisher.publish)
    first = adapter.commit(numerical_result(()))
    second = adapter.commit(numerical_result((1, 2)))
    assert first == second
    assert len(publisher.messages) == 1
    assert publisher.messages[0].data == 0


def test_watchdog_force_publishes_best_available_before_expiry(node) -> None:
    publisher = _Publisher()
    node._numerical_output_adapter = NumericalOutputAdapter(publisher.publish)
    node._on_question(_question('How many cups are on the coffee table?'))
    node._force_numerical_commit('forced_watchdog_test')
    node._force_numerical_commit('repeat_must_not_publish')
    assert len(publisher.messages) == 1
    assert publisher.messages[0].data == 0
    assert node._numerical_episode.stability.state.status.value == 'published'
