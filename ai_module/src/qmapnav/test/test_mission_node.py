"""ROS-adapter tests for question and sequential waypoint protocols."""

from collections.abc import Iterator

import pytest
from qmapnav.language import parse_question
from qmapnav.navigation import Waypoint2D
from qmapnav.navigation import WaypointExecutorState


class _RecordingPublisher:
    """Record published messages without requiring a DDS subscriber."""

    def __init__(self) -> None:
        self.messages = []

    def publish(self, message: object) -> None:
        """Record one outgoing message."""
        self.messages.append(message)


@pytest.fixture
def node() -> Iterator[object]:
    """Create and cleanly destroy one ROS composition node."""
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    rclpy.init()
    qmapnav_node = QMapNavNode()
    try:
        yield qmapnav_node
    finally:
        qmapnav_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _question(text: str) -> object:
    from std_msgs.msg import String

    message = String()
    message.data = text
    return message


def _pose(x: float, y: float) -> object:
    from nav_msgs.msg import Odometry

    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    return message


def test_node_uses_only_required_official_protocol_topics(node: object) -> None:
    assert node._question_subscription.topic_name == '/challenge_question'
    assert node._pose_subscription.topic_name == '/state_estimation'
    assert node._waypoint_publisher.topic_name == '/way_point_with_heading'


def test_node_parses_first_question_and_ignores_repeated_publications() -> None:
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    parser_calls = []

    def recording_parser(question: str):
        parser_calls.append(question)
        return parse_question(question)

    rclpy.init()
    node = QMapNavNode(question_parser=recording_parser)
    question = 'How many computer monitors are on the table?'
    try:
        node._on_question(_question('   '))
        node._on_question(_question(question))
        accepted_task = node.task_specification
        for _ in range(5):
            node._on_question(_question(question))

        assert parser_calls == [question]
        assert node.question_latch.active_question == question
        assert node.question_latch.duplicate_count == 5
        assert node.task_specification is accepted_task
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_node_does_not_replace_active_question(node: object) -> None:
    node._on_question(_question('How many pillows are on the bed?'))
    accepted_task = node.task_specification

    node._on_question(_question('Find the flowers near the window.'))

    assert node.question_latch.active_question == (
        'How many pillows are on the bed?'
    )
    assert node.question_latch.conflict_count == 1
    assert node.task_specification is accepted_task


def test_node_publishes_route_one_goal_at_a_time_from_pose_updates(
    node: object,
) -> None:
    recorder = _RecordingPublisher()
    node._waypoint_publisher = recorder
    route = [
        Waypoint2D(1.0, 0.0, 0.0),
        Waypoint2D(2.0, 1.0, 1.57),
        Waypoint2D(3.0, 1.0, 0.0),
    ]

    node.start_route(route)
    node._on_pose(_pose(0.0, 0.0))
    node._on_pose(_pose(1.0, 0.0))
    node._on_pose(_pose(1.0, 0.0))
    node._on_pose(_pose(2.0, 1.0))
    node._on_pose(_pose(3.0, 1.0))

    published = [
        (message.x, message.y, message.theta) for message in recorder.messages
    ]
    assert published == [
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 1.57),
        (3.0, 1.0, 0.0),
    ]
    assert node.waypoint_executor.state is WaypointExecutorState.COMPLETE
