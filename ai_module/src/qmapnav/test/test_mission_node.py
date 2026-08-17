"""ROS-adapter tests for question and sequential waypoint protocols."""

from collections.abc import Iterator

import numpy as np
import pytest
from qmapnav.common.decision_trace import InMemoryTraceRecorder
from qmapnav.language import parse_question
from qmapnav.mapping import RegisteredScanAccumulator
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
    qmapnav_node = QMapNavNode(trace_recorder=InMemoryTraceRecorder())
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


def test_node_uses_only_permitted_official_topics(node: object) -> None:
    assert node._question_subscription.topic_name == '/challenge_question'
    assert node._pose_subscription.topic_name == '/state_estimation'
    assert node._scan_subscription.topic_name == '/registered_scan'
    assert node._image_subscription.topic_name == '/camera/image'
    assert node._waypoint_publisher.topic_name == '/way_point_with_heading'
    assert node._candidate_marker_publisher.topic_name == (
        '/qmapnav/debug/object_candidates'
    )
    assert node._object_map_marker_publisher.topic_name == (
        '/qmapnav/debug/object_map'
    )
    assert node._structural_map_marker_publisher.topic_name == (
        '/qmapnav/debug/structural_map'
    )
    assert node._relation_marker_publisher.topic_name == (
        '/qmapnav/debug/relations'
    )
    assert node._official_marker_publisher.topic_name == '/selected_object_marker'
    assert not node._final_object_answer_guard.committed


def test_node_commits_persistent_object_with_matching_waypoint(node) -> None:
    from qmapnav.common import ObjectInstance

    waypoint_recorder = _RecordingPublisher()
    node._waypoint_publisher = waypoint_recorder
    instance = ObjectInstance(
        7,
        {'chair': 0.9},
        {},
        np.array([1.0, 2.0, 0.5]),
        np.array([0.5, 1.5, 0.0]),
        np.array([1.5, 2.5, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        0.2,
        0.8,
        2,
        0.85,
    )

    answer = node._final_object_answer_guard.commit(instance, timestamp_ns=1)

    assert answer.marker.frame_id == 'map'
    assert node._final_object_answer_guard.committed
    assert len(waypoint_recorder.messages) == 1
    waypoint = waypoint_recorder.messages[0]
    assert (waypoint.x, waypoint.y) == (1.0, 2.0)


def test_node_loads_reasoning_policy_and_system_robot_footprint(
    node: object,
) -> None:
    assert node._reasoning_candidate_config.minimum_class_probability == 0.15
    assert node._reasoning_spatial_config.near_size_scale == 0.75
    assert node._reasoning_ambiguity_config.resolved_minimum_margin == 0.12
    assert node._reasoning_corridor_config.robot_width_m == 0.55
    assert node._reasoning_corridor_config.safety_clearance_m == 0.15


def test_colour_support_crop_keeps_own_mask_and_cluster_coordinates() -> None:
    from qmapnav.mission.node import _crop_colour_support
    from qmapnav.perception.contracts import Detection2D
    from qmapnav.perception.contracts import PanoramaBox

    boundary = np.array([
        [10.0, 5.0], [30.0, 5.0], [30.0, 20.0], [10.0, 20.0]
    ])
    polygon = ((12.0, 7.0), (28.0, 7.0), (28.0, 18.0), (12.0, 18.0))
    detection = Detection2D(
        'chair', 'chair', 'chair', 0.9,
        PanoramaBox(100, 40, ((10.0, 30.0),), 5.0, 20.0, boundary),
        (0,), ((0.0, 0.0, 10.0, 10.0),), (20.0, 12.0),
        np.array([1.0, 0.0, 0.0]),
        metadata={'mask_polygons_panorama_uv': (polygon,)},
    )

    mask, support = _crop_colour_support(
        (40, 100), detection, np.array([[15.0, 10.0], [80.0, 10.0]])
    )

    assert mask.shape == (15, 20)
    assert mask[5, 5]
    assert support.tolist() == [[5.0, 5.0]]


def test_node_resets_persistent_maps_without_touching_frozen_protocol(
    node: object,
) -> None:
    node._persistent_path_xy.append((1.0, 2.0))
    node.reset_persistent_maps()

    assert node.object_map.active_instances() == []
    assert node.object_map.next_instance_id == 0
    assert node.structural_map.walls() == []
    assert node.structural_map.anchors() == []
    assert list(node._persistent_path_xy) == []


@pytest.mark.parametrize('encoding', ['rgb8', 'bgr8'])
def test_camera_image_decoder_handles_row_padding_and_channel_order(
    encoding: str,
) -> None:
    from qmapnav.mission.node import _decode_image_rgb
    from sensor_msgs.msg import Image

    message = Image()
    message.height = 2
    message.width = 2
    message.encoding = encoding
    message.step = 8
    first_pixel = [1, 2, 3] if encoding == 'rgb8' else [3, 2, 1]
    message.data = bytes(
        first_pixel + [4, 5, 6, 99, 99]
        + [7, 8, 9, 10, 11, 12, 99, 99]
    )

    decoded = _decode_image_rgb(message)

    assert decoded.shape == (2, 2, 3)
    assert decoded[0, 0].tolist() == [1, 2, 3]
    if encoding == 'rgb8':
        assert decoded[1, 1].tolist() == [10, 11, 12]
    else:
        assert decoded[1, 1].tolist() == [12, 11, 10]


def test_node_parses_first_question_and_ignores_repeated_publications() -> None:
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    parser_calls = []

    def recording_parser(question: str):
        parser_calls.append(question)
        return parse_question(question)

    rclpy.init()
    node = QMapNavNode(
        question_parser=recording_parser,
        trace_recorder=InMemoryTraceRecorder(),
    )
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


def test_node_cancellation_publishes_current_pose_hold_once(node: object) -> None:
    recorder = _RecordingPublisher()
    node._waypoint_publisher = recorder
    node.start_route([Waypoint2D(10.0, 0.0)])
    node._on_pose(_pose(1.0, 2.0))

    node.cancel_route()
    node.cancel_route()

    published = [
        (message.x, message.y, message.theta) for message in recorder.messages
    ]
    assert published == [(10.0, 0.0, 0.0), (1.0, 2.0, 0.0)]
    assert node.waypoint_executor.state is WaypointExecutorState.CANCELLED


def test_node_records_question_route_and_completion_events() -> None:
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    trace = InMemoryTraceRecorder()
    rclpy.init()
    node = QMapNavNode(trace_recorder=trace)
    try:
        node._on_question(
            _question('First go near the plant, then stop near the window.')
        )
        node.start_route([Waypoint2D(1.0, 0.0)])
        node._on_pose(_pose(1.0, 0.0))

        event_names = [event.event for event in trace.events]
        assert 'question_latched' in event_names
        assert 'task_parsed' in event_names
        assert 'route_started' in event_names
        assert 'route_completed' in event_names
        completion = next(
            event for event in trace.events if event.event == 'route_completed'
        )
        assert completion.terminal_status == 'complete'
        assert completion.active_route_index == 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_node_adapts_registered_scan_into_persistent_map() -> None:
    from qmapnav.mission.node import QMapNavNode
    from sensor_msgs.msg import PointCloud2
    import numpy as np
    import rclpy

    points = np.array([[1.0, 0.0, 0.5], [1.01, 0.0, 0.5]])
    accumulator = RegisteredScanAccumulator()
    rclpy.init()
    node = QMapNavNode(
        scan_accumulator=accumulator,
        trace_recorder=InMemoryTraceRecorder(),
        point_cloud_decoder=lambda message: points,
    )
    try:
        message = PointCloud2()
        message.header.frame_id = 'map'
        node._on_registered_scan(message)
        node._on_registered_scan(message)

        assert accumulator.stats().accepted_scan_count == 2
        assert accumulator.stats().voxel_count == 1
        assert any(
            event.event == 'registered_scan_accumulated'
            for event in node._trace_recorder.events
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
