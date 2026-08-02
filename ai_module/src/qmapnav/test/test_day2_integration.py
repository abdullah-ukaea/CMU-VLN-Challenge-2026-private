"""ROS-facing Day 2 integration tests over the official topic interfaces."""

from collections.abc import Callable
from time import monotonic

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import pytest
from qmapnav.evaluation import InMemoryTraceRecorder
from qmapnav.language import parse_question
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mission.node import QMapNavNode
from qmapnav.navigation import Waypoint2D
from qmapnav.navigation import WaypointExecutorState
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_msgs.msg import String


QUESTION = (
    'First go near the plant, then pass between the two tables and stop near '
    'the window.'
)
ROUTE = (
    Waypoint2D(1.0, 0.0, 0.0),
    Waypoint2D(2.0, 1.0, 1.57),
    Waypoint2D(3.0, 1.0, 0.0),
)


def _spin_until(
    executor: SingleThreadedExecutor,
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
) -> None:
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert predicate(), 'timed out waiting for ROS integration condition'


def _pose(x: float, y: float) -> Odometry:
    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.w = 1.0
    return message


def _publish_pose(
    publisher: object,
    executor: SingleThreadedExecutor,
    x: float,
    y: float,
) -> None:
    publisher.publish(_pose(x, y))
    executor.spin_once(timeout_sec=0.05)


def _cleanup(
    executor: SingleThreadedExecutor,
    qmapnav_node: QMapNavNode,
    driver: Node,
) -> None:
    executor.remove_node(driver)
    executor.remove_node(qmapnav_node)
    driver.destroy_node()
    qmapnav_node.destroy_node()
    executor.shutdown()
    if rclpy.ok():
        rclpy.shutdown()


def test_repeated_one_hz_questions_create_one_episode_and_parse_once() -> None:
    parser_calls = []
    publish_times = []
    trace = InMemoryTraceRecorder()

    def recording_parser(question: str):
        parser_calls.append(question)
        return parse_question(question)

    rclpy.init()
    qmapnav_node = QMapNavNode(
        question_parser=recording_parser,
        trace_recorder=trace,
    )
    driver = Node('day2_question_driver')
    publisher = driver.create_publisher(String, '/challenge_question', 5)

    def publish_question() -> None:
        message = String(data=QUESTION)
        publish_times.append(monotonic())
        publisher.publish(message)

    timer = driver.create_timer(1.0, publish_question)
    executor = SingleThreadedExecutor()
    executor.add_node(qmapnav_node)
    executor.add_node(driver)
    try:
        _spin_until(
            executor,
            lambda: qmapnav_node.question_latch.duplicate_count >= 1,
            timeout=2.5,
        )

        assert parser_calls == [QUESTION]
        assert qmapnav_node.question_latch.active_question == QUESTION
        assert len(publish_times) == 2
        assert 0.8 <= publish_times[1] - publish_times[0] <= 1.2
        assert sum(event.event == 'question_latched' for event in trace.events) == 1
    finally:
        driver.destroy_timer(timer)
        _cleanup(executor, qmapnav_node, driver)


def test_question_to_three_waypoint_route_and_logged_completion() -> None:
    trace = InMemoryTraceRecorder()
    accumulator = RegisteredScanAccumulator()
    published_waypoints = []
    rclpy.init()
    qmapnav_node = QMapNavNode(
        scan_accumulator=accumulator,
        trace_recorder=trace,
    )
    driver = Node('day2_route_driver')
    question_publisher = driver.create_publisher(
        String,
        '/challenge_question',
        5,
    )
    pose_publisher = driver.create_publisher(
        Odometry,
        '/state_estimation',
        5,
    )
    scan_publisher = driver.create_publisher(
        PointCloud2,
        '/registered_scan',
        5,
    )
    waypoint_subscription = driver.create_subscription(
        Pose2D,
        '/way_point_with_heading',
        lambda message: published_waypoints.append(
            (message.x, message.y, message.theta)
        ),
        5,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(qmapnav_node)
    executor.add_node(driver)
    try:
        _spin_until(
            executor,
            lambda: question_publisher.get_subscription_count() == 1
            and pose_publisher.get_subscription_count() == 1
            and scan_publisher.get_subscription_count() == 1
            and qmapnav_node._waypoint_publisher.get_subscription_count() == 1,
        )
        question_publisher.publish(String(data=QUESTION))
        question_publisher.publish(String(data=QUESTION))
        _spin_until(
            executor,
            lambda: qmapnav_node.question_latch.duplicate_count == 1,
        )
        assert qmapnav_node.task_specification is not None
        assert qmapnav_node.task_specification.parse_mode == 'full'

        scan = point_cloud2.create_cloud_xyz32(
            Header(frame_id='map'),
            [(1.0, 0.0, 0.5), (1.01, 0.0, 0.5)],
        )
        for _ in range(5):
            scan_publisher.publish(scan)
            executor.spin_once(timeout_sec=0.05)

        qmapnav_node.start_route(ROUTE)
        _spin_until(executor, lambda: len(published_waypoints) == 1)
        _publish_pose(pose_publisher, executor, 0.0, 0.0)
        assert published_waypoints == [(1.0, 0.0, 0.0)]

        _publish_pose(pose_publisher, executor, 1.0, 0.0)
        _spin_until(executor, lambda: len(published_waypoints) == 2)
        _publish_pose(pose_publisher, executor, 2.0, 1.0)
        _spin_until(executor, lambda: len(published_waypoints) == 3)
        _publish_pose(pose_publisher, executor, 3.0, 1.0)
        _spin_until(
            executor,
            lambda: qmapnav_node.waypoint_executor.state
            is WaypointExecutorState.COMPLETE,
        )

        assert published_waypoints == [
            (1.0, 0.0, 0.0),
            (2.0, 1.0, 1.57),
            (3.0, 1.0, 0.0),
        ]
        assert accumulator.stats().accepted_scan_count == 5
        assert accumulator.stats().voxel_count == 1
        completions = [
            event for event in trace.events if event.event == 'route_completed'
        ]
        assert len(completions) == 1
        assert completions[0].terminal_status == 'complete'
        assert completions[0].active_route_index == 2
    finally:
        driver.destroy_subscription(waypoint_subscription)
        _cleanup(executor, qmapnav_node, driver)


def test_no_progress_recovery_is_bounded_and_does_not_deadlock() -> None:
    trace = InMemoryTraceRecorder()
    published_waypoints = []
    overrides = [
        Parameter('no_progress_timeout', value=0.10),
        Parameter('watchdog_period', value=0.02),
        Parameter('episode_time_limit', value=3.0),
    ]
    rclpy.init()
    qmapnav_node = QMapNavNode(
        trace_recorder=trace,
        parameter_overrides=overrides,
    )
    driver = Node('day2_recovery_driver')
    pose_publisher = driver.create_publisher(
        Odometry,
        '/state_estimation',
        5,
    )
    scan_publisher = driver.create_publisher(
        PointCloud2,
        '/registered_scan',
        5,
    )
    waypoint_subscription = driver.create_subscription(
        Pose2D,
        '/way_point_with_heading',
        lambda message: published_waypoints.append((message.x, message.y)),
        5,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(qmapnav_node)
    executor.add_node(driver)
    try:
        _spin_until(
            executor,
            lambda: pose_publisher.get_subscription_count() == 1
            and scan_publisher.get_subscription_count() == 1
            and qmapnav_node._waypoint_publisher.get_subscription_count() == 1,
        )
        _publish_pose(pose_publisher, executor, 0.0, 0.0)
        scan_publisher.publish(
            point_cloud2.create_cloud_xyz32(
                Header(frame_id='map'),
                [(0.0, 2.0, 1.0)],
            )
        )
        _spin_until(
            executor,
            lambda: qmapnav_node.scan_accumulator.stats().accepted_scan_count
            == 1,
        )

        qmapnav_node.start_route([Waypoint2D(5.0, 0.0)])
        _spin_until(executor, lambda: len(published_waypoints) >= 3)
        _spin_until(
            executor,
            lambda: qmapnav_node.waypoint_executor.state
            is WaypointExecutorState.FAILED,
        )

        assert published_waypoints[0] == (5.0, 0.0)
        assert published_waypoints[1] == (5.0, 0.0)
        assert published_waypoints[2] == pytest.approx((0.0, 0.75))
        event_names = [event.event for event in trace.events]
        assert event_names.count('goal_republished') == 1
        assert event_names.count('recovery_started') == 1
        assert event_names.count('route_failed') == 1
    finally:
        driver.destroy_subscription(waypoint_subscription)
        _cleanup(executor, qmapnav_node, driver)


def test_episode_deadline_fails_route_and_publishes_pose_hold() -> None:
    trace = InMemoryTraceRecorder()
    published_waypoints = []
    overrides = [
        Parameter('watchdog_period', value=0.02),
        Parameter('episode_time_limit', value=0.15),
    ]
    rclpy.init()
    qmapnav_node = QMapNavNode(
        trace_recorder=trace,
        parameter_overrides=overrides,
    )
    driver = Node('day2_deadline_driver')
    pose_publisher = driver.create_publisher(
        Odometry,
        '/state_estimation',
        5,
    )
    waypoint_subscription = driver.create_subscription(
        Pose2D,
        '/way_point_with_heading',
        lambda message: published_waypoints.append((message.x, message.y)),
        5,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(qmapnav_node)
    executor.add_node(driver)
    try:
        _spin_until(
            executor,
            lambda: pose_publisher.get_subscription_count() == 1
            and qmapnav_node._waypoint_publisher.get_subscription_count() == 1,
        )
        _publish_pose(pose_publisher, executor, 0.25, -0.50)
        qmapnav_node.start_route([Waypoint2D(5.0, 0.0)])
        _spin_until(
            executor,
            lambda: qmapnav_node.waypoint_executor.state
            is WaypointExecutorState.FAILED,
        )
        _spin_until(executor, lambda: len(published_waypoints) == 2)

        assert published_waypoints == [(5.0, 0.0), (0.25, -0.50)]
        failure = next(
            event for event in trace.events if event.event == 'route_failed'
        )
        assert failure.selection_reason == 'episode_deadline_exceeded'
        assert failure.terminal_status == 'failed'
        assert failure.time_remaining_seconds == 0.0
    finally:
        driver.destroy_subscription(waypoint_subscription)
        _cleanup(executor, qmapnav_node, driver)
