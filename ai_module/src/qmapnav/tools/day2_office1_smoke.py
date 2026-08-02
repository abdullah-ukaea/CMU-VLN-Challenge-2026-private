"""Run the final Day 2 three-waypoint integration route in Office 1."""

from json import dumps
from json import loads
import os
from pathlib import Path
import resource
from time import monotonic

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from qmapnav.evaluation import JsonlDecisionTraceRecorder
from qmapnav.mission.node import QMapNavNode
from qmapnav.navigation import Waypoint2D
from qmapnav.navigation import WaypointExecutorState
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String


QUESTION = (
    'First go near the plant, then pass between the two tables and stop near '
    'the window.'
)
ROUTE = (
    Waypoint2D(2.30, 0.26, 0.0),
    Waypoint2D(0.40, -1.58, 0.0),
    Waypoint2D(-3.25, -4.20, 0.0),
)
OUTPUT_DIRECTORY = Path('/tmp/qmapnav')


def _read_trace(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def main() -> int:
    """Execute the measured Day 1 route and write bounded Day 2 evidence."""
    mode = os.environ.get('QMAPNAV_DAY2_SMOKE_MODE', 'success')
    if mode not in {'recovery', 'success'}:
        raise ValueError('QMAPNAV_DAY2_SMOKE_MODE must be recovery or success')
    recovery_mode = mode == 'recovery'
    route = (Waypoint2D(100.0, 100.0, 0.0),) if recovery_mode else ROUTE
    expected_state = (
        WaypointExecutorState.FAILED
        if recovery_mode
        else WaypointExecutorState.COMPLETE
    )
    trace_path = OUTPUT_DIRECTORY / f'day2_office1_{mode}_trace.jsonl'
    summary_path = OUTPUT_DIRECTORY / f'day2_office1_{mode}_summary.json'
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    trace_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    rclpy.init()
    trace_recorder = JsonlDecisionTraceRecorder(
        trace_path,
        episode_id=f'day2-office1-{mode}',
    )
    parameter_overrides = [Parameter('episode_time_limit', value=120.0)]
    if recovery_mode:
        parameter_overrides = [
            Parameter('episode_time_limit', value=35.0),
            Parameter('no_progress_timeout', value=2.0),
            Parameter('watchdog_period', value=0.10),
        ]
    qmapnav_node = QMapNavNode(
        trace_recorder=trace_recorder,
        parameter_overrides=parameter_overrides,
    )
    driver = Node('day2_office1_driver')
    question_publisher = driver.create_publisher(
        String,
        '/challenge_question',
        5,
    )
    latest_pose: list[tuple[float, float]] = []
    published_waypoints: list[tuple[float, float, float]] = []
    question_publish_times: list[float] = []
    pose_subscription = driver.create_subscription(
        Odometry,
        '/state_estimation',
        lambda message: latest_pose.append(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
            )
        ),
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

    def publish_question() -> None:
        question_publish_times.append(monotonic())
        question_publisher.publish(String(data=QUESTION))

    question_timer = driver.create_timer(1.0, publish_question)
    executor = SingleThreadedExecutor()
    executor.add_node(qmapnav_node)
    executor.add_node(driver)
    route_started_at: float | None = None
    error: str | None = None
    try:
        readiness_deadline = monotonic() + 20.0
        while monotonic() < readiness_deadline:
            executor.spin_once(timeout_sec=0.05)
            if (
                latest_pose
                and qmapnav_node.scan_accumulator.stats().accepted_scan_count > 0
                and question_publisher.get_subscription_count() > 0
                and qmapnav_node._waypoint_publisher.get_subscription_count() >= 2
            ):
                break
        else:
            raise RuntimeError('Office 1 topics or base subscriber not ready')

        publish_question()
        parse_deadline = monotonic() + 5.0
        while (
            qmapnav_node.task_specification is None
            and monotonic() < parse_deadline
        ):
            executor.spin_once(timeout_sec=0.05)
        if qmapnav_node.task_specification is None:
            raise RuntimeError('scripted question was not parsed')

        route_started_at = monotonic()
        qmapnav_node.start_route(route)
        route_deadline = route_started_at + 90.0
        while (
            qmapnav_node.waypoint_executor.state
            not in {
                WaypointExecutorState.COMPLETE,
                WaypointExecutorState.FAILED,
                WaypointExecutorState.CANCELLED,
            }
            and monotonic() < route_deadline
        ):
            executor.spin_once(timeout_sec=0.05)
        if qmapnav_node.waypoint_executor.state not in {
            WaypointExecutorState.COMPLETE,
            WaypointExecutorState.FAILED,
            WaypointExecutorState.CANCELLED,
        }:
            qmapnav_node.cancel_route()
            raise RuntimeError('scripted Office 1 route exceeded 90 seconds')
    except Exception as caught:
        error = str(caught)
    finally:
        finished_at = monotonic()
        state = qmapnav_node.waypoint_executor.state.value
        task = qmapnav_node.task_specification
        scan_stats = qmapnav_node.scan_accumulator.stats()
        driver.destroy_timer(question_timer)
        driver.destroy_subscription(pose_subscription)
        driver.destroy_subscription(waypoint_subscription)
        executor.remove_node(driver)
        executor.remove_node(qmapnav_node)
        driver.destroy_node()
        qmapnav_node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

    trace_records = _read_trace(trace_path)
    summary = {
        'mode': mode,
        'result': state,
        'error': error,
        'question': QUESTION,
        'question_publish_count': len(question_publish_times),
        'duplicate_question_count': qmapnav_node.question_latch.duplicate_count,
        'task_type': task.task_type if task is not None else None,
        'parse_mode': task.parse_mode if task is not None else None,
        'route': [[item.x, item.y, item.heading] for item in route],
        'published_waypoints': [list(item) for item in published_waypoints],
        'route_runtime_seconds': (
            finished_at - route_started_at
            if route_started_at is not None
            else None
        ),
        'final_pose': list(latest_pose[-1]) if latest_pose else None,
        'accepted_scan_count': scan_stats.accepted_scan_count,
        'voxel_count': scan_stats.voxel_count,
        'scan_view_count': scan_stats.scan_view_count,
        'route_completion_trace_count': sum(
            record.get('event') == 'route_completed' for record in trace_records
        ),
        'route_failure_trace_count': sum(
            record.get('event') == 'route_failed' for record in trace_records
        ),
        'trace_record_count': len(trace_records),
        'trace_bytes': trace_path.stat().st_size if trace_path.exists() else 0,
        'process_max_rss_kib': resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }
    summary_path.write_text(
        dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(dumps(summary, indent=2, sort_keys=True))
    return 0 if state == expected_state.value else 1


if __name__ == '__main__':
    raise SystemExit(main())
