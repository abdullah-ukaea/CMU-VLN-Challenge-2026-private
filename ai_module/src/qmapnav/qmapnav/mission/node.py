"""ROS 2 composition root for Q-MapNav."""

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import asdict
from math import atan2

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from qmapnav.common import TaskSpecification
from qmapnav.evaluation import DecisionTraceEvent
from qmapnav.evaluation import JsonlDecisionTraceRecorder
from qmapnav.evaluation import TraceRecorder
from qmapnav.language import parse_question
from qmapnav.mapping import AccumulationStatus
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mapping import ScanAccumulatorConfig
from qmapnav.mapping.point_cloud import decode_xyz_points
from qmapnav.mission.question_latch import QuestionLatch
from qmapnav.mission.question_latch import QuestionLatchStatus
from qmapnav.navigation import DEFAULT_ARRIVAL_RADIUS
from qmapnav.navigation import DEFAULT_DIRECT_REPUBLISH_LIMIT
from qmapnav.navigation import DEFAULT_NO_PROGRESS_TIMEOUT
from qmapnav.navigation import DEFAULT_PROGRESS_EPSILON
from qmapnav.navigation import DEFAULT_SAFE_OFFSET_LIMIT
from qmapnav.navigation import ExecutorEvent
from qmapnav.navigation import ExecutorEventType
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import Waypoint2D
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class QMapNavNode(Node):
    """Own the ROS lifecycle and composition of Q-MapNav subsystems."""

    def __init__(
        self,
        question_parser: Callable[[str], TaskSpecification] = parse_question,
        *,
        scan_accumulator: RegisteredScanAccumulator | None = None,
        trace_recorder: TraceRecorder | None = None,
        point_cloud_decoder: Callable[[object], object] = decode_xyz_points,
        parameter_overrides: list[Parameter] | None = None,
    ) -> None:
        super().__init__(
            'qmapnav',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._question_parser = question_parser
        self._point_cloud_decoder = point_cloud_decoder
        self._episode_start_time = self._now()
        self._episode_time_limit = float(
            self.get_parameter('episode_time_limit').value
        )
        if self._episode_time_limit <= 0.0:
            raise ValueError('episode_time_limit must be positive')
        self._trace_flush_timeout = float(
            self.get_parameter('trace_flush_timeout').value
        )
        self._recovery_offset_distance = float(
            self.get_parameter('recovery_offset_distance').value
        )
        self._recovery_clearance = float(
            self.get_parameter('recovery_clearance').value
        )
        self._latest_pose_xy: tuple[float, float] | None = None
        self._last_traced_scan_count = 0
        self._trace_closed = False

        self._question_latch = QuestionLatch()
        self._task_specification: TaskSpecification | None = None
        self._scan_accumulator = scan_accumulator or self._create_accumulator()
        self._trace_recorder = trace_recorder or self._create_trace_recorder()
        self._waypoint_executor = SequentialWaypointExecutor(
            arrival_radius=float(self.get_parameter('arrival_radius').value),
            progress_epsilon=float(
                self.get_parameter('progress_epsilon').value
            ),
            no_progress_timeout=float(
                self.get_parameter('no_progress_timeout').value
            ),
            direct_republish_limit=int(
                self.get_parameter('direct_republish_limit').value
            ),
            safe_offset_limit=int(
                self.get_parameter('safe_offset_limit').value
            ),
            safe_offset_selector=self._select_safe_offset,
            clock=self._now,
        )

        self._question_subscription = self.create_subscription(
            String,
            '/challenge_question',
            self._on_question,
            5,
        )
        self._pose_subscription = self.create_subscription(
            Odometry,
            '/state_estimation',
            self._on_pose,
            5,
        )
        self._scan_subscription = self.create_subscription(
            PointCloud2,
            '/registered_scan',
            self._on_registered_scan,
            5,
        )
        self._waypoint_publisher = self.create_publisher(
            Pose2D,
            '/way_point_with_heading',
            5,
        )
        self._watchdog_timer = self.create_timer(
            float(self.get_parameter('watchdog_period').value),
            self._on_watchdog,
        )
        self._trace(
            event='node_initialized',
            selection_reason='day_2_runtime_ready',
            details={
                'executor': {
                    'arrival_radius': self._waypoint_executor.arrival_radius,
                    'progress_epsilon': (
                        self._waypoint_executor.progress_epsilon
                    ),
                    'no_progress_timeout': (
                        self._waypoint_executor.no_progress_timeout
                    ),
                    'direct_republish_limit': (
                        self._waypoint_executor.direct_republish_limit
                    ),
                    'safe_offset_limit': (
                        self._waypoint_executor.safe_offset_limit
                    ),
                },
                'accumulator': asdict(self._scan_accumulator.config),
            },
        )
        self.get_logger().info('Q-MapNav node initialized')

    @property
    def question_latch(self) -> QuestionLatch:
        """Expose read-only latch state for composition and evaluation."""
        return self._question_latch

    @property
    def task_specification(self) -> TaskSpecification | None:
        """Return the task parsed from the one accepted question."""
        return self._task_specification

    @property
    def waypoint_executor(self) -> SequentialWaypointExecutor:
        """Expose read-only route execution state for composition."""
        return self._waypoint_executor

    @property
    def scan_accumulator(self) -> RegisteredScanAccumulator:
        """Expose the bounded persistent registered-scan map."""
        return self._scan_accumulator

    def start_route(self, route: Iterable[Waypoint2D]) -> None:
        """Start a route and publish exactly its first active waypoint."""
        if self._expire_episode_if_needed():
            raise RuntimeError('cannot start route after episode deadline')
        first_goal = self._waypoint_executor.start(route, now=self._now())
        self._publish_waypoint(first_goal)
        self._record_executor_events()

    def cancel_route(self) -> None:
        """Cancel the route and replace an active base goal with a pose hold."""
        hold_goal = self._waypoint_executor.cancel(now=self._now())
        if hold_goal is not None:
            self._publish_waypoint(hold_goal)
        self._record_executor_events()

    def destroy_node(self) -> None:
        """Perform bounded trace shutdown before destroying ROS entities."""
        if not self._trace_closed:
            self._trace(
                event='node_shutdown',
                selection_reason='clean_shutdown',
                terminal_status=self._terminal_status(),
            )
            try:
                self._trace_recorder.close(self._trace_flush_timeout)
            except Exception as error:  # tracing must remain observational
                self.get_logger().warning(
                    f'Failed to close decision trace: {error}'
                )
            self._trace_closed = True
        super().destroy_node()

    def _on_question(self, message: String) -> None:
        if self._expire_episode_if_needed():
            return
        decision = self._question_latch.offer(message.data)
        if decision.status is QuestionLatchStatus.ACCEPTED:
            try:
                self._task_specification = self._question_parser(
                    decision.question
                )
            except Exception as error:
                self.get_logger().error(f'Question parser failed: {error}')
                self._trace(
                    event='task_parse_failed',
                    selected_action='retain_latched_question',
                    selection_reason='parser_raised_exception',
                    details={
                        'question': decision.question,
                        'error': str(error),
                    },
                )
                return
            self._trace(
                event='question_latched',
                selected_action='parse_question',
                selection_reason='first_valid_non_empty_question',
                details={'question': decision.question},
            )
            self._trace(
                event='task_parsed',
                selected_action='accept_task_specification',
                selection_reason=(
                    f'{self._task_specification.parse_mode}_deterministic_parse'
                ),
                details={
                    'question': decision.question,
                    'task_specification': asdict(self._task_specification),
                },
            )
            self.get_logger().info('Accepted challenge question')
        elif decision.status is QuestionLatchStatus.DUPLICATE:
            self._trace(
                event='question_ignored',
                selected_action='ignore_question',
                selection_reason='identical_repeat',
                details={'duplicate_count': self._question_latch.duplicate_count},
            )
            self.get_logger().debug('Ignored repeated challenge question')
        elif decision.status is QuestionLatchStatus.CONFLICT:
            self._trace(
                event='question_ignored',
                selected_action='ignore_question',
                selection_reason='different_question_during_active_episode',
                details={'conflict_count': self._question_latch.conflict_count},
            )
            self.get_logger().warning(
                'Ignored different question during active episode'
            )

    def _on_pose(self, message: Odometry) -> None:
        if self._expire_episode_if_needed():
            return
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        heading = atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
        )
        self._latest_pose_xy = (position.x, position.y)
        next_goal = self._waypoint_executor.update_pose(
            position.x,
            position.y,
            heading,
            now=self._now(),
        )
        if next_goal is not None:
            self._publish_waypoint(next_goal)
        self._record_executor_events()

    def _on_watchdog(self) -> None:
        if self._expire_episode_if_needed():
            return
        goal = self._waypoint_executor.tick(now=self._now())
        if goal is not None:
            self._publish_waypoint(goal)
        self._record_executor_events()

    def _on_registered_scan(self, message: PointCloud2) -> None:
        if self._expire_episode_if_needed():
            return
        try:
            points = self._point_cloud_decoder(message)
            result = self._scan_accumulator.add_scan(
                points,
                frame_id=message.header.frame_id,
                timestamp=self._now(),
                sensor_origin_xy=self._latest_pose_xy,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f'Rejected malformed registered scan: {error}')
            self._trace(
                event='registered_scan_rejected',
                selected_action='ignore_scan',
                selection_reason='malformed_scan',
                details={'error': str(error)},
            )
            return

        if result.status in {
            AccumulationStatus.REJECTED_FRAME,
            AccumulationStatus.STALE,
        }:
            self.get_logger().warning(
                f'Rejected registered scan: {result.status.value}'
            )
            self._trace(
                event='registered_scan_rejected',
                selected_action='ignore_scan',
                selection_reason=result.status.value,
                details={
                    **asdict(result),
                    'status': result.status.value,
                },
            )
            return

        stats = self._scan_accumulator.stats()
        if (
            stats.accepted_scan_count == 1
            or stats.accepted_scan_count - self._last_traced_scan_count >= 25
        ):
            self._last_traced_scan_count = stats.accepted_scan_count
            self._trace(
                event='registered_scan_accumulated',
                selected_action='retain_bounded_scan_map',
                selection_reason=result.status.value,
                details={
                    'result': {
                        **asdict(result),
                        'status': result.status.value,
                    },
                    'stats': asdict(stats),
                },
            )

    def _select_safe_offset(
        self,
        current_x: float,
        current_y: float,
        goal: Waypoint2D,
    ) -> Waypoint2D | None:
        candidate = self._scan_accumulator.select_safe_offset(
            current_x,
            current_y,
            goal.x,
            goal.y,
            offset_distance=self._recovery_offset_distance,
            clearance=self._recovery_clearance,
        )
        if candidate is None:
            return None
        return Waypoint2D(*candidate)

    def _publish_waypoint(self, waypoint: Waypoint2D) -> None:
        message = Pose2D()
        message.x = waypoint.x
        message.y = waypoint.y
        message.theta = waypoint.heading
        self._waypoint_publisher.publish(message)

    def _record_executor_events(self) -> None:
        for event in self._waypoint_executor.drain_events():
            self._trace_executor_event(event)

    def _trace_executor_event(self, event: ExecutorEvent) -> None:
        waypoint_details = None
        if event.waypoint is not None:
            waypoint_details = asdict(event.waypoint)
        terminal_status = None
        if event.event_type is ExecutorEventType.ROUTE_COMPLETED:
            terminal_status = 'complete'
        elif event.event_type is ExecutorEventType.ROUTE_FAILED:
            terminal_status = 'failed'
        elif event.event_type is ExecutorEventType.ROUTE_CANCELLED:
            terminal_status = 'cancelled'

        action = 'continue_current_goal'
        if event.waypoint is not None:
            action = 'publish_waypoint'
        if terminal_status is not None:
            action = f'mark_route_{terminal_status}'
            if (
                event.event_type is ExecutorEventType.ROUTE_CANCELLED
                and event.waypoint is not None
            ):
                action = 'publish_pose_hold_and_cancel'
            elif (
                event.event_type is ExecutorEventType.ROUTE_FAILED
                and event.waypoint is not None
            ):
                action = 'publish_pose_hold_and_fail'
        self._trace(
            event=event.event_type.value,
            candidate_actions=(action,),
            selected_action=action,
            selection_reason=event.reason,
            active_route_index=event.route_index,
            terminal_status=terminal_status,
            details={
                'distance': event.distance,
                'waypoint': waypoint_details,
                'executor_timestamp': event.timestamp,
            },
        )

    def _trace(
        self,
        *,
        event: str,
        candidate_actions: tuple[str, ...] = (),
        selected_action: str | None = None,
        selection_reason: str = '',
        active_route_index: int | None = None,
        terminal_status: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        task = self._task_specification
        missing_entities = (
            tuple(entity.entity_id for entity in task.entities)
            if task is not None
            else ()
        )
        raw_question = self._question_latch.active_question
        try:
            self._trace_recorder.record(
                DecisionTraceEvent(
                    event=event,
                    episode_elapsed_seconds=self._elapsed(),
                    mission_state=self._waypoint_executor.state.value,
                    raw_question=raw_question,
                    normalized_question=(
                        ' '.join(raw_question.casefold().split())
                        if raw_question is not None
                        else None
                    ),
                    parser_mode=task.parse_mode if task is not None else None,
                    parse_confidence=(
                        task.parse_confidence if task is not None else None
                    ),
                    known_object_count=0,
                    known_structure_count=0,
                    missing_entities=missing_entities,
                    candidate_actions=candidate_actions,
                    selected_action=selected_action,
                    selection_reason=selection_reason,
                    active_route_index=(
                        self._waypoint_executor.active_index
                        if active_route_index is None
                        else active_route_index
                    ),
                    direct_republish_count=(
                        self._waypoint_executor.direct_republish_count
                    ),
                    recovery_count=self._waypoint_executor.recovery_count,
                    ignored_question_count=(
                        self._question_latch.duplicate_count
                        + self._question_latch.conflict_count
                    ),
                    time_remaining_seconds=max(
                        0.0,
                        self._episode_time_limit - self._elapsed(),
                    ),
                    terminal_status=terminal_status,
                    details=details or {},
                )
            )
        except Exception as error:  # trace failures must never alter control
            self.get_logger().warning(f'Decision trace event dropped: {error}')

    def _create_accumulator(self) -> RegisteredScanAccumulator:
        return RegisteredScanAccumulator(
            ScanAccumulatorConfig(
                frame_id=str(self.get_parameter('scan_frame').value),
                voxel_size=float(self.get_parameter('scan_voxel_size').value),
                max_range=float(self.get_parameter('scan_max_range').value),
                max_age_seconds=float(
                    self.get_parameter('scan_max_age_seconds').value
                ),
                max_voxels=int(self.get_parameter('scan_max_voxels').value),
                max_scan_views=int(
                    self.get_parameter('scan_max_views').value
                ),
            )
        )

    def _expire_episode_if_needed(self) -> bool:
        if self._elapsed() < self._episode_time_limit:
            return False
        hold_goal = self._waypoint_executor.expire(now=self._now())
        if hold_goal is not None:
            self._publish_waypoint(hold_goal)
        self._record_executor_events()
        return True

    def _create_trace_recorder(self) -> JsonlDecisionTraceRecorder:
        return JsonlDecisionTraceRecorder(
            str(self.get_parameter('trace_path').value),
            max_queue_size=int(
                self.get_parameter('trace_max_queue_size').value
            ),
            max_file_bytes=int(
                self.get_parameter('trace_max_file_bytes').value
            ),
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('arrival_radius', DEFAULT_ARRIVAL_RADIUS)
        self.declare_parameter('progress_epsilon', DEFAULT_PROGRESS_EPSILON)
        self.declare_parameter(
            'no_progress_timeout', DEFAULT_NO_PROGRESS_TIMEOUT
        )
        self.declare_parameter(
            'direct_republish_limit', DEFAULT_DIRECT_REPUBLISH_LIMIT
        )
        self.declare_parameter('safe_offset_limit', DEFAULT_SAFE_OFFSET_LIMIT)
        self.declare_parameter('watchdog_period', 0.25)
        self.declare_parameter('recovery_offset_distance', 0.75)
        self.declare_parameter('recovery_clearance', 0.35)
        self.declare_parameter('episode_time_limit', 600.0)
        self.declare_parameter('scan_frame', 'map')
        self.declare_parameter('scan_voxel_size', 0.20)
        self.declare_parameter('scan_max_range', 30.0)
        self.declare_parameter('scan_max_age_seconds', 120.0)
        self.declare_parameter('scan_max_voxels', 200_000)
        self.declare_parameter('scan_max_views', 16)
        self.declare_parameter(
            'trace_path', '/tmp/qmapnav/decision_trace.jsonl'
        )
        self.declare_parameter('trace_max_queue_size', 512)
        self.declare_parameter('trace_max_file_bytes', 4 * 1024 * 1024)
        self.declare_parameter('trace_flush_timeout', 1.0)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def _elapsed(self) -> float:
        return max(0.0, self._now() - self._episode_start_time)

    def _terminal_status(self) -> str | None:
        state = self._waypoint_executor.state.value
        if state in {'complete', 'failed', 'cancelled'}:
            return state
        return None


def main(args: list[str] | None = None) -> None:
    """Run the Q-MapNav ROS node until shutdown."""
    rclpy.init(args=args)
    node = QMapNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
