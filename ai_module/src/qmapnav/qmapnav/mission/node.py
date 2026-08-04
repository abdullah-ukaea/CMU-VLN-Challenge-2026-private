"""ROS 2 composition root for Q-MapNav."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import replace
from math import atan2
from pathlib import Path
from threading import RLock

import cv2
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import numpy as np
from qmapnav.common import TaskSpecification
from qmapnav.evaluation import DecisionTraceEvent
from qmapnav.evaluation import JsonlDecisionTraceRecorder
from qmapnav.evaluation import TraceRecorder
from qmapnav.language import parse_question
from qmapnav.mapping import AccumulationStatus
from qmapnav.mapping import AssociationConfig
from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import BoundedProjectionWorker
from qmapnav.mapping import Day5ProjectionPipeline
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping import ProjectionConfig
from qmapnav.mapping import ProjectionFrame
from qmapnav.mapping import ProjectionQualityConfig
from qmapnav.mapping import ProjectionSynchronizer
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mapping import ScanAccumulatorConfig
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping import TimedPose
from qmapnav.mapping import TimedRegisteredScan
from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.lifting_pipeline import Day6LiftingPipeline
from qmapnav.mapping.lifting_pipeline import LiftingFrame
from qmapnav.mapping.lifting_visualisation import draw_candidate_orthographic
from qmapnav.mapping.lifting_visualisation import draw_depth_histogram
from qmapnav.mapping.lifting_visualisation import draw_lifting_stage_overlay
from qmapnav.mapping.map_visualisation import draw_persistent_map_top_down
from qmapnav.mapping.object_association import (
    AssociationConfig as ObjectAssociationConfig,
)
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.object_map import ObjectMapConfig
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_cloud import decode_scan_arrays
from qmapnav.mapping.point_cloud import ScanArrays
from qmapnav.mapping.point_cloud import stamp_to_nanoseconds
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.mapping.projection_regression import save_projection_regression_case
from qmapnav.mapping.projection_visualisation import draw_detection_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_top_down_projection
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.structural_map import StructuralMapConfig
from qmapnav.mapping.transforms import invert_transform
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.mapping.wall_extraction import WallExtractionConfig
from qmapnav.mission.marker_adapter import candidate_marker_array
from qmapnav.mission.marker_adapter import CANDIDATE_MARKER_TOPIC
from qmapnav.mission.marker_adapter import FinalMarkerGuard
from qmapnav.mission.marker_adapter import object_map_marker_array
from qmapnav.mission.marker_adapter import OBJECT_MAP_MARKER_TOPIC
from qmapnav.mission.marker_adapter import OFFICIAL_MARKER_TOPIC
from qmapnav.mission.marker_adapter import structural_map_marker_array
from qmapnav.mission.marker_adapter import STRUCTURAL_MAP_MARKER_TOPIC
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
from qmapnav.perception.baseline import make_day4_baseline_worker
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PerceptionRequest
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.vocabulary import detector_classes_from_task_specification
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class QMapNavNode(Node):
    """Own the ROS lifecycle and composition of Q-MapNav subsystems."""

    def __init__(
        self,
        question_parser: Callable[[str], TaskSpecification] = parse_question,
        *,
        scan_accumulator: RegisteredScanAccumulator | None = None,
        projection_pipeline: Day5ProjectionPipeline | None = None,
        lifting_pipeline: Day6LiftingPipeline | None = None,
        object_map: ObjectMap | None = None,
        structural_map: StructuralMap | None = None,
        perception_worker: object | None = None,
        trace_recorder: TraceRecorder | None = None,
        point_cloud_decoder: Callable[[object], object] = decode_scan_arrays,
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
        self._latest_timed_pose: TimedPose | None = None
        self._last_traced_scan_count = 0
        self._trace_closed = False
        self._projection_lock = RLock()
        self._latest_projection_frame: ProjectionFrame | None = None
        self._latest_lifting_frame: LiftingFrame | None = None
        self._perception_worker = perception_worker
        self._processed_image_ids: deque[str] = deque(maxlen=128)
        self._processed_image_id_set: set[str] = set()
        self._saved_projection_count = 0
        self._structural_frame_count = 0
        self._persistent_path_xy: deque[tuple[float, float]] = deque(
            maxlen=2048
        )

        self._question_latch = QuestionLatch()
        self._task_specification: TaskSpecification | None = None
        self._scan_accumulator = scan_accumulator or self._create_accumulator()
        self._projection_pipeline = (
            projection_pipeline or self._create_projection_pipeline()
        )
        self._lifting_pipeline = (
            lifting_pipeline or self._create_lifting_pipeline()
        )
        self._object_map = object_map or self._create_object_map()
        self._structural_map = structural_map or self._create_structural_map()
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
        self._image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self._on_image,
            5,
        )
        self._waypoint_publisher = self.create_publisher(
            Pose2D,
            '/way_point_with_heading',
            5,
        )
        self._candidate_marker_publisher = self.create_publisher(
            MarkerArray,
            CANDIDATE_MARKER_TOPIC,
            5,
        )
        self._object_map_marker_publisher = self.create_publisher(
            MarkerArray,
            OBJECT_MAP_MARKER_TOPIC,
            5,
        )
        self._structural_map_marker_publisher = self.create_publisher(
            MarkerArray,
            STRUCTURAL_MAP_MARKER_TOPIC,
            5,
        )
        self._official_marker_publisher = self.create_publisher(
            Marker,
            OFFICIAL_MARKER_TOPIC,
            5,
        )
        self._final_marker_guard = FinalMarkerGuard(
            self._official_marker_publisher.publish
        )
        self._watchdog_timer = self.create_timer(
            float(self.get_parameter('watchdog_period').value),
            self._on_watchdog,
        )
        self._trace(
            event='node_initialized',
            selection_reason='day_7_persistent_mapping_runtime_ready',
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
                'dense_accumulator': asdict(
                    self._projection_pipeline.dense_accumulator.config
                ),
                'object_map': asdict(self._object_map.config),
                'structural_map': asdict(self._structural_map.config),
            },
        )
        self._projection_worker = BoundedProjectionWorker(
            self._process_panorama,
            self._on_projection_result,
            max_queue_size=int(
                self.get_parameter('projection_worker_queue_size').value
            ),
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

    @property
    def latest_projection_frame(self) -> ProjectionFrame | None:
        """Return the latest immutable Day 5 projection result."""
        with self._projection_lock:
            return self._latest_projection_frame

    @property
    def latest_lifting_frame(self) -> LiftingFrame | None:
        """Return the latest immutable single-observation lifting result."""
        with self._projection_lock:
            return self._latest_lifting_frame

    @property
    def object_map(self) -> ObjectMap:
        """Expose the bounded episode-local persistent object map."""
        return self._object_map

    @property
    def structural_map(self) -> StructuralMap:
        """Expose the bounded episode-local architectural map."""
        return self._structural_map

    def reset_persistent_maps(self) -> None:
        """Reset Day 7 object and structural identities at an episode boundary."""
        self._object_map.reset_episode()
        self._structural_map.reset_episode()
        self._structural_frame_count = 0
        self._persistent_path_xy.clear()

    def publish_final_object_candidate(
        self,
        candidate: ObjectCandidate3D,
    ) -> None:
        """Explicitly commit one externally selected candidate as official."""
        self._final_marker_guard.commit(candidate)
        self._trace(
            event='official_object_marker_committed',
            selected_action='publish_selected_object_marker',
            selection_reason='explicit_external_candidate_commit',
            details={'candidate_id': candidate.candidate_id},
        )

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
        projection_worker = getattr(self, '_projection_worker', None)
        if projection_worker is not None:
            if not projection_worker.close(
                float(self.get_parameter('projection_shutdown_timeout').value)
            ):
                self.get_logger().warning('Projection worker did not stop in time')
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
        if (
            not self._persistent_path_xy
            or np.linalg.norm(
                np.asarray(self._persistent_path_xy[-1])
                - np.asarray(self._latest_pose_xy)
            ) >= 0.02
        ):
            self._persistent_path_xy.append(self._latest_pose_xy)
        try:
            if message.header.frame_id and message.child_frame_id:
                self._latest_timed_pose = TimedPose(
                    timestamp_ns=stamp_to_nanoseconds(message.header.stamp),
                    parent_frame_id=message.header.frame_id,
                    child_frame_id=message.child_frame_id,
                    position_xyz=np.array(
                        [position.x, position.y, position.z],
                        dtype=np.float64,
                    ),
                    orientation_xyzw=np.array(
                        [
                            orientation.x,
                            orientation.y,
                            orientation.z,
                            orientation.w,
                        ],
                        dtype=np.float64,
                    ),
                    receipt_timestamp_ns=self.get_clock().now().nanoseconds,
                )
                self._projection_pipeline.add_pose(self._latest_timed_pose)
        except ValueError as error:
            self.get_logger().warning(f'Rejected projection pose: {error}')
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
            decoded = self._point_cloud_decoder(message)
            if isinstance(decoded, ScanArrays):
                points = decoded.xyz
                intensity = decoded.intensity
            else:
                points = np.asarray(decoded, dtype=np.float64)
                intensity = None
            source_timestamp_ns = stamp_to_nanoseconds(message.header.stamp)
            result = self._scan_accumulator.add_scan(
                points,
                frame_id=message.header.frame_id,
                timestamp=source_timestamp_ns / 1_000_000_000.0,
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
            AccumulationStatus.ACCEPTED,
            AccumulationStatus.EMPTY,
        }:
            try:
                timed_scan = TimedRegisteredScan(
                    timestamp_ns=source_timestamp_ns,
                    frame_id=message.header.frame_id,
                    points_xyz=points,
                    intensity=intensity,
                    receipt_timestamp_ns=self.get_clock().now().nanoseconds,
                )
                sensor_origin = (
                    self._latest_timed_pose.position_xyz
                    if self._latest_timed_pose is not None
                    else None
                )
                self._projection_pipeline.add_scan(
                    timed_scan,
                    sensor_origin_xyz=sensor_origin,
                )
            except ValueError as error:
                self.get_logger().warning(
                    f'Rejected scan from Day 5 projection path: {error}'
                )

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

    def _on_image(self, message: Image) -> None:
        if self._expire_episode_if_needed():
            return
        try:
            timestamp_ns = stamp_to_nanoseconds(message.header.stamp)
            if not message.header.frame_id:
                raise ValueError('camera image frame_id is empty')
            image_rgb = _decode_image_rgb(message)
            image_id = f'{timestamp_ns}'
            if image_id in self._processed_image_id_set:
                return
            if len(self._processed_image_ids) == self._processed_image_ids.maxlen:
                removed = self._processed_image_ids.popleft()
                self._processed_image_id_set.discard(removed)
            self._processed_image_ids.append(image_id)
            self._processed_image_id_set.add(image_id)
            panorama = TimedPanorama(
                image_id=image_id,
                timestamp_ns=timestamp_ns,
                frame_id=message.header.frame_id,
                image_rgb=image_rgb,
                receipt_timestamp_ns=self.get_clock().now().nanoseconds,
            )
            self._projection_worker.submit(panorama)
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f'Rejected camera image: {error}')
            self._trace(
                event='camera_image_rejected',
                selected_action='ignore_image',
                selection_reason='malformed_image',
                details={'error': str(error)},
            )

    def _process_panorama(
        self,
        panorama: TimedPanorama,
    ) -> ProjectionFrame | AssociationFailure:
        detections = ()
        if self._perception_worker is not None and self._task_specification is not None:
            request = PerceptionRequest(
                image_id=panorama.image_id,
                timestamp_ns=panorama.timestamp_ns,
                panorama_rgb=panorama.image_rgb,
                detector_classes=detector_classes_from_task_specification(
                    self._task_specification
                ),
                task_type=self._task_specification.task_type,
            )
            detections = self._perception_worker.process(request).detections
        return self._projection_pipeline.process(panorama, detections)

    def _on_projection_result(
        self,
        result: ProjectionFrame | AssociationFailure,
    ) -> None:
        if isinstance(result, AssociationFailure):
            self._trace(
                event='projection_skipped',
                selected_action='skip_keyframe',
                selection_reason=result.reason,
                details={
                    'image_id': result.panorama.image_id,
                    'image_timestamp_ns': result.panorama.timestamp_ns,
                },
            )
            return
        with self._projection_lock:
            self._latest_projection_frame = result
        lifting = self._lifting_pipeline.process(result)
        with self._projection_lock:
            self._latest_lifting_frame = lifting
        self._candidate_marker_publisher.publish(
            candidate_marker_array(lifting.candidates)
        )
        self._update_persistent_maps(result, lifting)
        self._trace(
            event='projection_completed',
            selected_action='retain_projection',
            selection_reason='valid_time_and_frame_association',
            details={
                'image_id': result.panorama.image_id,
                'current': asdict(result.current.diagnostics),
                'accumulated': asdict(result.accumulated.diagnostics),
                'dense_stats': asdict(
                    self._projection_pipeline.dense_accumulator.stats()
                ),
                'worker': self._projection_worker.stats(),
                'lifting': {
                    'candidate_count': len(lifting.candidates),
                    'result_count': len(lifting.results),
                    'ground_reason': lifting.ground_estimate.reason,
                    'results': tuple(
                        {
                            'detection_id': item.detection_id,
                            'status': item.status.value,
                            'reason': item.reason,
                            'counts': asdict(item.counts),
                            'processing_time_ms': item.processing_time_ms,
                        }
                        for item in lifting.results
                    ),
                },
            },
        )
        self._save_projection_debug(result, lifting)

    def _update_persistent_maps(
        self,
        result: ProjectionFrame,
        lifting: LiftingFrame,
    ) -> None:
        """Fuse lifted objects, extract walls, and anchor structural rays."""
        detections = {
            detection.detection_id: detection
            for detection in result.detections
        }
        pose = result.association.pose
        quaternion = pose.orientation_xyzw
        heading = atan2(
            2.0 * (
                quaternion[3] * quaternion[2]
                + quaternion[0] * quaternion[1]
            ),
            1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
        )
        candidates = list(lifting.candidates)
        observations = []
        for candidate in candidates:
            detection = detections.get(candidate.detection_id)
            crop = (
                _crop_detection(result.panorama.image_rgb, detection)
                if detection is not None else None
            )
            crop_score = _best_crop_score(candidate, detection, crop)
            if candidate.partial_geometry:
                visibility = 'partial'
            elif candidate.geometry_status is GeometryStatus.SPARSE:
                visibility = 'sparse'
            else:
                visibility = 'full'
            observations.append(ViewpointObservation(
                viewpoint_id=result.panorama.image_id,
                robot_pose_xyz_yaw=np.array([
                    *pose.position_xyz,
                    heading,
                ]),
                timestamp_ns=result.panorama.timestamp_ns,
                detection_id=candidate.detection_id,
                point_count=candidate.point_count,
                geometry_confidence=candidate.geometry_confidence,
                visibility=visibility,
                best_crop=crop,
                best_crop_score=crop_score,
            ))
        try:
            self._object_map.add_viewpoint_candidates(
                candidates, observations
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f'ObjectMap update rejected: {error}')
            self._trace(
                event='object_map_update_rejected',
                selected_action='retain_previous_object_map',
                selection_reason='invalid_candidate_or_observation',
                details={'error': str(error)},
            )
        else:
            for event in self._object_map.last_events:
                self._trace(
                    event='object_association',
                    selected_action=event.decision,
                    selection_reason=event.reason,
                    details=event.to_dict(),
                )
        self._object_map_marker_publisher.publish(object_map_marker_array(
            self._object_map.active_instances(),
            candidates=lifting.candidates,
            association_events=self._object_map.last_events,
        ))
        self._structural_frame_count += 1
        interval = max(1, int(
            self.get_parameter('structural_wall_update_interval').value
        ))
        if (
            self._structural_frame_count == 1
            or self._structural_frame_count % interval == 0
        ):
            try:
                wall_ids = self._structural_map.update_walls_from_points(
                    result.accumulated_snapshot.points_xyz,
                    timestamp_ns=result.panorama.timestamp_ns,
                    viewpoint_id=result.panorama.image_id,
                )
            except (TypeError, ValueError) as error:
                self.get_logger().warning(
                    f'Structural wall update rejected: {error}'
                )
                wall_ids = ()
            if wall_ids:
                self._trace(
                    event='structural_walls_updated',
                    selected_action='retain_wall_segments',
                    selection_reason='vertically_supported_line_fit',
                    details={
                        'wall_ids': wall_ids,
                        'wall_count': len(self._structural_map.walls()),
                    },
                )
        transform_map_from_camera = invert_transform(
            result.current.transform_camera_internal_from_map
        )
        for detection in result.detections:
            metadata = dict(detection.metadata)
            metadata.update({
                'viewpoint_id': result.panorama.image_id,
                'timestamp_ns': result.panorama.timestamp_ns,
            })
            annotated = replace(detection, metadata=metadata)
            anchor = self._structural_map.anchor_detection_to_wall(
                annotated, transform_map_from_camera
            )
            for event in self._structural_map.last_events:
                if event.reason == 'class_is_not_structural':
                    continue
                self._trace(
                    event='structural_anchor_association',
                    selected_action=event.decision,
                    selection_reason=event.reason,
                    details=event.to_dict(),
                )
            del anchor
        self._structural_map_marker_publisher.publish(
            structural_map_marker_array(
                self._structural_map.walls(),
                self._structural_map.anchors(),
                self._structural_map.last_events,
            )
        )

    def _save_projection_debug(
        self,
        result: ProjectionFrame,
        lifting: LiftingFrame,
    ) -> None:
        output_value = str(self.get_parameter('projection_debug_directory').value)
        max_saved = int(self.get_parameter('projection_max_saved_frames').value)
        if not output_value or self._saved_projection_count >= max_saved:
            return
        output = Path(output_value) / result.panorama.image_id
        output.mkdir(parents=True, exist_ok=True)
        images = {
            'current.png': draw_projection_overlay(
                result.panorama.image_rgb,
                result.current,
            ),
            'accumulated.png': draw_projection_overlay(
                result.panorama.image_rgb,
                result.accumulated,
            ),
            'detections.png': draw_detection_projection_overlay(
                result.panorama.image_rgb,
                result.current,
                result.detections,
                result.current_detection_support,
            ),
            'persistent_map.png': draw_persistent_map_top_down(
                [
                    self._object_map.record(instance.instance_id)
                    for instance in self._object_map.active_instances()
                ],
                self._structural_map.walls(),
                self._structural_map.anchors(),
                (
                    np.asarray(self._persistent_path_xy)
                    if self._persistent_path_xy else None
                ),
            ),
        }
        orientation = result.association.pose.orientation_xyzw
        heading = atan2(
            2.0 * (
                orientation[3] * orientation[2]
                + orientation[0] * orientation[1]
            ),
            1.0 - 2.0 * (orientation[1] ** 2 + orientation[2] ** 2),
        )
        images['top_down.png'] = draw_top_down_projection(
            result.association.scan.points_xyz,
            result.accumulated_snapshot.points_xyz,
            result.association.pose.position_xyz,
            heading,
        )
        lifting_projection = (
            result.current
            if lifting.source is GeometrySource.CURRENT
            else result.accumulated
        )
        for index, (detection, lifted) in enumerate(
            zip(result.detections, lifting.results)
        ):
            prefix = f'lift_{index:02d}_{detection.class_name.replace(" ", "_")}'
            images[f'{prefix}_stages.png'] = draw_lifting_stage_overlay(
                result.panorama.image_rgb,
                lifting_projection,
                detection,
                lifted,
            )
            images[f'{prefix}_depth.png'] = draw_depth_histogram(
                lifting_projection,
                lifted,
            )
            images[f'{prefix}_geometry.png'] = draw_candidate_orthographic(
                lifted,
                result.association.pose.position_xyz,
            )
        for filename, image_rgb in images.items():
            if not cv2.imwrite(
                str(output / filename),
                np.ascontiguousarray(image_rgb[..., ::-1]),
            ):
                raise RuntimeError(f'failed to save projection debug image {filename}')
        regression_category = str(
            self.get_parameter('projection_regression_category').value
        )
        if regression_category:
            save_projection_regression_case(
                output,
                category=regression_category,
                scene_id=str(
                    self.get_parameter('projection_regression_scene_id').value
                ),
                pose_id=str(
                    self.get_parameter('projection_regression_pose_id').value
                ),
                frame=result,
                transform_sensor_from_camera_optical=(
                    self._projection_pipeline.transform_sensor_from_camera_optical
                ),
                panorama_model=self._projection_pipeline.panorama_model,
                projection_config=self._projection_pipeline.projection_config,
                overlay_rgb=images['current.png'],
                notes=str(
                    self.get_parameter('projection_regression_notes').value
                ),
            )
        self._saved_projection_count += 1

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
                    known_object_count=len(
                        self._object_map.active_instances()
                    ),
                    known_structure_count=(
                        len(self._structural_map.walls())
                        + len(self._structural_map.anchors())
                    ),
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

    def _create_projection_pipeline(self) -> Day5ProjectionPipeline:
        translation = np.asarray(
            self.get_parameter('camera_translation_sensor_xyz').value,
            dtype=np.float64,
        )
        quaternion = np.asarray(
            self.get_parameter('camera_orientation_sensor_xyzw').value,
            dtype=np.float64,
        )
        transform_sensor_from_camera = make_transform(
            quaternion_xyzw_to_rotation(quaternion),
            translation,
        )
        association = AssociationConfig(
            max_pose_delta_ns=int(
                float(self.get_parameter('projection_max_pose_delta_ms').value)
                * 1_000_000
            ),
            max_scan_delta_ns=int(
                float(self.get_parameter('projection_max_scan_delta_ms').value)
                * 1_000_000
            ),
            buffer_duration_ns=int(
                float(self.get_parameter('projection_buffer_seconds').value)
                * 1_000_000_000
            ),
            max_pose_items=int(
                self.get_parameter('projection_max_pose_items').value
            ),
            max_scan_items=int(
                self.get_parameter('projection_max_scan_items').value
            ),
        )
        dense_config = DenseScanAccumulatorConfig(
            frame_id=str(self.get_parameter('scan_frame').value),
            voxel_size_m=float(
                self.get_parameter('dense_scan_voxel_size').value
            ),
            max_age_seconds=float(
                self.get_parameter('dense_scan_max_age_seconds').value
            ),
            max_radius_m=float(
                self.get_parameter('dense_scan_max_radius').value
            ),
            max_points=int(self.get_parameter('dense_scan_max_points').value),
        )
        projection_config = ProjectionConfig(
            expected_scan_frame=str(self.get_parameter('scan_frame').value),
            expected_pose_parent_frame=str(
                self.get_parameter('pose_parent_frame').value
            ),
            expected_pose_child_frame=str(
                self.get_parameter('pose_child_frame').value
            ),
            min_range_m=float(
                self.get_parameter('projection_min_range').value
            ),
            max_range_m=float(
                self.get_parameter('projection_max_range').value
            ),
            timing_warning_ms=float(
                self.get_parameter('projection_timing_warning_ms').value
            ),
        )
        quality_config = ProjectionQualityConfig(
            sparse_point_threshold=int(
                self.get_parameter('projection_sparse_point_threshold').value
            ),
            high_depth_iqr_m=float(
                self.get_parameter('projection_high_depth_iqr').value
            ),
            timing_warning_ms=projection_config.timing_warning_ms,
        )
        return Day5ProjectionPipeline(
            synchronizer=ProjectionSynchronizer(association),
            dense_accumulator=DenseRegisteredScanAccumulator(dense_config),
            transform_sensor_from_camera_optical=transform_sensor_from_camera,
            panorama_model=PanoramaCameraModel(
                int(self.get_parameter('panorama_width').value),
                int(self.get_parameter('panorama_height').value),
            ),
            projection_config=projection_config,
            quality_config=quality_config,
        )

    def _create_lifting_pipeline(self) -> Day6LiftingPipeline:
        source_text = str(self.get_parameter('lifting_source').value)
        try:
            source = GeometrySource(source_text)
        except ValueError as error:
            raise ValueError(
                'lifting_source must be current or accumulated'
            ) from error
        config = ObjectLiftingConfig(
            selection=PointSelectionConfig(
                bbox_inner_margin_fraction=float(
                    self.get_parameter('lifting_bbox_inner_margin').value
                )
            ),
            depth=DepthFilterConfig(
                bin_width_m=float(
                    self.get_parameter('lifting_depth_bin_width').value
                ),
                minimum_mode_points=int(
                    self.get_parameter('lifting_depth_minimum_mode_points').value
                ),
                maximum_band_width_m=float(
                    self.get_parameter('lifting_depth_maximum_band').value
                ),
            ),
            clustering=ClusterSelectionConfig(
                base_epsilon_m=float(
                    self.get_parameter('lifting_dbscan_base_epsilon').value
                ),
                range_epsilon_slope=float(
                    self.get_parameter('lifting_dbscan_range_slope').value
                ),
                minimum_samples=int(
                    self.get_parameter('lifting_dbscan_minimum_samples').value
                ),
            ),
            boxes=BoxEstimationConfig(
                lower_percentile=float(
                    self.get_parameter('lifting_box_lower_percentile').value
                ),
                upper_percentile=float(
                    self.get_parameter('lifting_box_upper_percentile').value
                ),
            ),
            orientation=OrientationConfidenceConfig(
                low_confidence=float(
                    self.get_parameter('lifting_orientation_low_confidence').value
                ),
                high_confidence=float(
                    self.get_parameter('lifting_orientation_high_confidence').value
                ),
            ),
            ground_clearance_m=float(
                self.get_parameter('lifting_ground_clearance').value
            ),
            floor_standing_clearance_m=float(
                self.get_parameter('lifting_floor_standing_clearance').value
            ),
            sparse_point_threshold=int(
                self.get_parameter('lifting_sparse_point_threshold').value
            ),
        )
        return Day6LiftingPipeline(
            ObjectLifter(config),
            source=source,
            use_masks=bool(self.get_parameter('lifting_use_masks').value),
        )

    def _create_object_map(self) -> ObjectMap:
        return ObjectMap(ObjectMapConfig(
            association=ObjectAssociationConfig(
                accept_threshold=float(
                    self.get_parameter('object_map_accept_threshold').value
                ),
                uncertain_threshold=float(
                    self.get_parameter('object_map_uncertain_threshold').value
                ),
                yaw_confidence_threshold=float(self.get_parameter(
                    'object_map_yaw_confidence_threshold'
                ).value),
            ),
            same_keyframe_distance_m=float(self.get_parameter(
                'object_map_same_keyframe_distance'
            ).value),
            same_keyframe_overlap_threshold=float(self.get_parameter(
                'object_map_same_keyframe_overlap'
            ).value),
            fused_voxel_size_m=float(
                self.get_parameter('object_map_fused_voxel_size').value
            ),
            max_fused_points_per_instance=int(self.get_parameter(
                'object_map_max_points_per_instance'
            ).value),
            max_total_fused_points=int(
                self.get_parameter('object_map_max_total_points').value
            ),
            max_observation_history=int(self.get_parameter(
                'object_map_max_observation_history'
            ).value),
        ))

    def _create_structural_map(self) -> StructuralMap:
        wall_config = WallExtractionConfig(
            min_height_above_ground_m=float(self.get_parameter(
                'wall_min_height_above_ground'
            ).value),
            min_segment_length_m=float(
                self.get_parameter('wall_min_segment_length').value
            ),
            max_line_residual_m=float(
                self.get_parameter('wall_max_line_residual').value
            ),
            merge_angle_deg=float(
                self.get_parameter('wall_merge_angle_deg').value
            ),
            merge_perpendicular_distance_m=float(self.get_parameter(
                'wall_merge_perpendicular_distance'
            ).value),
            preserve_opening_width_m=float(
                self.get_parameter('wall_preserve_opening_width').value
            ),
            max_candidate_points=int(
                self.get_parameter('wall_max_candidate_points').value
            ),
        )
        return StructuralMap(StructuralMapConfig(
            wall_extraction=wall_config,
            ray_parallel_epsilon=float(
                self.get_parameter('structural_ray_parallel_epsilon').value
            ),
            max_wall_extent_margin_m=float(self.get_parameter(
                'structural_max_wall_extent_margin'
            ).value),
            anchor_merge_distance_m=float(self.get_parameter(
                'structural_anchor_merge_distance'
            ).value),
            ambiguous_wall_distance_margin_m=float(self.get_parameter(
                'structural_ambiguous_wall_distance_margin'
            ).value),
        ))

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
        self.declare_parameter('pose_parent_frame', 'map')
        self.declare_parameter('pose_child_frame', 'sensor')
        self.declare_parameter('panorama_width', 1920)
        self.declare_parameter('panorama_height', 640)
        self.declare_parameter('camera_translation_sensor_xyz', [0.0, 0.0, 0.1])
        self.declare_parameter(
            'camera_orientation_sensor_xyzw',
            [-0.5, 0.5, -0.5, 0.5],
        )
        self.declare_parameter('projection_max_pose_delta_ms', 50.0)
        self.declare_parameter('projection_max_scan_delta_ms', 150.0)
        self.declare_parameter('projection_buffer_seconds', 5.0)
        self.declare_parameter('projection_max_pose_items', 2_000)
        self.declare_parameter('projection_max_scan_items', 64)
        self.declare_parameter('projection_min_range', 0.30)
        self.declare_parameter('projection_max_range', 30.0)
        self.declare_parameter('projection_timing_warning_ms', 100.0)
        self.declare_parameter('projection_sparse_point_threshold', 8)
        self.declare_parameter('projection_high_depth_iqr', 2.0)
        self.declare_parameter('dense_scan_voxel_size', 0.04)
        self.declare_parameter('dense_scan_max_age_seconds', 15.0)
        self.declare_parameter('dense_scan_max_radius', 12.0)
        self.declare_parameter('dense_scan_max_points', 1_000_000)
        self.declare_parameter('projection_worker_queue_size', 2)
        self.declare_parameter('projection_shutdown_timeout', 2.0)
        self.declare_parameter('projection_debug_directory', '')
        self.declare_parameter('projection_max_saved_frames', 5)
        self.declare_parameter('projection_regression_category', '')
        self.declare_parameter('projection_regression_scene_id', 'unknown')
        self.declare_parameter('projection_regression_pose_id', 'unknown')
        self.declare_parameter(
            'projection_regression_notes',
            'Live Day 5 camera-LiDAR alignment regression.',
        )
        self.declare_parameter('lifting_source', 'accumulated')
        self.declare_parameter('lifting_use_masks', False)
        self.declare_parameter('lifting_bbox_inner_margin', 0.05)
        self.declare_parameter('lifting_ground_clearance', 0.07)
        self.declare_parameter('lifting_floor_standing_clearance', 0.02)
        self.declare_parameter('lifting_depth_bin_width', 0.15)
        self.declare_parameter('lifting_depth_minimum_mode_points', 5)
        self.declare_parameter('lifting_depth_maximum_band', 1.5)
        self.declare_parameter('lifting_dbscan_base_epsilon', 0.07)
        self.declare_parameter('lifting_dbscan_range_slope', 0.015)
        self.declare_parameter('lifting_dbscan_minimum_samples', 5)
        self.declare_parameter('lifting_box_lower_percentile', 2.5)
        self.declare_parameter('lifting_box_upper_percentile', 97.5)
        self.declare_parameter('lifting_orientation_low_confidence', 0.40)
        self.declare_parameter('lifting_orientation_high_confidence', 0.70)
        self.declare_parameter('lifting_sparse_point_threshold', 8)
        self.declare_parameter('object_map_accept_threshold', 0.62)
        self.declare_parameter('object_map_uncertain_threshold', 0.55)
        self.declare_parameter('object_map_yaw_confidence_threshold', 0.50)
        self.declare_parameter('object_map_same_keyframe_distance', 0.30)
        self.declare_parameter('object_map_same_keyframe_overlap', 0.60)
        self.declare_parameter('object_map_fused_voxel_size', 0.03)
        self.declare_parameter('object_map_max_points_per_instance', 50_000)
        self.declare_parameter('object_map_max_total_points', 500_000)
        self.declare_parameter('object_map_max_observation_history', 100)
        self.declare_parameter('structural_wall_update_interval', 10)
        self.declare_parameter('wall_min_height_above_ground', 0.20)
        self.declare_parameter('wall_min_segment_length', 0.50)
        self.declare_parameter('wall_max_line_residual', 0.08)
        self.declare_parameter('wall_merge_angle_deg', 5.0)
        self.declare_parameter('wall_merge_perpendicular_distance', 0.12)
        self.declare_parameter('wall_preserve_opening_width', 0.60)
        self.declare_parameter('wall_max_candidate_points', 50_000)
        self.declare_parameter('structural_ray_parallel_epsilon', 1.0e-6)
        self.declare_parameter('structural_max_wall_extent_margin', 0.25)
        self.declare_parameter('structural_anchor_merge_distance', 0.55)
        self.declare_parameter(
            'structural_ambiguous_wall_distance_margin', 0.10
        )
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


def _decode_image_rgb(message: Image) -> np.ndarray:
    """Decode contiguous/padded RGB8 or BGR8 ROS images without CvBridge."""
    if message.encoding not in {'rgb8', 'bgr8'}:
        raise ValueError(f'unsupported camera encoding {message.encoding!r}')
    if message.height < 2 or message.width < 2:
        raise ValueError('camera image dimensions must be at least 2 x 2')
    minimum_step = message.width * 3
    if message.step < minimum_step:
        raise ValueError('camera image step is smaller than packed RGB data')
    data = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = message.height * message.step
    if data.size != expected_size:
        raise ValueError('camera image data size does not match height and step')
    rows = data.reshape((message.height, message.step))
    image = rows[:, :minimum_step].reshape((message.height, message.width, 3))
    if message.encoding == 'bgr8':
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def _crop_detection(
    panorama_rgb: np.ndarray,
    detection: Detection2D,
) -> np.ndarray | None:
    """Return a bounded wrap-aware panorama crop for best-view memory."""
    image = np.asarray(panorama_rgb)
    y_min = max(0, int(np.floor(detection.panorama_box.y_min)))
    y_max = min(image.shape[0], int(np.ceil(detection.panorama_box.y_max)))
    if y_max <= y_min:
        return None
    pieces = []
    for x_min, x_max in detection.panorama_box.x_intervals:
        left = max(0, int(np.floor(x_min)))
        right = min(image.shape[1], int(np.ceil(x_max)))
        if right > left:
            pieces.append(image[y_min:y_max, left:right])
    if not pieces:
        return None
    return np.ascontiguousarray(np.concatenate(pieces, axis=1)).copy()


def _best_crop_score(
    candidate: ObjectCandidate3D,
    detection: Detection2D | None,
    crop: np.ndarray | None,
) -> float:
    """Score crop evidence from detection, geometry, area, and point support."""
    if detection is None or crop is None or crop.size == 0:
        return 0.0
    area_score = min(1.0, float(np.sqrt(crop.shape[0] * crop.shape[1])) / 300.0)
    support_score = min(1.0, candidate.point_count / 100.0)
    score = (
        0.45 * detection.confidence
        + 0.35 * candidate.geometry_confidence
        + 0.10 * area_score
        + 0.10 * support_score
    )
    return min(1.0, max(0.0, float(score)))


def main(args: list[str] | None = None) -> None:
    """Run the Q-MapNav ROS node until shutdown."""
    rclpy.init(args=args)
    node = QMapNavNode(
        perception_worker=make_day4_baseline_worker(1920, 640),
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
