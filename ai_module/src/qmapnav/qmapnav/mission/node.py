"""ROS 2 composition root for Q-MapNav."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import asdict
from math import atan2

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import numpy as np
from qmapnav.common import TaskSpecification
from qmapnav.common.decision_trace import DecisionTraceEvent
from qmapnav.common.decision_trace import TraceRecorder
from qmapnav.counting import CountStabilityConfig
from qmapnav.language import parse_question
from qmapnav.mapping import AccumulationStatus
from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import ProjectionFrame
from qmapnav.mapping import ProjectionPipeline
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping import TimedPose
from qmapnav.mapping import TimedRegisteredScan
from qmapnav.mapping.lifting_pipeline import LiftingFrame
from qmapnav.mapping.lifting_pipeline import LiftingPipeline
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.occupancy_grid import occupancy_from_scan_accumulator
from qmapnav.mapping.point_cloud import decode_scan_arrays
from qmapnav.mapping.point_cloud import ScanArrays
from qmapnav.mapping.point_cloud import stamp_to_nanoseconds
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mission.episode_reports import write_object_reference_result
from qmapnav.mission.instruction_episode import InstructionEpisodeCoordinator
from qmapnav.mission.marker_adapter import CANDIDATE_MARKER_TOPIC
from qmapnav.mission.marker_adapter import FinalObjectAnswerGuard
from qmapnav.mission.marker_adapter import OBJECT_MAP_MARKER_TOPIC
from qmapnav.mission.marker_adapter import OFFICIAL_MARKER_TOPIC
from qmapnav.mission.marker_adapter import RELATION_MARKER_TOPIC
from qmapnav.mission.marker_adapter import STRUCTURAL_MAP_MARKER_TOPIC
from qmapnav.mission.marker_adapter import validate_marker_spec
from qmapnav.mission.numerical_episode import NumericalEpisodeCoordinator
from qmapnav.mission.numerical_episode import NumericalEpisodeState
from qmapnav.mission.numerical_output_adapter import NumericalOutputAdapter
from qmapnav.mission.numerical_output_adapter import OFFICIAL_NUMERICAL_TOPIC
from qmapnav.mission.perception_runtime import (
    _crop_colour_support as _perception_crop_colour_support,
    _decode_image_rgb as _perception_decode_image_rgb,
    decode_image_rgb,
    PerceptionRuntime,
)
from qmapnav.mission.question_latch import QuestionLatch
from qmapnav.mission.question_latch import QuestionLatchStatus
from qmapnav.mission.runtime_config import colour_prototype_path
from qmapnav.mission.runtime_config import declare_parameters
from qmapnav.mission.runtime_config import make_accumulator
from qmapnav.mission.runtime_config import make_colour_classifier_config
from qmapnav.mission.runtime_config import make_colour_selection_config
from qmapnav.mission.runtime_config import make_lifting_pipeline
from qmapnav.mission.runtime_config import make_object_map
from qmapnav.mission.runtime_config import make_projection_pipeline
from qmapnav.mission.runtime_config import make_reasoning_configs
from qmapnav.mission.runtime_config import make_relation_graph
from qmapnav.mission.runtime_config import make_structural_map
from qmapnav.mission.runtime_config import make_targeted_viewpoint_config
from qmapnav.mission.runtime_config import make_trace_recorder
from qmapnav.navigation import EvidenceSufficiency
from qmapnav.navigation import ExecutorEvent
from qmapnav.navigation import ExecutorEventType
from qmapnav.navigation import generate_targeted_viewpoints
from qmapnav.navigation import OneViewpointGuard
from qmapnav.navigation import SemanticStageExecutor
from qmapnav.navigation import SemanticStageState
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import stage_waypoints
from qmapnav.navigation import TwoStageRouteError
from qmapnav.navigation import Waypoint2D
from qmapnav.navigation import WaypointExecutorState
from qmapnav.perception.baseline import make_default_perception_worker
from qmapnav.reasoning.colour_prototypes import load_colour_prototypes
from qmapnav.reasoning.object_reference_solver import (
    resolve_object_reference_from_maps,
)
from qmapnav.reasoning.relation_graph import RelationGraph
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


_crop_colour_support = _perception_crop_colour_support
_decode_image_rgb = _perception_decode_image_rgb


class QMapNavNode(Node):
    """Own the ROS lifecycle and composition of Q-MapNav subsystems."""

    def __init__(
        self,
        question_parser: Callable[[str], TaskSpecification] = parse_question,
        *,
        scan_accumulator: RegisteredScanAccumulator | None = None,
        projection_pipeline: ProjectionPipeline | None = None,
        lifting_pipeline: LiftingPipeline | None = None,
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
        declare_parameters(self)
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
        self._perception_worker = perception_worker
        self._processed_image_ids: deque[str] = deque(maxlen=128)
        self._processed_image_id_set: set[str] = set()
        self._persistent_path_xy: deque[tuple[float, float]] = deque(
            maxlen=2048
        )
        self._object_reference_state = 'idle'
        self._object_reference_projection_count = 0
        self._object_reference_reobservation_start = 0
        self._object_reference_resolution = None
        self._object_reference_viewpoint_guard = OneViewpointGuard()
        self._object_reference_viewpoint_reason: str | None = None
        self._object_reference_selected_viewpoint = None
        self._object_reference_started_at: float | None = None
        self._object_reference_fusion_events: deque[dict] = deque(maxlen=512)
        self._numerical_episode: NumericalEpisodeCoordinator | None = None
        self._instruction_episode: InstructionEpisodeCoordinator | None = None
        self._instruction_stage_executor: SemanticStageExecutor | None = None
        self._instruction_plan = None
        self._instruction_grid = None
        self._instruction_state = 'idle'
        self._instruction_projection_count = 0
        self._instruction_reobservation_start = 0
        self._instruction_selected_viewpoint = None
        self._instruction_motion_started_at: float | None = None

        self._question_latch = QuestionLatch()
        self._task_specification: TaskSpecification | None = None
        self._scan_accumulator = scan_accumulator or make_accumulator(self)
        self._projection_pipeline = (
            projection_pipeline or make_projection_pipeline(self)
        )
        self._lifting_pipeline = (
            lifting_pipeline or make_lifting_pipeline(self)
        )
        self._object_map = object_map or make_object_map(self)
        self._structural_map = structural_map or make_structural_map(self)
        self._colour_selection_config = make_colour_selection_config(self)
        self._colour_classifier_config = make_colour_classifier_config(self)
        self._colour_prototypes = load_colour_prototypes(
            colour_prototype_path(self)
        )
        self._relation_graph = make_relation_graph(self)
        (
            self._reasoning_candidate_config,
            self._reasoning_spatial_config,
            self._reasoning_ambiguity_config,
            self._reasoning_corridor_config,
        ) = make_reasoning_configs(self)
        self._trace_recorder = trace_recorder or make_trace_recorder(self)
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
        self._relation_marker_publisher = self.create_publisher(
            MarkerArray,
            RELATION_MARKER_TOPIC,
            5,
        )
        self._official_marker_publisher = self.create_publisher(
            Marker,
            OFFICIAL_MARKER_TOPIC,
            5,
        )
        self._numerical_publisher = self.create_publisher(
            Int32,
            OFFICIAL_NUMERICAL_TOPIC,
            5,
        )
        self._numerical_output_adapter = NumericalOutputAdapter(
            self._numerical_publisher.publish
        )
        self._final_object_answer_guard = FinalObjectAnswerGuard(
            self._official_marker_publisher.publish,
            self._publish_matching_object_waypoint,
            publish_matching_waypoint=bool(
                self.get_parameter('publish_object_matching_waypoint').value
            ),
            orientation_confidence_threshold=float(
                self.get_parameter('lifting_orientation_low_confidence').value
            ),
        )
        self._watchdog_timer = self.create_timer(
            float(self.get_parameter('watchdog_period').value),
            self._on_watchdog,
        )
        self._trace(
            event='node_initialized',
            selection_reason='colour_relation_runtime_ready',
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
                'colour_selection': asdict(self._colour_selection_config),
                'colour_classifier': asdict(self._colour_classifier_config),
                'colour_prototype_path': str(colour_prototype_path(self)),
            },
        )
        self._perception_runtime = PerceptionRuntime(
            self,
            projection_pipeline=self._projection_pipeline,
            lifting_pipeline=self._lifting_pipeline,
            object_map=self._object_map,
            structural_map=self._structural_map,
            relation_graph=self._relation_graph,
            colour_selection_config=self._colour_selection_config,
            colour_classifier_config=self._colour_classifier_config,
            colour_prototypes=self._colour_prototypes,
            persistent_path_xy=self._persistent_path_xy,
            object_reference_fusion_events=self._object_reference_fusion_events,
            candidate_marker_publisher=self._candidate_marker_publisher,
            object_map_marker_publisher=self._object_map_marker_publisher,
            structural_map_marker_publisher=self._structural_map_marker_publisher,
            relation_marker_publisher=self._relation_marker_publisher,
            perception_worker=self._perception_worker,
        )
        self.get_logger().info('Q-MapNav node initialized')

    @property
    def question_latch(self) -> QuestionLatch:
        """Expose read-only latch state for composition and evaluation."""
        return self._question_latch

    def configure_perception_worker(self, worker: object) -> None:
        """Install the submission detector once, before ROS spinning starts."""
        if worker is None:
            raise ValueError('perception worker must not be None')
        if self._perception_worker is not None:
            raise RuntimeError('perception worker is already configured')
        self._perception_worker = worker
        self._perception_runtime.set_perception_worker(worker)

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
        """Return the latest immutable projection result."""
        return self._perception_runtime.latest_projection_frame

    @property
    def latest_lifting_frame(self) -> LiftingFrame | None:
        """Return the latest immutable single-observation lifting result."""
        return self._perception_runtime.latest_lifting_frame

    @property
    def object_map(self) -> ObjectMap:
        """Expose the bounded episode-local persistent object map."""
        return self._object_map

    @property
    def structural_map(self) -> StructuralMap:
        """Expose the bounded episode-local architectural map."""
        return self._structural_map

    @property
    def relation_graph(self) -> RelationGraph:
        """Expose current derived Day 8 relations for downstream reasoning."""
        return self._relation_graph

    def reset_persistent_maps(self) -> None:
        """Reset Day 7 object and structural identities at an episode boundary."""
        self._object_map.reset_episode()
        self._structural_map.reset_episode()
        self._relation_graph.recompute([])
        self._persistent_path_xy.clear()

    def _advance_object_reference_episode(self) -> None:
        """Rank after bounded evidence and optionally reobserve exactly once."""
        task = self._task_specification
        if task is None or task.task_type != 'object_reference':
            return
        if self._object_reference_state in {
            'committed', 'marker_published', 'terminal',
        }:
            return
        self._object_reference_projection_count += 1
        if self._object_reference_state == 'initial_observation':
            required = int(
                self.get_parameter(
                    'object_reference_initial_observations'
                ).value
            )
            if self._object_reference_projection_count < required:
                return
            resolution = self._rank_object_reference()
            self._object_reference_resolution = resolution
            self._maybe_request_targeted_viewpoint(resolution)
            return
        if (
            self._object_reference_state == 'optional_reobservation'
            and self._waypoint_executor.state is WaypointExecutorState.COMPLETE
            and self._object_reference_projection_count
            > self._object_reference_reobservation_start
        ):
            resolution = self._rank_object_reference()
            self._object_reference_resolution = resolution
            self._trace(
                event='object_reference_final_ranking',
                selected_action='commit_after_one_reobservation',
                selection_reason='one_viewpoint_maximum_reached',
                details=resolution.to_dict(),
            )
            self._commit_object_reference_answer(resolution)

    def _rank_object_reference(self):
        task = self._task_specification
        if task is None:
            raise RuntimeError('object-reference task is unavailable')
        resolution = resolve_object_reference_from_maps(
            task,
            self._object_map,
            self._structural_map,
            candidate_config=self._reasoning_candidate_config,
            spatial_config=self._reasoning_spatial_config,
            vertical_config=self._relation_graph.vertical_config,
            support_config=self._relation_graph.support_config,
            ambiguity_config=self._reasoning_ambiguity_config,
        )
        self._trace(
            event='object_reference_ranking',
            selected_action='rank_persistent_targets',
            selection_reason='complete_constraint_scoring',
            details=resolution.to_dict(),
        )
        return resolution

    def _maybe_request_targeted_viewpoint(self, resolution) -> None:
        missing = tuple(
            entity.entity_id
            for entity in self._task_specification.entities[1:]
            if not resolution.candidate_generation[entity.entity_id].retained
        )
        selected = resolution.selected_target_id
        geometry_confidence = None
        if selected is not None and selected.isdigit():
            geometry_confidence = self._object_map.record(
                int(selected)
            ).geometry_confidence
        evidence = EvidenceSufficiency(
            reliable_target_candidates=len(
                resolution.ranked_hypotheses
            ),
            missing_anchor_ids=missing,
            confidence_margin=resolution.normalized_margin,
            geometry_confidence=geometry_confidence,
            likely_occluded=False,
            time_remaining_sec=max(
                0.0, self._episode_time_limit - self._elapsed()
            ),
            viewpoint_attempts=int(
                self._object_reference_viewpoint_guard.attempted
            ),
        )
        policy = make_targeted_viewpoint_config(self)
        from qmapnav.navigation import decide_targeted_viewpoint
        decision = decide_targeted_viewpoint(evidence, policy)
        self._object_reference_viewpoint_reason = decision.reason
        if not decision.requested or self._latest_pose_xy is None:
            self._commit_object_reference_answer(resolution)
            return
        focus = self._object_reference_focus_xy(resolution)
        heading = self._latest_heading()
        candidates = generate_targeted_viewpoints(
            (*self._latest_pose_xy, heading),
            focus,
            anchor_missing=bool(missing),
            ambiguous=(resolution.resolution_status == 'ambiguous'),
            safe_pose=lambda x, y: self._scan_accumulator.is_known_free(
                x,
                y,
                clearance=float(
                    self.get_parameter(
                        'targeted_viewpoint_clearance_m'
                    ).value
                ),
            ),
            config=policy,
        )
        viewpoint = self._object_reference_viewpoint_guard.select(candidates)
        if viewpoint is None:
            self._trace(
                event='targeted_viewpoint_skipped',
                selected_action='commit_initial_ranking',
                selection_reason='no_safe_reachable_viewpoint',
                details={'trigger_reason': decision.reason},
            )
            self._commit_object_reference_answer(resolution)
            return
        self._object_reference_selected_viewpoint = viewpoint
        self._object_reference_reobservation_start = (
            self._object_reference_projection_count
        )
        self._object_reference_state = 'optional_reobservation'
        self._trace(
            event='targeted_viewpoint_selected',
            selected_action='execute_one_reobservation',
            selection_reason=decision.reason,
            details={
                'pose_xy_heading': viewpoint.pose_xy_heading,
                'utility': viewpoint.utility,
                'travel_cost_m': viewpoint.travel_cost_m,
                'candidate_count': len(candidates),
            },
        )
        self.start_route([Waypoint2D(*viewpoint.pose_xy_heading)])

    def _object_reference_focus_xy(self, resolution):
        selected = resolution.selected_target_id
        if selected is not None and selected.isdigit():
            centre = self._object_map.get(int(selected)).centroid_xyz
            return float(centre[0]), float(centre[1])
        for entity in self._task_specification.entities[1:]:
            generated = resolution.candidate_generation[entity.entity_id]
            for item in generated.retained:
                if item.geometry is not None:
                    centre = item.geometry.centre_xyz
                    return float(centre[0]), float(centre[1])
        heading = self._latest_heading()
        return (
            float(self._latest_pose_xy[0] + 2.0 * np.cos(heading)),
            float(self._latest_pose_xy[1] + 2.0 * np.sin(heading)),
        )

    def _latest_heading(self) -> float:
        pose = self._latest_timed_pose
        if pose is None:
            return 0.0
        orientation = pose.orientation_xyzw
        return atan2(
            2.0 * (
                orientation[3] * orientation[2]
                + orientation[0] * orientation[1]
            ),
            1.0 - 2.0 * (orientation[1] ** 2 + orientation[2] ** 2),
        )

    @staticmethod
    def _projection_viewpoint_id(result: ProjectionFrame) -> str:
        """Quantize map pose so stationary frames are not independent views."""
        pose = result.association.pose
        x, y = pose.position_xyz[:2]
        orientation = pose.orientation_xyzw
        yaw = atan2(
            2.0 * (
                orientation[3] * orientation[2]
                + orientation[0] * orientation[1]
            ),
            1.0 - 2.0 * (
                orientation[1] ** 2 + orientation[2] ** 2
            ),
        )
        return 'map_pose_{:.1f}_{:.1f}_{:.1f}'.format(
            round(float(x) * 2.0) / 2.0,
            round(float(y) * 2.0) / 2.0,
            round(float(yaw) * 6.0) / 6.0,
        )

    def _commit_object_reference_answer(self, resolution) -> None:
        if self._object_reference_state in {'committed', 'marker_published'}:
            return
        self._object_reference_state = 'committed'
        selected = None if resolution is None else resolution.selected_target_id
        marker_errors = ('missing_target_candidate',)
        marker_spec = None
        if selected is not None and selected.isdigit():
            try:
                instance = self._object_map.get(int(selected))
                answer = self._final_object_answer_guard.commit(
                    instance,
                    timestamp_ns=self.get_clock().now().nanoseconds,
                )
                marker_spec = answer.marker
                marker_errors = validate_marker_spec(marker_spec)
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                marker_errors = (f'commit_failed:{error}',)
        protocol_valid = marker_spec is not None and not marker_errors
        self._object_reference_state = (
            'marker_published' if protocol_valid else 'terminal'
        )
        self._trace(
            event='official_object_answer_committed',
            selected_action=(
                'publish_marker_and_matching_waypoint'
                if protocol_valid else 'log_no_valid_marker_response'
            ),
            selection_reason=(
                'final_persistent_target_snapshot'
                if protocol_valid else 'no_publishable_persistent_target'
            ),
            terminal_status=(
                'complete' if protocol_valid else 'protocol_failure'
            ),
            details={
                'selected_target_id': selected,
                'marker_errors': marker_errors,
                'targeted_viewpoint_used': (
                    self._object_reference_selected_viewpoint is not None
                ),
            },
        )
        write_object_reference_result(
            self, resolution, marker_spec, marker_errors, protocol_valid
        )

    def _advance_numerical_episode(self, viewpoint_id: str) -> None:
        """Fold one persistent-map snapshot into bounded count stability."""
        coordinator = self._numerical_episode
        if (
            coordinator is None
            or coordinator.state is not NumericalEpisodeState.COLLECTING
        ):
            return
        action = coordinator.evaluate(
            self._object_map,
            self._structural_map,
            viewpoint_id=viewpoint_id,
            time_remaining_sec=max(
                0.0, self._episode_time_limit - self._elapsed()
            ),
            episode_time_sec=self._elapsed(),
        )
        self._trace(
            event='numerical_count_evaluated',
            selected_action=action.action,
            selection_reason=action.reason,
            details=action.to_dict(),
        )
        if action.action == 'commit':
            self._publish_numerical_action(action)

    def _force_numerical_commit(self, reason: str) -> None:
        """Publish a best-available count before a deadline."""
        coordinator = self._numerical_episode
        if (
            coordinator is None
            or coordinator.state is not NumericalEpisodeState.COLLECTING
        ):
            return
        action = coordinator.force_commit(
            self._object_map,
            self._structural_map,
            reason=reason,
        )
        self._trace(
            event='numerical_deadline_commit',
            selected_action='publish_best_available_count',
            selection_reason=reason,
            details=action.to_dict(),
        )
        self._publish_numerical_action(action)

    def _publish_numerical_action(self, action) -> None:
        """Commit exactly one official Int32 response, including zero."""
        was_committed = self._numerical_output_adapter.committed
        commitment = self._numerical_output_adapter.commit(action.result)
        if not was_committed and self._numerical_episode is not None:
            self._numerical_episode.notify_published()
            self._trace(
                event='official_numerical_response_committed',
                selected_action='publish_numerical_response',
                selection_reason=commitment.reason,
                terminal_status='complete',
                details={
                    'count': commitment.count,
                    'stable': commitment.stable,
                    'topic': OFFICIAL_NUMERICAL_TOPIC,
                    'message_type': 'std_msgs/msg/Int32',
                },
            )

    def _advance_instruction_episode(self, *, force: bool = False) -> None:
        """Evaluate the live two-stage instruction after a map update."""
        task = self._task_specification
        coordinator = self._instruction_episode
        if (
            task is None
            or task.task_type != 'instruction_following'
            or coordinator is None
            or self._latest_pose_xy is None
        ):
            return
        if self._instruction_state not in {
            'initial_observation', 'reobservation'
        }:
            return

        self._instruction_projection_count += 1
        if self._instruction_state == 'initial_observation' and not force:
            required = int(self.get_parameter(
                'instruction_initial_observations'
            ).value)
            if self._instruction_projection_count < required:
                return
        if self._instruction_state == 'reobservation' and not force:
            required = int(self.get_parameter(
                'instruction_reobservation_observations'
            ).value)
            if (
                self._instruction_projection_count
                - self._instruction_reobservation_start
                < required
            ):
                return

        grid = occupancy_from_scan_accumulator(
            self._scan_accumulator,
            centre_xy=self._latest_pose_xy,
            half_extent_m=float(self.get_parameter(
                'instruction_grid_half_extent_m'
            ).value),
            resolution=float(self.get_parameter(
                'instruction_grid_resolution_m'
            ).value),
            clearance=float(self.get_parameter(
                'instruction_grid_known_free_clearance_m'
            ).value),
        )
        action = coordinator.evaluate(
            self._object_map,
            self._structural_map,
            grid=grid,
            current_pose_xy_yaw=(
                self._latest_pose_xy[0],
                self._latest_pose_xy[1],
                self._latest_heading(),
            ),
            time_remaining_sec=max(
                0.0, self._episode_time_limit - self._elapsed()
            ),
        )
        self._instruction_grid = grid
        self._trace(
            event='two_stage_route_decision',
            candidate_actions=('route', 'explore', 'fallback', 'abort'),
            selected_action=action.action,
            selection_reason=action.reason,
            details=action.to_dict(),
        )
        if action.action in {'route', 'fallback'}:
            self._start_instruction_plan(action.plan, grid, action.action)
            return
        if action.action == 'explore':
            viewpoint = action.selection.selected
            if viewpoint is None:
                self._instruction_state = 'terminal'
                return
            self._instruction_selected_viewpoint = viewpoint
            self._instruction_state = 'exploration_active'
            self._instruction_motion_started_at = self._now()
            self.start_route([Waypoint2D(*viewpoint.pose_xy_yaw)])
            return
        self._instruction_state = 'terminal'

    def _start_instruction_plan(self, plan, grid, action: str) -> None:
        """Start semantic stage A and publish only its first waypoint."""
        if plan is None or self._latest_pose_xy is None:
            self._instruction_state = 'terminal'
            return
        waypoints = stage_waypoints(
            plan,
            0,
            grid=grid,
            start_xy=self._latest_pose_xy,
            spacing_m=float(self.get_parameter(
                'instruction_waypoint_spacing_m'
            ).value),
        )
        if not waypoints:
            self._instruction_state = 'terminal'
            self._trace(
                event='instruction_route_failed',
                selected_action='terminate_without_unreachable_waypoint',
                selection_reason='stage_a_path_unavailable',
                terminal_status='failed',
            )
            return
        self._instruction_plan = plan
        self._instruction_stage_executor = SemanticStageExecutor(
            plan, self._waypoint_executor
        )
        first = self._instruction_stage_executor.start(
            waypoints, now=self._now()
        )
        self._instruction_state = 'semantic_route_active'
        self._publish_waypoint(first)
        self._record_executor_events()
        self._trace(
            event='instruction_route_started',
            selected_action='publish_stage_a_first_waypoint',
            selection_reason=f'{action}:{plan.route_status}',
            active_route_index=0,
            details={
                'waypoint_count': len(waypoints),
                'plan': plan.to_dict(),
            },
        )

    def _update_instruction_pose(
        self,
        x: float,
        y: float,
        heading: float,
    ) -> bool:
        """Advance semantic execution and begin stage B only after stage A."""
        executor = self._instruction_stage_executor
        if (
            self._instruction_state != 'semantic_route_active'
            or executor is None
        ):
            return False
        next_goal = executor.update_pose(
            x, y, heading, now=self._now()
        )
        if next_goal is not None:
            self._publish_waypoint(next_goal)
        self._record_executor_events()
        self._record_semantic_stage_events()
        if executor.state is SemanticStageState.VERIFY_STAGE_A:
            if self._instruction_plan.route_status == 'stage_a_only':
                self._finish_instruction_stage_a_information(
                    (x, y, heading)
                )
            else:
                self._begin_instruction_stage_b((x, y))
        elif executor.state is SemanticStageState.COMPLETE:
            self._instruction_state = 'complete'
            self._trace(
                event='instruction_route_completed',
                selected_action='finish_instruction_episode',
                selection_reason='terminal_semantic_region_satisfied',
                terminal_status='complete',
            )
        return True

    def _finish_instruction_stage_a_information(
        self,
        pose_xy_yaw: tuple[float, float, float],
    ) -> None:
        """Re-observe after a grounded stage A when stage B was missing."""
        coordinator = self._instruction_episode
        if coordinator is None:
            self._instruction_state = 'terminal'
            return
        coordinator.notify_stage_a_information_arrived(
            pose_xy_yaw=pose_xy_yaw
        )
        self._instruction_state = 'reobservation'
        self._instruction_reobservation_start = (
            self._instruction_projection_count
        )
        self._instruction_stage_executor = None
        self._instruction_plan = None
        self._instruction_grid = None
        self._trace(
            event='instruction_stage_a_information_complete',
            selected_action='collect_terminal_reobservation',
            selection_reason='stage_a_semantically_verified',
            active_route_index=0,
        )

    def _begin_instruction_stage_b(self, start_xy) -> None:
        """Plan stage B from the semantically verified stage-A pose."""
        executor = self._instruction_stage_executor
        if (
            executor is None
            or self._instruction_plan is None
            or self._instruction_grid is None
        ):
            self._instruction_state = 'terminal'
            return
        waypoints = stage_waypoints(
            self._instruction_plan,
            1,
            grid=self._instruction_grid,
            start_xy=start_xy,
            spacing_m=float(self.get_parameter(
                'instruction_waypoint_spacing_m'
            ).value),
        )
        if not waypoints:
            executor.fail('stage_b_path_unavailable')
            self._instruction_state = 'terminal'
            return
        first = executor.begin_stage_b(waypoints, now=self._now())
        self._publish_waypoint(first)
        self._record_executor_events()
        self._trace(
            event='instruction_stage_b_started',
            selected_action='publish_stage_b_first_waypoint',
            selection_reason='stage_a_semantically_verified',
            active_route_index=1,
            details={'waypoint_count': len(waypoints)},
        )

    def _record_semantic_stage_events(self) -> None:
        """Copy semantic-region completion events into the decision trace."""
        executor = self._instruction_stage_executor
        if executor is None:
            return
        for event in executor.drain_events():
            self._trace(
                event='semantic_stage_complete',
                selected_action='advance_semantic_route',
                selection_reason='target_region_satisfied',
                active_route_index=event.stage_index,
                details=event.to_dict(),
            )

    def _finish_instruction_exploration(self, *, succeeded: bool) -> None:
        """Finish one bounded viewpoint and open re-observation/fallback."""
        coordinator = self._instruction_episode
        viewpoint = self._instruction_selected_viewpoint
        if (
            self._instruction_state != 'exploration_active'
            or coordinator is None
            or viewpoint is None
        ):
            return
        duration = max(
            0.0,
            self._now() - (self._instruction_motion_started_at or self._now()),
        )
        distance = viewpoint.travel_cost_m
        if succeeded and self._latest_pose_xy is not None:
            coordinator.notify_viewpoint_arrived(
                pose_xy_yaw=(
                    self._latest_pose_xy[0],
                    self._latest_pose_xy[1],
                    self._latest_heading(),
                ),
                distance_m=distance,
                duration_sec=duration,
                focus_key=viewpoint.viewpoint_id,
            )
        else:
            coordinator.notify_viewpoint_failed(
                distance_m=distance,
                duration_sec=duration,
            )
        self._instruction_state = 'reobservation'
        self._instruction_reobservation_start = (
            self._instruction_projection_count
        )
        self._trace(
            event='instruction_viewpoint_finished',
            selected_action=(
                'collect_reobservation' if succeeded
                else 'evaluate_bounded_fallback'
            ),
            selection_reason=(
                'viewpoint_arrived' if succeeded
                else 'viewpoint_execution_failed'
            ),
            details={
                'viewpoint_id': viewpoint.viewpoint_id,
                'duration_sec': duration,
                'travel_cost_m': distance,
            },
        )
        if not succeeded:
            self._advance_instruction_episode(force=True)

    def _publish_matching_object_waypoint(
        self,
        pose_xy_heading: tuple[float, float, float],
    ) -> None:
        message = Pose2D()
        message.x, message.y, message.theta = pose_xy_heading
        self._waypoint_publisher.publish(message)

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
        perception_runtime = getattr(self, '_perception_runtime', None)
        if perception_runtime is not None and not perception_runtime.close(
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
            if self._task_specification.task_type == 'object_reference':
                self._object_reference_state = 'initial_observation'
                self._object_reference_projection_count = 0
                self._object_reference_reobservation_start = 0
                self._object_reference_resolution = None
                self._object_reference_viewpoint_guard = OneViewpointGuard()
                self._object_reference_viewpoint_reason = None
                self._object_reference_selected_viewpoint = None
                self._object_reference_started_at = self._now()
                self._object_reference_fusion_events.clear()
                self._trace(
                    event='object_reference_episode_started',
                    selected_action='collect_initial_evidence',
                    selection_reason='object_reference_task_latched',
                )
            if self._task_specification.task_type == 'numerical':
                self._numerical_episode = NumericalEpisodeCoordinator(
                    stability_config=CountStabilityConfig(
                        required_consecutive_updates=int(self.get_parameter(
                            'numerical_required_consecutive_updates'
                        ).value),
                        required_independent_viewpoints=int(self.get_parameter(
                            'numerical_required_independent_viewpoints'
                        ).value),
                        minimum_count_confidence=float(self.get_parameter(
                            'numerical_minimum_count_confidence'
                        ).value),
                        maximum_unresolved_candidates=int(self.get_parameter(
                            'numerical_maximum_unresolved_candidates'
                        ).value),
                        final_commit_reserve_sec=float(self.get_parameter(
                            'numerical_final_commit_reserve_sec'
                        ).value),
                        maximum_verification_sec=float(self.get_parameter(
                            'numerical_maximum_verification_sec'
                        ).value),
                    )
                )
                self._numerical_episode.start(self._task_specification)
                self._trace(
                    event='numerical_episode_started',
                    selected_action='collect_persistent_count_evidence',
                    selection_reason='numerical_task_latched',
                )
            if self._task_specification.task_type == 'instruction_following':
                self._instruction_episode = InstructionEpisodeCoordinator(
                    candidate_config=self._reasoning_candidate_config,
                    spatial_config=self._reasoning_spatial_config,
                    vertical_config=self._relation_graph.vertical_config,
                    support_config=self._relation_graph.support_config,
                    ambiguity_config=self._reasoning_ambiguity_config,
                )
                try:
                    self._instruction_episode.start(
                        self._task_specification
                    )
                except TwoStageRouteError as error:
                    self._instruction_episode = None
                    self._instruction_state = 'unsupported'
                    self._trace(
                        event='instruction_episode_unsupported',
                        selected_action='retain_terminal_target_for_fallback',
                        selection_reason='unsupported_instruction_form',
                        details={'error': str(error)},
                    )
                else:
                    self._instruction_stage_executor = None
                    self._instruction_plan = None
                    self._instruction_grid = None
                    self._instruction_state = 'initial_observation'
                    self._instruction_projection_count = 0
                    self._instruction_reobservation_start = 0
                    self._instruction_selected_viewpoint = None
                    self._instruction_motion_started_at = None
                    self._trace(
                        event='instruction_episode_started',
                        selected_action='collect_initial_evidence',
                        selection_reason='supported_two_stage_task_latched',
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
        if not self._update_instruction_pose(
            position.x, position.y, heading
        ):
            next_goal = self._waypoint_executor.update_pose(
                position.x,
                position.y,
                heading,
                now=self._now(),
            )
            if next_goal is not None:
                self._publish_waypoint(next_goal)
            self._record_executor_events()
        if (
            self._instruction_state == 'exploration_active'
            and self._waypoint_executor.state is WaypointExecutorState.COMPLETE
        ):
            self._finish_instruction_exploration(succeeded=True)

    def _on_watchdog(self) -> None:
        if self._expire_episode_if_needed():
            return
        if (
            self._numerical_episode is not None
            and self._numerical_episode.state
            is NumericalEpisodeState.COLLECTING
            and self._episode_time_limit - self._elapsed()
            <= float(self.get_parameter(
                'numerical_final_commit_reserve_sec'
            ).value)
        ):
            self._force_numerical_commit('final_response_reserve_reached')
            return
        if (
            self._object_reference_state not in {
                'idle', 'committed', 'marker_published', 'terminal',
            }
            and self._episode_time_limit - self._elapsed()
            <= float(self.get_parameter(
                'object_reference_final_commit_reserve_sec'
            ).value)
        ):
            self._trace(
                event='object_reference_deadline_commit',
                selected_action='commit_best_available_target',
                selection_reason='final_response_reserve_reached',
            )
            self._commit_object_reference_answer(
                self._object_reference_resolution
            )
            return
        goal = self._waypoint_executor.tick(now=self._now())
        if goal is not None:
            self._publish_waypoint(goal)
        self._record_executor_events()
        if (
            self._object_reference_state == 'optional_reobservation'
            and self._waypoint_executor.state is WaypointExecutorState.FAILED
        ):
            self._trace(
                event='targeted_viewpoint_failed',
                selected_action='commit_initial_ranking',
                selection_reason='bounded_waypoint_execution_failed',
            )
            self._commit_object_reference_answer(
                self._object_reference_resolution
            )
        if (
            self._instruction_state == 'exploration_active'
            and self._waypoint_executor.state is WaypointExecutorState.FAILED
        ):
            self._finish_instruction_exploration(succeeded=False)
        semantic = self._instruction_stage_executor
        if (
            self._instruction_state == 'semantic_route_active'
            and semantic is not None
            and self._waypoint_executor.state is WaypointExecutorState.FAILED
        ):
            semantic.fail('bounded_waypoint_execution_failed')
            self._instruction_state = 'terminal'
            self._trace(
                event='instruction_route_failed',
                selected_action='terminate_failed_semantic_route',
                selection_reason='bounded_waypoint_execution_failed',
                terminal_status='failed',
            )
        if (
            self._instruction_state in {
                'initial_observation', 'reobservation'
            }
            and self._episode_time_limit - self._elapsed()
            <= float(self.get_parameter(
                'instruction_final_route_reserve_sec'
            ).value)
        ):
            self._advance_instruction_episode(force=True)

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
            image_rgb = decode_image_rgb(message)
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
            self._perception_runtime.submit(panorama)
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
        """Delegate one panorama to the perception runtime."""
        return self._perception_runtime.process_panorama(panorama)

    def _on_projection_result(
        self,
        result: ProjectionFrame | AssociationFailure,
    ) -> None:
        """Delegate one completed projection to the perception runtime."""
        self._perception_runtime.on_projection_result(result)

    def _update_persistent_maps(
        self,
        result: ProjectionFrame,
        lifting: LiftingFrame,
    ) -> None:
        """Delegate persistent object, structure, and relation updates."""
        self._perception_runtime.update_persistent_maps(result, lifting)

    def _save_projection_debug(
        self,
        result: ProjectionFrame,
        lifting: LiftingFrame,
    ) -> None:
        """Delegate bounded projection diagnostics."""
        self._perception_runtime.save_projection_debug(result, lifting)

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

    def _expire_episode_if_needed(self) -> bool:
        if self._elapsed() < self._episode_time_limit:
            return False
        self._force_numerical_commit('episode_watchdog_expired')
        hold_goal = self._waypoint_executor.expire(now=self._now())
        if hold_goal is not None:
            self._publish_waypoint(hold_goal)
        self._record_executor_events()
        if self._object_reference_state not in {
            'idle', 'committed', 'marker_published', 'terminal',
        }:
            self._commit_object_reference_answer(
                self._object_reference_resolution
            )
        return True

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
    node.configure_perception_worker(
        make_default_perception_worker(
            int(node.get_parameter('panorama_width').value),
            int(node.get_parameter('panorama_height').value),
            checkpoint=str(node.get_parameter('detector_checkpoint').value),
            confidence_threshold=float(node.get_parameter(
                'detector_confidence_threshold'
            ).value),
            cross_crop_iou_threshold=float(node.get_parameter(
                'detector_cross_crop_iou_threshold'
            ).value),
        )
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            # ros2 launch may forward SIGINT while rclpy is already cleaning
            # up. Process exit remains safe and should not emit a traceback.
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
