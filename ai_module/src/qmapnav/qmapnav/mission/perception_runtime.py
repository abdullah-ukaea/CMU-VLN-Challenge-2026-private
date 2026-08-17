"""Perception worker scheduling and persistent map updates."""

from dataclasses import asdict, replace
from math import atan2
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import BoundedProjectionWorker
from qmapnav.mapping import ProjectionFrame
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping.lifting_pipeline import LiftingFrame
from qmapnav.mapping.lifting_visualisation import draw_candidate_orthographic
from qmapnav.mapping.lifting_visualisation import draw_depth_histogram
from qmapnav.mapping.lifting_visualisation import draw_lifting_stage_overlay
from qmapnav.mapping.map_visualisation import draw_persistent_map_top_down
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.projection_visualisation import draw_detection_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_top_down_projection
from qmapnav.mapping.transforms import invert_transform
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.mission.marker_adapter import candidate_marker_array
from qmapnav.mission.marker_adapter import object_map_marker_array
from qmapnav.mission.marker_adapter import relation_marker_array
from qmapnav.mission.marker_adapter import structural_map_marker_array
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PerceptionRequest
from qmapnav.perception.vocabulary import detector_classes_from_task_specification
from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_types import ColourEstimate
from qmapnav.reasoning.support_geometry import support_geometry
from sensor_msgs.msg import Image


class PerceptionRuntime:
    """Own bounded perception scheduling and persistent world-model updates."""

    def __init__(
        self,
        node,
        *,
        projection_pipeline,
        lifting_pipeline,
        object_map,
        structural_map,
        relation_graph,
        colour_selection_config,
        colour_classifier_config,
        colour_prototypes,
        persistent_path_xy,
        object_reference_fusion_events,
        candidate_marker_publisher,
        object_map_marker_publisher,
        structural_map_marker_publisher,
        relation_marker_publisher,
        perception_worker=None,
    ) -> None:
        self._node = node
        self._projection_pipeline = projection_pipeline
        self._lifting_pipeline = lifting_pipeline
        self._object_map = object_map
        self._structural_map = structural_map
        self._relation_graph = relation_graph
        self._colour_selection_config = colour_selection_config
        self._colour_classifier_config = colour_classifier_config
        self._colour_prototypes = colour_prototypes
        self._persistent_path_xy = persistent_path_xy
        self._object_reference_fusion_events = object_reference_fusion_events
        self._candidate_marker_publisher = candidate_marker_publisher
        self._object_map_marker_publisher = object_map_marker_publisher
        self._structural_map_marker_publisher = structural_map_marker_publisher
        self._relation_marker_publisher = relation_marker_publisher
        self._perception_worker = perception_worker
        self._projection_lock = RLock()
        self._latest_projection_frame = None
        self._latest_lifting_frame = None
        self._saved_projection_count = 0
        self._structural_frame_count = 0
        self._projection_worker = BoundedProjectionWorker(
            self._process_panorama,
            self._on_projection_result,
            max_queue_size=int(
                self.get_parameter('projection_worker_queue_size').value
            ),
        )

    def get_parameter(self, name):
        return self._node.get_parameter(name)

    def get_logger(self):
        return self._node.get_logger()

    @property
    def _task_specification(self):
        return self._node._task_specification

    def _trace(self, *args, **kwargs):
        return self._node._trace(*args, **kwargs)

    def _advance_object_reference_episode(self):
        return self._node._advance_object_reference_episode()

    def _advance_numerical_episode(self, viewpoint_id):
        return self._node._advance_numerical_episode(viewpoint_id)

    def _advance_instruction_episode(self):
        return self._node._advance_instruction_episode()

    def set_perception_worker(self, worker) -> None:
        """Install the detector worker before image processing begins."""
        self._perception_worker = worker

    def submit(self, panorama: TimedPanorama) -> None:
        """Submit one decoded panorama to the bounded projection worker."""
        self._projection_worker.submit(panorama)

    def close(self, timeout: float) -> bool:
        """Close the projection worker within the configured bound."""
        return self._projection_worker.close(timeout)

    def stats(self):
        """Return current bounded worker statistics."""
        return self._projection_worker.stats()

    @property
    def latest_projection_frame(self):
        with self._projection_lock:
            return self._latest_projection_frame

    @property
    def latest_lifting_frame(self):
        with self._projection_lock:
            return self._latest_lifting_frame

    def process_panorama(self, panorama):
        """Process one panorama synchronously for tests and worker use."""
        return self._process_panorama(panorama)

    def on_projection_result(self, result):
        """Apply one worker result and advance episode coordinators."""
        return self._on_projection_result(result)

    def update_persistent_maps(self, result, lifting):
        """Update object, structural, and relation maps for one frame."""
        return self._update_persistent_maps(result, lifting)

    def save_projection_debug(self, result, lifting):
        """Persist bounded projection diagnostics when configured."""
        return self._save_projection_debug(result, lifting)

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
        lifting = self._node._lifting_pipeline.process(result)
        with self._projection_lock:
            self._latest_lifting_frame = lifting
        self._candidate_marker_publisher.publish(
            candidate_marker_array(lifting.candidates)
        )
        self._node._update_persistent_maps(result, lifting)
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
        self._node._save_projection_debug(result, lifting)
        self._advance_object_reference_episode()
        self._advance_numerical_episode(
            self._projection_viewpoint_id(result)
        )
        self._advance_instruction_episode()

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
            instance_ids = self._object_map.add_viewpoint_candidates(
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
            self._update_persistent_colours(
                result, lifting, candidates, detections, observations,
                instance_ids,
            )
            for event in self._object_map.last_events:
                self._object_reference_fusion_events.append(event.to_dict())
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
        relation_entities = [
            support_geometry(self._object_map.record(item.instance_id))
            for item in self._object_map.active_instances()
        ]
        relation_entities.extend(
            support_geometry(anchor)
            for anchor in self._structural_map.anchors()
            if anchor.extent_xyz is not None
        )
        self._relation_graph.recompute(relation_entities)
        self._relation_marker_publisher.publish(relation_marker_array(
            self._relation_graph.edges, relation_entities
        ))
        self._trace(
            event='relation_graph_recomputed',
            selected_action='replace_derived_relation_graph',
            selection_reason='persistent_geometry_updated',
            details={
                'revision': self._relation_graph.revision,
                'relations': [
                    {
                        'relation': edge.relation,
                        'subject_id': edge.subject_id,
                        'anchor_id': edge.anchor_id,
                        'confidence': edge.confidence,
                        'gap_m': edge.evidence.vertical_gap_m,
                        'support_overlap': (
                            edge.evidence.subject_support_overlap
                        ),
                        'geometry_confidence': (
                            edge.evidence.geometry_confidence
                        ),
                    }
                    for edge in self._relation_graph.edges
                ],
                'contradictions': self._relation_graph.contradictions,
            },
        )

    def _update_persistent_colours(
        self,
        result,
        lifting,
        candidates,
        detections,
        observations,
        instance_ids,
    ) -> None:
        """Classify each selected observation and fuse it into its stable ID."""
        projection_uv = _lifting_projection_uv(result, lifting.source)
        for candidate, observation, instance_id in zip(
            candidates, observations, instance_ids
        ):
            detection = detections.get(candidate.detection_id)
            selection = None
            if detection is not None and observation.best_crop is not None:
                mask, support_uv = _crop_colour_support(
                    result.panorama.image_rgb.shape[:2],
                    detection,
                    projection_uv[candidate.source_projection_indices],
                )
                selection = select_object_pixels(
                    observation.best_crop,
                    segmentation_mask=mask,
                    geometry_support_uv=support_uv,
                    config=self._colour_selection_config,
                )
            if selection is None:
                estimate = ColourEstimate(
                    {}, None, 0.0, 0, None, None,
                    observation.viewpoint_id,
                    observation.detection_id,
                    'no_crop',
                )
                mask_quality = 0.0
                geometry_quality = 0.0
            else:
                pixels = filter_reliable_pixels(
                    selection, self._colour_selection_config
                )
                if pixels.rgb.shape[0] == 0:
                    estimate = ColourEstimate(
                        {}, None, 0.0, 0, None, None,
                        observation.viewpoint_id,
                        observation.detection_id,
                        pixels.status,
                    )
                else:
                    features = extract_colour_features(pixels)
                    estimate = classify_colour(
                        features,
                        pixels,
                        self._colour_prototypes,
                        source_viewpoint_id=observation.viewpoint_id,
                        source_detection_id=observation.detection_id,
                        config=self._colour_classifier_config,
                    )
                mask_quality = max(
                    0.0, 1.0 - selection.contamination_score
                )
                if selection.source.startswith('segmentation_mask'):
                    mask_quality = max(mask_quality, 0.85)
                geometry_quality = (
                    candidate.geometry_confidence
                    if 'geometry_support' in selection.source else 0.65
                )
            weight = self._object_map.update_colour(
                instance_id,
                estimate,
                crop_quality=observation.best_crop_score,
                mask_quality=mask_quality,
                geometry_support=geometry_quality,
            )
            self._trace(
                event='colour_observation_fused',
                selected_action=(
                    'update_colour_evidence' if weight > 0.0
                    else 'preserve_previous_colour_evidence'
                ),
                selection_reason=estimate.status,
                details={
                    'instance_id': instance_id,
                    'detection_id': observation.detection_id,
                    'probabilities': dict(estimate.probabilities),
                    'confidence': estimate.confidence,
                    'valid_pixel_count': estimate.valid_pixel_count,
                    'observation_weight': weight,
                },
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
        self._saved_projection_count += 1


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


def _lifting_projection_uv(
    result: ProjectionFrame,
    source: GeometrySource,
) -> np.ndarray:
    """Return panorama coordinates indexed by lifted candidate support."""
    if source is GeometrySource.CURRENT:
        return result.current.panorama_uv
    if source is GeometrySource.ACCUMULATED:
        return result.accumulated.panorama_uv
    if source is GeometrySource.COMBINED:
        return np.vstack((
            result.current.panorama_uv,
            result.accumulated.panorama_uv,
        ))
    raise ValueError(f'unsupported lifting source {source!r}')


def _crop_colour_support(
    panorama_shape: tuple[int, int],
    detection: Detection2D,
    geometry_panorama_uv: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Crop the owning segmentation mask and map cluster UV into crop space."""
    height, width = panorama_shape
    y_min = max(0, int(np.floor(detection.panorama_box.y_min)))
    y_max = min(height, int(np.ceil(detection.panorama_box.y_max)))
    polygons = detection.metadata.get('mask_polygons_panorama_uv', ())
    panorama_mask = None
    if polygons:
        panorama_mask = np.zeros((height, width), dtype=np.uint8)
        valid_polygons = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float64)
            if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
                valid_polygons.append(np.rint(points).astype(np.int32))
        if valid_polygons:
            cv2.fillPoly(panorama_mask, valid_polygons, 1)
        else:
            panorama_mask = None
    mask_pieces = []
    local_support = []
    x_offset = 0
    support = np.asarray(geometry_panorama_uv, dtype=np.float64)
    for x_min, x_max in detection.panorama_box.x_intervals:
        left = max(0, int(np.floor(x_min)))
        right = min(width, int(np.ceil(x_max)))
        if right <= left:
            continue
        if panorama_mask is not None:
            mask_pieces.append(panorama_mask[y_min:y_max, left:right])
        if support.size:
            keep = (
                (support[:, 0] >= left) & (support[:, 0] < right)
                & (support[:, 1] >= y_min) & (support[:, 1] < y_max)
            )
            if np.any(keep):
                local = support[keep].copy()
                local[:, 0] = local[:, 0] - left + x_offset
                local[:, 1] -= y_min
                local_support.append(local)
        x_offset += right - left
    cropped_mask = (
        np.concatenate(mask_pieces, axis=1).astype(np.bool_)
        if mask_pieces else None
    )
    support_uv = (
        np.vstack(local_support)
        if local_support else np.empty((0, 2), dtype=np.float64)
    )
    return cropped_mask, support_uv


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


decode_image_rgb = _decode_image_rgb


__all__ = ['PerceptionRuntime', 'decode_image_rgb']
