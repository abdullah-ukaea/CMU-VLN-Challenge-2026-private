"""Debug marker and guarded official ROS marker adapters."""

from dataclasses import dataclass
from math import cos, isfinite, sin
from threading import Lock
from typing import Callable

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectAssociationEvent
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.mapping.structural_map import StructuralAssociationEvent
from qmapnav.reasoning.relation_graph import SpatialRelation
from qmapnav.reasoning.support_geometry import SupportGeometry


CANDIDATE_MARKER_TOPIC = '/qmapnav/debug/object_candidates'
OBJECT_MAP_MARKER_TOPIC = '/qmapnav/debug/object_map'
STRUCTURAL_MAP_MARKER_TOPIC = '/qmapnav/debug/structural_map'
RELATION_MARKER_TOPIC = '/qmapnav/debug/relations'
OFFICIAL_MARKER_TOPIC = '/selected_object_marker'


@dataclass(frozen=True)
class MarkerSpec:
    """ROS-independent upright CUBE marker representation."""

    frame_id: str
    timestamp_ns: int
    namespace: str
    marker_id: int
    centre_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    dimensions_xyz: tuple[float, float, float]
    colour_rgba: tuple[float, float, float, float]
    official: bool

    def __post_init__(self) -> None:
        if not self.frame_id or not self.namespace:
            raise ValueError('marker frame and namespace must be non-empty')
        if self.timestamp_ns < 0 or self.marker_id < 0:
            raise ValueError('marker timestamp and ID must be non-negative')
        if not all(isfinite(value) for value in self.centre_xyz):
            raise ValueError('marker centre must be finite')
        if len(self.orientation_xyzw) != 4 or not all(
            isfinite(value) for value in self.orientation_xyzw
        ):
            raise ValueError('marker orientation must contain four finite values')
        if not np.isclose(np.linalg.norm(self.orientation_xyzw), 1.0, atol=1e-9):
            raise ValueError('marker orientation must be normalized')
        if len(self.dimensions_xyz) != 3 or any(
            not isfinite(value) or value <= 0.0 for value in self.dimensions_xyz
        ):
            raise ValueError('marker dimensions must be finite and positive')
        if len(self.colour_rgba) != 4 or any(
            not 0.0 <= value <= 1.0 for value in self.colour_rgba
        ):
            raise ValueError('marker colour values must lie in [0, 1]')


def candidate_to_marker_spec(
    candidate: ObjectCandidate3D,
    *,
    marker_id: int,
    namespace: str,
    minimum_dimension_m: float = 0.05,
) -> MarkerSpec:
    """Convert one debug candidate to a map-frame upright CUBE specification."""
    if not isfinite(minimum_dimension_m) or minimum_dimension_m <= 0.0:
        raise ValueError('minimum_dimension_m must be positive')
    yaw = candidate.obb_yaw_rad
    dimensions = np.maximum(candidate.obb_dimensions_xyz, minimum_dimension_m)
    confidence = candidate.geometry_confidence
    colour = (
        float(1.0 - confidence),
        float(confidence),
        float(candidate.orientation_confidence),
        0.35 + 0.45 * confidence,
    )
    return MarkerSpec(
        frame_id='map',
        timestamp_ns=candidate.source_timestamp_ns,
        namespace=namespace,
        marker_id=marker_id,
        centre_xyz=tuple(float(value) for value in candidate.obb_centre_xyz),
        orientation_xyzw=(0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)),
        dimensions_xyz=tuple(float(value) for value in dimensions),
        colour_rgba=colour,
        official=False,
    )


def object_instance_to_marker_spec(
    instance: ObjectInstance,
    *,
    timestamp_ns: int,
    marker_id: int = 0,
    namespace: str = 'qmapnav_selected_object',
    official: bool = True,
    minimum_dimension_m: float = 0.05,
    orientation_confidence_threshold: float = 0.40,
) -> MarkerSpec:
    """Convert one persistent fused instance to its final marker snapshot."""
    if not isinstance(instance, ObjectInstance):
        raise TypeError('instance must be ObjectInstance')
    if timestamp_ns < 0:
        raise ValueError('timestamp_ns must be non-negative')
    if not isfinite(minimum_dimension_m) or minimum_dimension_m <= 0.0:
        raise ValueError('minimum_dimension_m must be positive')
    if not isfinite(orientation_confidence_threshold) or not (
        0.0 <= orientation_confidence_threshold <= 1.0
    ):
        raise ValueError('orientation confidence threshold must lie in [0, 1]')
    if instance.orientation_confidence < orientation_confidence_threshold:
        yaw = 0.0
        dimensions = instance.aabb_max_xyz - instance.aabb_min_xyz
        centre = (instance.aabb_min_xyz + instance.aabb_max_xyz) / 2.0
    else:
        yaw = instance.obb_yaw
        dimensions = instance.obb_dimensions
        centre = instance.centroid_xyz
    dimensions = np.maximum(dimensions, minimum_dimension_m)
    colour = (0.0, 0.2, 1.0, 0.65) if official else (
        float(1.0 - instance.confidence),
        float(instance.confidence),
        float(instance.orientation_confidence),
        0.70,
    )
    return MarkerSpec(
        frame_id='map',
        timestamp_ns=timestamp_ns,
        namespace=namespace,
        marker_id=marker_id,
        centre_xyz=tuple(float(value) for value in centre),
        orientation_xyzw=(0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)),
        dimensions_xyz=tuple(float(value) for value in dimensions),
        colour_rgba=colour,
        official=official,
    )


def validate_marker_spec(spec: MarkerSpec) -> tuple[str, ...]:
    """Return hard protocol errors for a final pure marker specification."""
    if not isinstance(spec, MarkerSpec):
        return ('wrong_message_type',)
    errors = []
    if spec.frame_id != 'map':
        errors.append('wrong_frame')
    if not spec.official:
        errors.append('not_official')
    if not all(isfinite(value) for value in spec.centre_xyz):
        errors.append('invalid_centre')
    norm = float(np.linalg.norm(spec.orientation_xyzw))
    if not isfinite(norm) or not np.isclose(norm, 1.0, atol=1.0e-6):
        errors.append('invalid_orientation')
    if any(
        not isfinite(value) or value <= 0.0
        for value in spec.dimensions_xyz
    ):
        errors.append('invalid_dimensions')
    return tuple(errors)


def validate_final_marker_message(message) -> tuple[str, ...]:
    """Validate an official visualization_msgs/Marker at the ROS boundary."""
    try:
        from visualization_msgs.msg import Marker
    except ImportError:
        Marker = None
    errors = []
    if message is None:
        return ('missing_marker',)
    if getattr(getattr(message, 'header', None), 'frame_id', None) != 'map':
        errors.append('wrong_frame')
    if Marker is not None:
        if getattr(message, 'type', None) != Marker.CUBE:
            errors.append('wrong_marker_type')
        if getattr(message, 'action', None) != Marker.ADD:
            errors.append('wrong_marker_action')
    pose = getattr(message, 'pose', None)
    position = getattr(pose, 'position', None)
    orientation = getattr(pose, 'orientation', None)
    scale = getattr(message, 'scale', None)
    centre = tuple(
        getattr(position, axis, float('nan')) for axis in ('x', 'y', 'z')
    )
    quaternion = tuple(
        getattr(orientation, axis, float('nan'))
        for axis in ('x', 'y', 'z', 'w')
    )
    dimensions = tuple(
        getattr(scale, axis, float('nan')) for axis in ('x', 'y', 'z')
    )
    if not all(isfinite(value) for value in centre):
        errors.append('invalid_centre')
    norm = float(np.linalg.norm(quaternion))
    if not isfinite(norm) or not np.isclose(norm, 1.0, atol=1.0e-6):
        errors.append('invalid_orientation')
    if any(not isfinite(value) or value <= 0.0 for value in dimensions):
        errors.append('invalid_dimensions')
    if getattr(message, 'id', -1) < 0:
        errors.append('invalid_marker_id')
    if not str(getattr(message, 'ns', '')).strip():
        errors.append('invalid_namespace')
    return tuple(errors)


def marker_spec_to_ros(spec: MarkerSpec, *, action: int | None = None):
    """Convert a pure marker specification into visualization_msgs/Marker."""
    from visualization_msgs.msg import Marker

    message = Marker()
    message.header.frame_id = spec.frame_id
    message.header.stamp.sec = spec.timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = spec.timestamp_ns % 1_000_000_000
    message.ns = spec.namespace
    message.id = spec.marker_id
    message.type = Marker.CUBE
    message.action = Marker.ADD if action is None else action
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (
        spec.centre_xyz
    )
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = spec.orientation_xyzw
    message.scale.x, message.scale.y, message.scale.z = spec.dimensions_xyz
    (
        message.color.r,
        message.color.g,
        message.color.b,
        message.color.a,
    ) = spec.colour_rgba
    return message


def candidate_marker_array(
    candidates: tuple[ObjectCandidate3D, ...],
    *,
    namespace: str = 'qmapnav_candidates',
):
    """Create an internal MarkerArray, including deterministic stale deletion."""
    from visualization_msgs.msg import Marker
    from visualization_msgs.msg import MarkerArray

    messages = MarkerArray()
    clear = Marker()
    clear.header.frame_id = 'map'
    clear.ns = namespace
    clear.action = Marker.DELETEALL
    messages.markers.append(clear)
    for marker_id, candidate in enumerate(
        sorted(candidates, key=lambda value: value.candidate_id)
    ):
        spec = candidate_to_marker_spec(
            candidate,
            marker_id=marker_id,
            namespace=namespace,
        )
        messages.markers.append(marker_spec_to_ros(spec))
    return messages


def object_map_marker_array(
    instances: list[ObjectInstance],
    *,
    candidates: tuple[ObjectCandidate3D, ...] = (),
    association_events: tuple[ObjectAssociationEvent, ...] = (),
):
    """Create fused OBB, label, and candidate-association debug markers."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker
    from visualization_msgs.msg import MarkerArray

    output = MarkerArray()
    clear = Marker()
    clear.header.frame_id = 'map'
    clear.ns = 'qmapnav_object_map'
    clear.action = Marker.DELETEALL
    output.markers.append(clear)
    by_id = {instance.instance_id: instance for instance in instances}
    for instance in sorted(instances, key=lambda item: item.instance_id):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'qmapnav_fused_objects'
        marker.id = instance.instance_id * 3
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            map(float, instance.centroid_xyz)
        )
        marker.pose.orientation.z = sin(instance.obb_yaw / 2.0)
        marker.pose.orientation.w = cos(instance.obb_yaw / 2.0)
        marker.scale.x, marker.scale.y, marker.scale.z = map(
            float, np.maximum(instance.obb_dimensions, 0.05)
        )
        marker.color.r = 0.1
        marker.color.g = float(instance.confidence)
        marker.color.b = 1.0 - float(instance.confidence) * 0.5
        marker.color.a = 0.45
        output.markers.append(marker)
        label = Marker()
        label.header.frame_id = 'map'
        label.ns = 'qmapnav_object_labels'
        label.id = instance.instance_id * 3 + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(instance.centroid_xyz[0])
        label.pose.position.y = float(instance.centroid_xyz[1])
        label.pose.position.z = float(
            instance.centroid_xyz[2] + instance.obb_dimensions[2] / 2.0 + 0.15
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        class_name = max(
            sorted(instance.class_scores),
            key=lambda name: instance.class_scores[name],
        )
        label.text = (
            f'#{instance.instance_id} {class_name} '
            f'n={instance.observation_count} c={instance.confidence:.2f}'
        )
        output.markers.append(label)
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    line_id = 0
    for event in association_events:
        candidate = candidate_by_id.get(event.candidate_id)
        instance = by_id.get(event.matched_instance_id)
        if candidate is None or instance is None:
            continue
        line = Marker()
        line.header.frame_id = 'map'
        line.ns = 'qmapnav_object_associations'
        line.id = line_id
        line_id += 1
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.scale.x = 0.015
        line.color.r = 1.0 if event.decision != 'merge' else 0.1
        line.color.g = 0.8 if event.decision == 'merge' else 0.4
        line.color.b = 0.1
        line.color.a = 0.8
        for xyz in (candidate.obb_centre_xyz, instance.centroid_xyz):
            point = Point()
            point.x, point.y, point.z = map(float, xyz)
            line.points.append(point)
        output.markers.append(line)
    return output


def structural_map_marker_array(
    walls: list[StructuralAnchor],
    anchors: list[StructuralAnchor],
    association_events: tuple[StructuralAssociationEvent, ...] = (),
):
    """Create walls, anchors, labels, and camera-ray hit diagnostics."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker
    from visualization_msgs.msg import MarkerArray

    output = MarkerArray()
    clear = Marker()
    clear.header.frame_id = 'map'
    clear.ns = 'qmapnav_structural_map'
    clear.action = Marker.DELETEALL
    output.markers.append(clear)
    marker_id = 0
    for wall in walls:
        line = Marker()
        line.header.frame_id = 'map'
        line.ns = 'qmapnav_walls'
        line.id = marker_id
        marker_id += 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.04
        line.color.r = 0.2
        line.color.g = 0.7
        line.color.b = 1.0
        line.color.a = 0.9
        for xy in wall.line_segment_xy:
            point = Point()
            point.x, point.y = map(float, xy)
            point.z = float(wall.position_xyz[2])
            line.points.append(point)
        output.markers.append(line)
        normal = Marker()
        normal.header.frame_id = 'map'
        normal.ns = 'qmapnav_wall_normals'
        normal.id = marker_id
        marker_id += 1
        normal.type = Marker.ARROW
        normal.action = Marker.ADD
        normal.scale.x = 0.03
        normal.scale.y = 0.07
        normal.scale.z = 0.07
        normal.color.r = 0.9
        normal.color.g = 0.5
        normal.color.b = 0.1
        normal.color.a = 0.9
        start = Point()
        start.x, start.y, start.z = map(float, wall.position_xyz)
        end = Point()
        end.x = start.x + float(wall.plane_parameters[0]) * 0.4
        end.y = start.y + float(wall.plane_parameters[1]) * 0.4
        end.z = start.z
        normal.points = [start, end]
        output.markers.append(normal)
    for anchor in anchors:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'qmapnav_structural_anchors'
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            map(float, anchor.position_xyz)
        )
        marker.pose.orientation.z = sin((anchor.yaw_rad or 0.0) / 2.0)
        marker.pose.orientation.w = cos((anchor.yaw_rad or 0.0) / 2.0)
        marker.scale.x, marker.scale.y, marker.scale.z = map(
            float,
            anchor.extent_xyz
            if anchor.extent_xyz is not None else np.full(3, 0.15),
        )
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.8
        marker.color.a = 0.65
        output.markers.append(marker)
        label = Marker()
        label.header.frame_id = 'map'
        label.ns = 'qmapnav_structural_labels'
        label.id = marker_id
        marker_id += 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(anchor.position_xyz[0])
        label.pose.position.y = float(anchor.position_xyz[1])
        label.pose.position.z = float(anchor.position_xyz[2] + 0.3)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.16
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = (
            f'{anchor.anchor_id} {anchor.semantic_class} '
            f'wall={anchor.supporting_wall_id}'
        )
        output.markers.append(label)
    for event in association_events:
        if event.ray_origin_xyz is None or event.intersection_xyz is None:
            continue
        ray = Marker()
        ray.header.frame_id = 'map'
        ray.ns = 'qmapnav_structural_rays'
        ray.id = marker_id
        marker_id += 1
        ray.type = Marker.LINE_LIST
        ray.action = Marker.ADD
        ray.scale.x = 0.02
        ray.color.r = 1.0
        ray.color.g = 0.85
        ray.color.b = 0.1
        ray.color.a = 0.9
        for xyz in (event.ray_origin_xyz, event.intersection_xyz):
            point = Point()
            point.x, point.y, point.z = xyz
            ray.points.append(point)
        output.markers.append(ray)
    return output


def relation_marker_array(
    relations: tuple[SpatialRelation, ...],
    entities: list[SupportGeometry],
):
    """Create debug-only relation arrows and confidence labels."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker
    from visualization_msgs.msg import MarkerArray

    output = MarkerArray()
    clear = Marker()
    clear.header.frame_id = 'map'
    clear.ns = 'qmapnav_relations'
    clear.action = Marker.DELETEALL
    output.markers.append(clear)
    by_id = {entity.entity_id: entity for entity in entities}
    for index, relation in enumerate(relations):
        subject = by_id.get(relation.subject_id)
        anchor = by_id.get(relation.anchor_id)
        if subject is None or anchor is None:
            continue
        arrow = Marker()
        arrow.header.frame_id = 'map'
        arrow.ns = 'qmapnav_relation_arrows'
        arrow.id = index * 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.025
        arrow.scale.y = 0.06
        arrow.scale.z = 0.08
        arrow.color.r = 0.1
        arrow.color.g = float(relation.confidence)
        arrow.color.b = 1.0
        arrow.color.a = 0.85
        for xyz in (subject.centre_xyz, anchor.centre_xyz):
            point = Point()
            point.x, point.y, point.z = map(float, xyz)
            arrow.points.append(point)
        output.markers.append(arrow)
        label = Marker()
        label.header.frame_id = 'map'
        label.ns = 'qmapnav_relation_labels'
        label.id = index * 2 + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        midpoint = (subject.centre_xyz + anchor.centre_xyz) / 2.0
        label.pose.position.x, label.pose.position.y, label.pose.position.z = (
            map(float, midpoint)
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.13
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f'{relation.relation}: {relation.confidence:.2f}'
        output.markers.append(label)
    return output


@dataclass(frozen=True)
class FinalObjectAnswer:
    """One persistent marker and matching object-centre waypoint commitment."""

    marker: MarkerSpec
    waypoint_xy_heading: tuple[float, float, float]


class FinalObjectAnswerGuard:
    """Publish one fused-object marker and configured matching waypoint."""

    def __init__(
        self,
        publish_marker: Callable[[object], None],
        publish_waypoint: Callable[[tuple[float, float, float]], None],
        *,
        publish_matching_waypoint: bool = True,
        namespace: str = 'qmapnav_selected_object',
        orientation_confidence_threshold: float = 0.40,
    ) -> None:
        self._publish_marker = publish_marker
        self._publish_waypoint = publish_waypoint
        self._publish_matching_waypoint = publish_matching_waypoint
        self._namespace = namespace
        self._orientation_confidence_threshold = (
            orientation_confidence_threshold
        )
        self._committed = False
        self._lock = Lock()

    @property
    def committed(self) -> bool:
        """Return whether the logical final answer was already committed."""
        with self._lock:
            return self._committed

    def commit(
        self,
        instance: ObjectInstance,
        *,
        timestamp_ns: int,
        marker_id: int = 0,
    ) -> FinalObjectAnswer:
        """Commit one immutable persistent-object snapshot exactly once."""
        with self._lock:
            if self._committed:
                raise RuntimeError('final object answer already committed')
            spec = object_instance_to_marker_spec(
                instance,
                timestamp_ns=timestamp_ns,
                marker_id=marker_id,
                namespace=self._namespace,
                orientation_confidence_threshold=(
                    self._orientation_confidence_threshold
                ),
            )
            errors = validate_marker_spec(spec)
            if errors:
                raise ValueError(f'invalid final marker: {errors}')
            waypoint = (
                float(spec.centre_xyz[0]),
                float(spec.centre_xyz[1]),
                0.0,
            )
            self._committed = True
            self._publish_marker(marker_spec_to_ros(spec))
            if self._publish_matching_waypoint:
                self._publish_waypoint(waypoint)
            return FinalObjectAnswer(spec, waypoint)


__all__ = [
    'candidate_marker_array',
    'candidate_to_marker_spec',
    'CANDIDATE_MARKER_TOPIC',
    'FinalObjectAnswer',
    'FinalObjectAnswerGuard',
    'MarkerSpec',
    'marker_spec_to_ros',
    'object_instance_to_marker_spec',
    'OBJECT_MAP_MARKER_TOPIC',
    'OFFICIAL_MARKER_TOPIC',
    'object_map_marker_array',
    'RELATION_MARKER_TOPIC',
    'relation_marker_array',
    'STRUCTURAL_MAP_MARKER_TOPIC',
    'structural_map_marker_array',
    'validate_final_marker_message',
    'validate_marker_spec',
]
