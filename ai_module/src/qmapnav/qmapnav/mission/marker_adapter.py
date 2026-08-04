"""Separated candidate and guarded official ROS marker adapters."""

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


CANDIDATE_MARKER_TOPIC = '/qmapnav/debug/object_candidates'
OBJECT_MAP_MARKER_TOPIC = '/qmapnav/debug/object_map'
STRUCTURAL_MAP_MARKER_TOPIC = '/qmapnav/debug/structural_map'
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
    official: bool,
    minimum_dimension_m: float = 0.05,
) -> MarkerSpec:
    """Convert one candidate to a map-frame upright CUBE specification."""
    if not isfinite(minimum_dimension_m) or minimum_dimension_m <= 0.0:
        raise ValueError('minimum_dimension_m must be positive')
    yaw = candidate.obb_yaw_rad
    dimensions = np.maximum(candidate.obb_dimensions_xyz, minimum_dimension_m)
    confidence = candidate.geometry_confidence
    if official:
        colour = (0.0, 0.2, 1.0, 0.65)
    else:
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
        official=official,
    )


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
            official=False,
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


class FinalMarkerGuard:
    """Allow one explicit official marker commitment per episode."""

    def __init__(
        self,
        publish: Callable[[object], None],
        *,
        namespace: str = 'qmapnav_selected_object',
    ) -> None:
        self._publish = publish
        self._namespace = namespace
        self._committed = False
        self._lock = Lock()

    @property
    def committed(self) -> bool:
        """Return whether an official answer was already published."""
        with self._lock:
            return self._committed

    def commit(self, candidate: ObjectCandidate3D, *, marker_id: int = 0) -> MarkerSpec:
        """Publish exactly one official marker from an explicitly chosen candidate."""
        with self._lock:
            if self._committed:
                raise RuntimeError('final object marker already committed')
            spec = candidate_to_marker_spec(
                candidate,
                marker_id=marker_id,
                namespace=self._namespace,
                official=True,
            )
            self._publish(marker_spec_to_ros(spec))
            self._committed = True
            return spec


__all__ = [
    'candidate_marker_array',
    'candidate_to_marker_spec',
    'CANDIDATE_MARKER_TOPIC',
    'FinalMarkerGuard',
    'MarkerSpec',
    'marker_spec_to_ros',
    'OBJECT_MAP_MARKER_TOPIC',
    'OFFICIAL_MARKER_TOPIC',
    'object_map_marker_array',
    'STRUCTURAL_MAP_MARKER_TOPIC',
    'structural_map_marker_array',
]
