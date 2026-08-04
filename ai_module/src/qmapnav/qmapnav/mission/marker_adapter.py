"""Separated candidate and guarded official ROS marker adapters."""

from dataclasses import dataclass
from math import cos, isfinite, sin
from threading import Lock
from typing import Callable

import numpy as np

from qmapnav.mapping.object_candidate import ObjectCandidate3D


CANDIDATE_MARKER_TOPIC = '/qmapnav/debug/object_candidates'
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
    'OFFICIAL_MARKER_TOPIC',
]
