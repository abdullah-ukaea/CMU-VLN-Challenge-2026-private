"""Tests for candidate and guarded official marker separation."""

import numpy as np
import pytest

from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mission.marker_adapter import candidate_marker_array
from qmapnav.mission.marker_adapter import candidate_to_marker_spec
from qmapnav.mission.marker_adapter import FinalMarkerGuard
from qmapnav.mission.marker_adapter import marker_spec_to_ros


def _candidate() -> ObjectCandidate3D:
    points = np.array([[1.0, 2.0, 0.1], [2.0, 2.5, 1.1], [1.5, 2.2, 0.6]])
    return ObjectCandidate3D(
        candidate_id='image:chair',
        detection_id='chair:0',
        class_name='chair',
        detection_confidence=0.8,
        source=GeometrySource.CURRENT,
        source_timestamp_ns=1_500_000_010,
        image_timestamp_ns=1_500_000_000,
        scan_timestamp_ns=1_500_000_010,
        pose_timestamp_ns=1_500_000_000,
        pose_mode='interpolated',
        image_scan_delta_ms=-0.00001,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.array([0, 1, 2]),
        point_centroid_xyz=np.median(points, axis=0),
        aabb_min_xyz=np.array([1.0, 2.0, 0.1]),
        aabb_max_xyz=np.array([2.0, 2.5, 1.1]),
        obb_centre_xyz=np.array([1.5, 2.25, 0.6]),
        obb_dimensions_xyz=np.array([1.0, 0.5, 1.0]),
        obb_yaw_rad=np.pi / 3.0,
        estimated_yaw_rad=np.pi / 3.0,
        orientation_confidence=0.8,
        geometry_confidence=0.7,
        geometry_status=GeometryStatus.GOOD,
        partial_geometry=False,
        low_orientation_fallback=False,
        counts=LiftingCounts(3, 3, 0, 3, 3, 3, 3),
        confidence_components=ConfidenceComponents(*(0.8 for _ in range(9))),
        diagnostics={},
    )


def test_marker_spec_and_ros_message_use_map_obb_contract() -> None:
    candidate = _candidate()
    spec = candidate_to_marker_spec(
        candidate, marker_id=7, namespace='debug', official=False
    )
    message = marker_spec_to_ros(spec)

    assert spec.frame_id == 'map'
    assert spec.centre_xyz == (1.5, 2.25, 0.6)
    assert spec.dimensions_xyz == (1.0, 0.5, 1.0)
    assert np.isclose(np.linalg.norm(spec.orientation_xyzw), 1.0)
    assert message.header.frame_id == 'map'
    assert message.type == message.CUBE
    assert message.action == message.ADD
    assert message.id == 7
    assert message.scale.x == 1.0


def test_candidate_array_clears_stale_markers_and_never_marks_official() -> None:
    messages = candidate_marker_array((_candidate(),))

    assert messages.markers[0].action == messages.markers[0].DELETEALL
    assert messages.markers[1].ns == 'qmapnav_candidates'


def test_final_marker_guard_publishes_once_only_when_explicitly_committed() -> None:
    published = []
    guard = FinalMarkerGuard(published.append)

    assert not guard.committed
    spec = guard.commit(_candidate())

    assert guard.committed
    assert spec.official
    assert len(published) == 1
    with pytest.raises(RuntimeError, match='already committed'):
        guard.commit(_candidate())
