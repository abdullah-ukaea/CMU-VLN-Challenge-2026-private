"""Tests for candidate and guarded official marker separation."""

import numpy as np
import pytest

from qmapnav.common import ObjectInstance
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectAssociationEvent
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.mapping.structural_map import StructuralAssociationEvent
from qmapnav.mission.marker_adapter import candidate_marker_array
from qmapnav.mission.marker_adapter import candidate_to_marker_spec
from qmapnav.mission.marker_adapter import FinalObjectAnswerGuard
from qmapnav.mission.marker_adapter import marker_spec_to_ros
from qmapnav.mission.marker_adapter import object_instance_to_marker_spec
from qmapnav.mission.marker_adapter import object_map_marker_array
from qmapnav.mission.marker_adapter import relation_marker_array
from qmapnav.mission.marker_adapter import structural_map_marker_array
from qmapnav.mission.marker_adapter import validate_final_marker_message
from qmapnav.mission.marker_adapter import validate_marker_spec
from qmapnav.reasoning.relation_graph import RelationGraph
from qmapnav.reasoning.support_geometry import support_geometry


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
    spec = candidate_to_marker_spec(candidate, marker_id=7, namespace='debug')
    message = marker_spec_to_ros(spec)

    assert spec.frame_id == 'map'
    assert np.allclose(spec.centre_xyz, (1.5, 2.25, 0.6))
    assert np.allclose(spec.dimensions_xyz, (1.0, 0.5, 1.0))
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


def _instance(orientation_confidence=0.8) -> ObjectInstance:
    candidate = _candidate()
    return ObjectInstance(
        7,
        {'chair': 1.0},
        {},
        candidate.obb_centre_xyz,
        candidate.aabb_min_xyz,
        candidate.aabb_max_xyz,
        candidate.obb_dimensions_xyz,
        candidate.obb_yaw_rad,
        orientation_confidence,
        3,
        0.85,
    )


def test_persistent_instance_marker_uses_fused_geometry() -> None:
    spec = object_instance_to_marker_spec(_instance(), timestamp_ns=99)

    assert np.allclose(spec.centre_xyz, (1.5, 2.25, 0.6))
    assert np.allclose(spec.dimensions_xyz, (1.0, 0.5, 1.0))
    assert validate_marker_spec(spec) == ()
    assert validate_final_marker_message(marker_spec_to_ros(spec)) == ()


def test_low_orientation_confidence_uses_conservative_aabb() -> None:
    spec = object_instance_to_marker_spec(
        _instance(orientation_confidence=0.2), timestamp_ns=99
    )

    assert np.allclose(spec.centre_xyz, (1.5, 2.25, 0.6))
    assert np.allclose(spec.dimensions_xyz, (1.0, 0.5, 1.0))
    assert spec.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_final_object_answer_guard_publishes_matching_waypoint_once() -> None:
    markers = []
    waypoints = []
    guard = FinalObjectAnswerGuard(markers.append, waypoints.append)

    answer = guard.commit(_instance(), timestamp_ns=99)

    assert guard.committed
    assert len(markers) == 1
    assert waypoints == [(1.5, 2.25, 0.0)]
    assert answer.marker.centre_xyz[:2] == answer.waypoint_xy_heading[:2]
    with pytest.raises(RuntimeError, match='already committed'):
        guard.commit(_instance(), timestamp_ns=100)


def test_fused_map_markers_include_labels_and_association_lines() -> None:
    candidate = _candidate()
    instance = ObjectInstance(
        7,
        {'chair': 1.0},
        {},
        candidate.obb_centre_xyz,
        candidate.aabb_min_xyz,
        candidate.aabb_max_xyz,
        candidate.obb_dimensions_xyz,
        candidate.obb_yaw_rad,
        0.8,
        3,
        0.85,
    )
    event = ObjectAssociationEvent(
        candidate.candidate_id,
        'chair',
        7,
        {'distance': 0.9},
        0.9,
        'merge',
        'accepted',
        3,
    )

    markers = object_map_marker_array(
        [instance],
        candidates=(candidate,),
        association_events=(event,),
    )

    namespaces = {marker.ns for marker in markers.markers}
    assert 'qmapnav_fused_objects' in namespaces
    assert 'qmapnav_object_labels' in namespaces
    assert 'qmapnav_object_associations' in namespaces


def test_structural_markers_include_wall_normal_anchor_and_label() -> None:
    wall = StructuralAnchor(
        'wall_0000',
        'wall',
        'wall',
        np.array([0.0, 2.0, 1.0]),
        np.array([[-1.0, 2.0], [1.0, 2.0]]),
        None,
        np.array([0.0, 1.0, 0.0, -2.0]),
        np.array([2.0, 0.05, 2.0]),
        0.0,
        None,
        0.9,
        1,
        1,
        ('scan',),
        (),
    )
    anchor = StructuralAnchor(
        'anchor_0000',
        'window',
        'window',
        np.array([0.0, 2.0, 1.0]),
        None,
        None,
        wall.plane_parameters,
        np.array([1.0, 0.05, 1.0]),
        0.0,
        wall.anchor_id,
        0.8,
        1,
        2,
        ('a', 'b'),
        ('window_a', 'window_b'),
    )

    event = StructuralAssociationEvent(
        'window_a', 'window', anchor.anchor_id, wall.anchor_id,
        'create_new', 'nearest_plausible_forward_wall',
        {wall.anchor_id: 2.0}, (0.0, 0.0, 1.0), (0.0, 2.0, 1.0),
    )
    markers = structural_map_marker_array([wall], [anchor], (event,))

    namespaces = {marker.ns for marker in markers.markers}
    assert 'qmapnav_walls' in namespaces
    assert 'qmapnav_wall_normals' in namespaces
    assert 'qmapnav_structural_anchors' in namespaces
    assert 'qmapnav_structural_labels' in namespaces
    assert 'qmapnav_structural_rays' in namespaces


def test_relation_markers_are_separate_debug_arrows_and_labels() -> None:
    first_candidate = _candidate()
    first = ObjectInstance(
        1, {'book': 1.0}, {}, np.array([1.5, 2.25, 1.15]),
        np.array([1.3, 2.1, 1.1]), np.array([1.7, 2.4, 1.2]),
        np.array([0.4, 0.3, 0.1]), 0.0, 0.9, 1, 0.9,
    )
    second = ObjectInstance(
        2, {'table': 1.0}, {}, np.array([1.5, 2.25, 0.55]),
        np.array([1.0, 1.75, 0.1]), np.array([2.0, 2.75, 1.1]),
        np.array([1.0, 1.0, 1.0]), 0.0, 0.9, 1, 0.9,
    )
    entities = [support_geometry(first), support_geometry(second)]
    graph = RelationGraph()
    graph.recompute(entities)

    markers = relation_marker_array(graph.edges, entities)
    namespaces = {marker.ns for marker in markers.markers}

    assert first_candidate is not None
    assert 'qmapnav_relation_arrows' in namespaces
    assert 'qmapnav_relation_labels' in namespaces
    assert all(marker.ns != 'qmapnav_selected_object'
               for marker in markers.markers)
