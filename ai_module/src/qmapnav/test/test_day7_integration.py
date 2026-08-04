"""End-to-end synthetic Day 7 identity and structural-map regression."""

import numpy as np

from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.structural_map import StructuralMapConfig
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.mapping.wall_extraction import WallExtractionConfig
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _candidate(name: str, x_value: float, y_value: float) -> ObjectCandidate3D:
    centre = np.array([x_value, y_value, 0.5])
    dimensions = np.array([0.6, 0.5, 1.0])
    offsets = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0],
        [0.2, -0.2, 0.2],
        [-0.2, 0.2, -0.2],
        [0.3, 0.3, -0.3],
    ])
    points = centre + offsets * dimensions
    return ObjectCandidate3D(
        candidate_id=f'image:{name}',
        detection_id=name,
        class_name='chair',
        detection_confidence=0.9,
        source=GeometrySource.ACCUMULATED,
        source_timestamp_ns=1,
        image_timestamp_ns=1,
        scan_timestamp_ns=1,
        pose_timestamp_ns=1,
        pose_mode='interpolated',
        image_scan_delta_ms=0.0,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.arange(8),
        point_centroid_xyz=centre,
        aabb_min_xyz=centre - dimensions / 2.0,
        aabb_max_xyz=centre + dimensions / 2.0,
        obb_centre_xyz=centre,
        obb_dimensions_xyz=dimensions,
        obb_yaw_rad=0.1,
        estimated_yaw_rad=0.1,
        orientation_confidence=0.8,
        geometry_confidence=0.8,
        geometry_status=GeometryStatus.GOOD,
        partial_geometry=False,
        low_orientation_fallback=False,
        counts=LiftingCounts(10, 10, 0, 9, 8, 8, 8),
        confidence_components=ConfidenceComponents(*([0.8] * 9)),
        diagnostics={},
    )


def _observation(candidate, viewpoint, timestamp):
    return ViewpointObservation(
        viewpoint,
        np.array([0.0, 0.0, 0.0, 0.0]),
        timestamp,
        candidate.detection_id,
        candidate.point_count,
        candidate.geometry_confidence,
        'full',
    )


def _scene_points() -> np.ndarray:
    wall = [
        [x_value, 3.0, z_value]
        for x_value in np.linspace(-2.0, 2.0, 41)
        for z_value in np.linspace(0.0, 2.5, 9)
    ]
    ground = [
        [x_value, y_value, 0.0]
        for x_value in np.linspace(-2.0, 2.0, 10)
        for y_value in np.linspace(-1.0, 3.0, 10)
    ]
    return np.asarray(wall + ground)


def _window(name: str, ray: np.ndarray, viewpoint: str, timestamp: int):
    boundary = np.array([
        [10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]
    ])
    return Detection2D(
        name,
        'window',
        'window',
        0.9,
        PanoramaBox(100, 50, ((10.0, 20.0),), 10.0, 20.0, boundary),
        (0,),
        ((10.0, 10.0, 20.0, 20.0),),
        (15.0, 15.0),
        ray / np.linalg.norm(ray),
        metadata={'viewpoint_id': viewpoint, 'timestamp_ns': timestamp},
    )


def test_two_viewpoint_episode_keeps_objects_and_structure_stable() -> None:
    object_map = ObjectMap()
    first_a = _candidate('left_a', 0.0, 1.0)
    first_b = _candidate('right_a', 0.75, 1.0)
    initial_ids = object_map.add_viewpoint_candidates(
        [first_a, first_b],
        [_observation(first_a, 'view_a', 1),
         _observation(first_b, 'view_a', 1)],
    )
    second_a = _candidate('left_b', 0.02, 1.01)
    second_b = _candidate('right_b', 0.77, 0.99)
    revisit_ids = object_map.add_viewpoint_candidates(
        [second_a, second_b],
        [_observation(second_a, 'view_b', 2),
         _observation(second_b, 'view_b', 2)],
    )

    structural_map = StructuralMap(StructuralMapConfig(
        wall_extraction=WallExtractionConfig(minimum_support_points=20)
    ))
    wall_ids = structural_map.update_walls_from_points(
        _scene_points(), timestamp_ns=1, viewpoint_id='view_a'
    )
    first_pose = np.eye(4)
    first_pose[:3, 3] = [0.0, 0.0, 1.0]
    first_window = structural_map.anchor_detection_to_wall(
        _window('window_a', np.array([0.0, 1.0, 0.0]), 'view_a', 1),
        first_pose,
    )
    second_pose = np.eye(4)
    second_pose[:3, 3] = [1.0, 0.0, 1.0]
    second_window = structural_map.anchor_detection_to_wall(
        _window('window_b', np.array([-1.0, 3.0, 0.0]), 'view_b', 2),
        second_pose,
    )

    assert initial_ids == revisit_ids == [0, 1]
    assert [
        instance.observation_count
        for instance in object_map.active_instances()
    ] == [2, 2]
    assert wall_ids
    assert len(structural_map.walls()) == 1
    assert first_window.anchor_id == second_window.anchor_id
    assert first_window.supporting_wall_id == second_window.supporting_wall_id
    assert structural_map.record(first_window.anchor_id).observation_count == 2
    assert object_map.last_events[0].to_dict()['event'] == 'object_association'
    assert structural_map.last_events[0].to_dict()['event'] == (
        'structural_anchor_association'
    )
