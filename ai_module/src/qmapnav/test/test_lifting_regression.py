"""Source-complete Day 6 lifting regression tests."""

import json

import numpy as np

from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.lifting_regression import replay_lifting_regression_case
from qmapnav.mapping.lifting_regression import save_lifting_regression_case
from qmapnav.mapping.lifting_regression import verify_lifting_regression_checksums
from qmapnav.mapping.lifting_visualisation import draw_candidate_orthographic
from qmapnav.mapping.lifting_visualisation import draw_depth_histogram
from qmapnav.mapping.lifting_visualisation import draw_lifting_stage_overlay
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _case_inputs():
    rng = np.random.default_rng(31)
    local = rng.uniform([-0.5, -0.2, 0.12], [0.5, 0.2, 1.0], (700, 3))
    yaw = np.deg2rad(35.0)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    points = local.copy()
    points[:, :2] = local[:, :2] @ rotation.T
    points += np.array([2.2, 0.1, 0.0])
    uv = rng.normal([80.0, 40.0], [10.0, 12.0], (len(points), 2))
    projection = ProjectionResult(
        image_id='regression',
        image_timestamp_ns=500,
        scan_timestamp_ns=490,
        transform_camera_internal_from_map=np.eye(4),
        source_valid_mask=np.ones(len(points), dtype=np.bool_),
        source_point_indices=np.arange(len(points)),
        points_map_xyz=points,
        points_camera_xyz=points,
        panorama_uv=uv,
        euclidean_range_m=np.linalg.norm(points, axis=1),
        forward_depth_m=points[:, 0],
        intensity=None,
        diagnostics=ProjectionDiagnostics(
            len(points), len(points), len(points), len(points),
            0.00001, 'exact', 0.0, 0.0, False,
        ),
    )
    box = PanoramaBox(
        160,
        80,
        ((45.0, 115.0),),
        5.0,
        75.0,
        np.array([[45.0, 5.0], [115.0, 5.0], [115.0, 75.0], [45.0, 75.0]]),
    )
    detection = Detection2D(
        'regression:chair:0',
        'chair',
        'chair',
        0.92,
        box,
        (0,),
        ((10.0, 10.0, 50.0, 60.0),),
        (80.0, 40.0),
        np.array([1.0, 0.0, 0.0]),
    )
    panorama = np.full((80, 160, 3), 40, dtype=np.uint8)
    plane = GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0)
    config = ObjectLiftingConfig()
    result = ObjectLifter(config).lift(
        detection,
        projection,
        source=GeometrySource.CURRENT,
        ground_plane=plane,
        use_mask=False,
    )
    return panorama, detection, projection, plane, config, result


def test_saved_lifting_case_replays_exactly_and_detects_mutation(tmp_path) -> None:
    panorama, detection, projection, plane, config, result = _case_inputs()
    assert result.candidate is not None
    stages = draw_lifting_stage_overlay(
        panorama, projection, detection, result
    )
    histogram = draw_depth_histogram(projection, result)
    geometry = draw_candidate_orthographic(result, np.zeros(3))

    case = save_lifting_regression_case(
        tmp_path / 'case',
        category='narrow_object',
        scene_id='synthetic',
        pose_id='pose_0',
        panorama_rgb=panorama,
        detection=detection,
        projection=projection,
        ground_plane=plane,
        source=GeometrySource.CURRENT,
        use_mask=False,
        config=config,
        result=result,
        stage_overlay_rgb=stages,
        depth_histogram_rgb=histogram,
        geometry_overlay_rgb=geometry,
        notes='Synthetic deterministic rotated chair cluster.',
    )

    replay = replay_lifting_regression_case(case)
    assert replay.passed
    assert replay.centre_error_m == 0.0
    assert replay.dimension_error_m == 0.0
    assert replay.yaw_error_rad == 0.0
    assert verify_lifting_regression_checksums(case)

    manifest_path = case / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['baseline']['yaw_rad'] += 0.2
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    mutated = replay_lifting_regression_case(case)
    assert not mutated.passed
    assert not mutated.checksum_valid


def test_regression_replay_can_expose_ground_configuration_change(tmp_path) -> None:
    panorama, detection, projection, plane, config, result = _case_inputs()
    case = save_lifting_regression_case(
        tmp_path / 'case',
        category='narrow_object',
        scene_id='synthetic',
        pose_id='pose_0',
        panorama_rgb=panorama,
        detection=detection,
        projection=projection,
        ground_plane=plane,
        source=GeometrySource.CURRENT,
        use_mask=False,
        config=config,
        result=result,
        stage_overlay_rgb=draw_lifting_stage_overlay(
            panorama, projection, detection, result
        ),
        depth_histogram_rgb=draw_depth_histogram(projection, result),
        geometry_overlay_rgb=draw_candidate_orthographic(result, np.zeros(3)),
        notes='Synthetic deterministic rotated chair cluster.',
    )

    replay = replay_lifting_regression_case(
        case,
        ground_plane_override=GroundPlane(
            np.array([0.0, 0.0, 1.0]), -0.60
        ),
    )

    assert not replay.passed
