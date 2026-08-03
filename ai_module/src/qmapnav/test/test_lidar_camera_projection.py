"""Tests for 360-degree map-point projection and crop correspondence."""

from math import pi

import numpy as np

from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import project_result_into_crops
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.perception.panorama_projection import make_perspective_geometry
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def _model(sign: int = -1) -> PanoramaCameraModel:
    return PanoramaCameraModel(
        width=360,
        height=120,
        vertical_fov_rad=2.0 * pi / 3.0,
        u_yaw_sign=sign,
    )


def _project(points: np.ndarray, sign: int = -1):
    return project_map_points(
        points_map_xyz=points,
        transform_camera_internal_from_map=np.eye(4),
        panorama_model=_model(sign),
        image_id='synthetic',
        image_timestamp_ns=1_000_000_000,
        scan_timestamp_ns=1_010_000_000,
        config=ProjectionConfig(min_range_m=0.01, max_range_m=10.0),
    )


def test_known_directions_cover_front_left_right_behind_above_below() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )

    result = _project(points)

    np.testing.assert_allclose(
        result.panorama_uv,
        [
            [180.0, 60.0],
            [90.0, 60.0],
            [270.0, 60.0],
            [0.0, 60.0],
            [180.0, 15.0],
            [180.0, 105.0],
        ],
        atol=1e-7,
    )
    assert result.diagnostics.image_scan_delta_ms == 10.0
    assert result.diagnostics.valid_fraction == 1.0


def test_global_panorama_keeps_rear_points_and_rejects_vertical_outside_fov() -> None:
    result = _project(
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.1, 0.0, 2.0],
                [0.0, 0.0, 0.001],
            ]
        )
    )

    assert result.point_count == 1
    assert result.source_point_indices.tolist() == [0]
    assert result.forward_depth_m[0] < 0.0


def test_wrong_yaw_sign_mutation_moves_left_point_to_wrong_half() -> None:
    point_left = np.array([[0.0, 1.0, 0.0]])

    correct_u = _project(point_left, -1).panorama_uv[0, 0]
    mutated_u = _project(point_left, 1).panorama_uv[0, 0]

    assert correct_u == 90.0
    assert mutated_u == 270.0


def test_points_either_side_of_pi_wrap_to_opposite_image_edges() -> None:
    epsilon = np.deg2rad(0.25)
    points = np.array(
        [
            [-np.cos(epsilon), np.sin(epsilon), 0.0],
            [-np.cos(epsilon), -np.sin(epsilon), 0.0],
        ]
    )

    u = _project(points).panorama_uv[:, 0]

    assert min(u) < 1.0
    assert max(u) > 359.0
    circular_separation = min(abs(u[0] - u[1]), 360.0 - abs(u[0] - u[1]))
    assert np.isclose(circular_separation, 0.5)


def test_crop_projection_allows_overlap_and_rejects_rear_point() -> None:
    result = _project(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.4, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
    )
    geometries = (
        make_perspective_geometry(
            crop_id=0,
            width=100,
            height=100,
            yaw_rad=0.0,
            pitch_rad=0.0,
            horizontal_fov_rad=pi / 3.0,
            vertical_fov_rad=pi / 2.0,
        ),
        make_perspective_geometry(
            crop_id=1,
            width=100,
            height=100,
            yaw_rad=pi / 4.0,
            pitch_rad=0.0,
            horizontal_fov_rad=pi / 3.0,
            vertical_fov_rad=pi / 2.0,
        ),
    )

    crop_zero, crop_one = project_result_into_crops(result, geometries)

    assert 0 in crop_zero.source_point_indices
    assert 1 in crop_zero.source_point_indices
    assert 1 in crop_one.source_point_indices
    assert 2 not in crop_zero.source_point_indices
    assert 2 not in crop_one.source_point_indices


def test_empty_projection_is_a_valid_result() -> None:
    result = _project(np.empty((0, 3)))

    assert result.point_count == 0
    assert result.panorama_uv.shape == (0, 2)
    assert result.diagnostics.valid_fraction == 0.0
