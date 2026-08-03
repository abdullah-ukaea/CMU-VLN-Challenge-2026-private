"""Tests for bounded nearest-depth projection visualisations."""

import numpy as np

from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.projection_visualisation import draw_projection_overlay
from qmapnav.mapping.projection_visualisation import ProjectionVisualisationConfig
from qmapnav.mapping.projection_visualisation import sparse_z_buffer_indices
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def test_sparse_z_buffer_keeps_nearest_point_per_cell() -> None:
    indices = sparse_z_buffer_indices(
        np.array([[10.0, 10.0], [11.0, 11.0], [30.0, 30.0]]),
        np.array([3.0, 1.0, 2.0]),
        image_width=100,
        image_height=50,
        cell_size_px=4,
    )

    assert indices.tolist() == [1, 2]


def test_projection_overlay_is_deterministic_and_handles_empty() -> None:
    panorama = np.zeros((120, 360, 3), dtype=np.uint8)
    result = project_map_points(
        points_map_xyz=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        transform_camera_internal_from_map=np.eye(4),
        panorama_model=PanoramaCameraModel(width=360, height=120),
        image_id='image',
        image_timestamp_ns=0,
        scan_timestamp_ns=0,
        config=ProjectionConfig(min_range_m=0.01, max_range_m=10.0),
    )
    policy = ProjectionVisualisationConfig(max_draw_points=1)

    first = draw_projection_overlay(panorama, result, config=policy)
    second = draw_projection_overlay(panorama, result, config=policy)

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) > 0
