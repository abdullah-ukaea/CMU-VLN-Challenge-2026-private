"""Tests for wrap-aware in-box geometric support summaries."""

import numpy as np

from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.projection_quality import ProjectionQualityConfig
from qmapnav.mapping.projection_quality import summarize_detection_projection
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def _detection(intervals, detection_id='detection') -> Detection2D:
    box = PanoramaBox(
        panorama_width=360,
        panorama_height=120,
        x_intervals=intervals,
        y_min=40.0,
        y_max=80.0,
        boundary_uv=np.array(
            [
                [intervals[0][0], 40.0],
                [intervals[0][1], 40.0],
                [intervals[-1][1], 80.0],
                [intervals[-1][0], 80.0],
            ]
        ),
    )
    return Detection2D(
        detection_id=detection_id,
        class_name='chair',
        prompt_used='chair',
        confidence=0.8,
        panorama_box=box,
        crop_ids=(0,),
        crop_boxes_xyxy=((0.0, 0.0, 10.0, 10.0),),
        centre_panorama_uv=(intervals[0][0], 60.0),
        centre_camera_ray=np.array([1.0, 0.0, 0.0]),
    )


def _projection(points: np.ndarray, delta_ms: float = 0.0):
    return project_map_points(
        points_map_xyz=points,
        transform_camera_internal_from_map=np.eye(4),
        panorama_model=PanoramaCameraModel(width=360, height=120),
        image_id='image',
        image_timestamp_ns=0,
        scan_timestamp_ns=int(delta_ms * 1_000_000),
        config=ProjectionConfig(min_range_m=0.01, max_range_m=20.0),
    )


def test_ordinary_box_returns_depth_statistics() -> None:
    points = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    summary = summarize_detection_projection(
        _detection(((170.0, 190.0),)),
        _projection(points),
        ProjectionQualityConfig(sparse_point_threshold=2),
    )

    assert summary.point_count == 3
    assert summary.depth_min_m == 2.0
    assert summary.depth_median_m == 3.0
    assert summary.depth_max_m == 4.0
    assert summary.quality == 'good'


def test_seam_box_selects_both_edges_without_middle_points() -> None:
    points = np.array(
        [
            [-1.0, 0.02, 0.0],
            [-1.0, -0.02, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    summary = summarize_detection_projection(
        _detection(((0.0, 10.0), (350.0, 360.0))),
        _projection(points),
        ProjectionQualityConfig(sparse_point_threshold=3),
    )

    assert summary.point_count == 2
    assert summary.quality == 'sparse'


def test_no_points_never_fabricates_depth_and_timing_is_recorded() -> None:
    summary = summarize_detection_projection(
        _detection(((10.0, 20.0),)),
        _projection(np.array([[1.0, 0.0, 0.0]]), delta_ms=150.0),
    )

    assert summary.quality == 'no_points'
    assert summary.depth_median_m is None
    assert 'timing_warning' in summary.warnings


def test_large_depth_spread_is_explicit() -> None:
    points = np.array([[depth, 0.0, 0.0] for depth in (1, 1, 1, 6, 6, 6)])
    summary = summarize_detection_projection(
        _detection(((170.0, 190.0),)),
        _projection(points),
        ProjectionQualityConfig(sparse_point_threshold=2, high_depth_iqr_m=1.0),
    )

    assert summary.quality == 'high_depth_spread'
