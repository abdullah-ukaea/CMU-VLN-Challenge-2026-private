"""Tests for wrap-aware box and optional mask point selection."""

import numpy as np

from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.point_selection import points_inside_panorama_box
from qmapnav.mapping.point_selection import select_detection_points
from qmapnav.mapping.point_selection import SelectionMode
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _box(*, seam: bool = False) -> PanoramaBox:
    intervals = ((0.0, 20.0), (340.0, 360.0)) if seam else ((100.0, 200.0),)
    boundary = np.array(
        [[intervals[0][0], 20.0], [intervals[0][1], 20.0],
         [intervals[-1][1], 80.0], [intervals[-1][0], 80.0]]
    )
    return PanoramaBox(360, 120, intervals, 20.0, 80.0, boundary)


def _detection(box: PanoramaBox, metadata=None) -> Detection2D:
    centre_u = 0.0 if box.crosses_seam else 150.0
    return Detection2D(
        detection_id='image:chair:0',
        class_name='chair',
        prompt_used='chair',
        confidence=0.8,
        panorama_box=box,
        crop_ids=(0,),
        crop_boxes_xyxy=((1.0, 1.0, 10.0, 10.0),),
        centre_panorama_uv=(centre_u, 50.0),
        centre_camera_ray=np.array([1.0, 0.0, 0.0]),
        metadata=metadata or {},
    )


def _projection(uv: np.ndarray) -> ProjectionResult:
    count = len(uv)
    return ProjectionResult(
        image_id='image',
        image_timestamp_ns=1,
        scan_timestamp_ns=1,
        transform_camera_internal_from_map=np.eye(4),
        source_valid_mask=np.ones(count, dtype=np.bool_),
        source_point_indices=np.arange(count),
        points_map_xyz=np.column_stack((np.ones(count), np.zeros((count, 2)))),
        points_camera_xyz=np.column_stack((np.ones(count), np.zeros((count, 2)))),
        panorama_uv=uv,
        euclidean_range_m=np.ones(count),
        forward_depth_m=np.ones(count),
        intensity=None,
        diagnostics=ProjectionDiagnostics(
            count, count, count, count, 0.0, 'exact', 0.0, 0.0, False
        ),
    )


def test_box_membership_contraction_and_seam_wrap() -> None:
    uv = np.array(
        [
            [100.0, 50.0],
            [105.0, 50.0],
            [150.0, 50.0],
            [350.0, 50.0],
            [10.0, 50.0],
            [180.0, 90.0],
        ]
    )
    ordinary = points_inside_panorama_box(uv, _box())
    contracted = points_inside_panorama_box(uv, _box(), margin_fraction=0.05)
    seam = points_inside_panorama_box(uv, _box(seam=True))

    assert ordinary.tolist() == [True, True, True, False, False, False]
    assert contracted.tolist() == [False, True, True, False, False, False]
    assert seam.tolist() == [False, False, False, True, True, False]


def test_selector_uses_mask_polygon_and_invalid_mask_falls_back() -> None:
    uv = np.array([[110.0, 30.0], [150.0, 50.0], [190.0, 70.0], [250.0, 50.0]])
    polygon = ((120.0, 35.0), (180.0, 35.0), (180.0, 65.0), (120.0, 65.0))
    detection = _detection(_box(), {'mask_polygons_panorama_uv': (polygon,)})

    selected = select_detection_points(detection, _projection(uv))
    invalid = select_detection_points(
        _detection(_box()),
        _projection(uv),
        mask=np.zeros((4, 4)),
    )

    assert selected.mode is SelectionMode.MASK
    assert selected.selected_projection_indices.tolist() == [1]
    assert invalid.mode is SelectionMode.BOX_FALLBACK
    assert invalid.selected_projection_indices.tolist() == [0, 1, 2]


def test_seam_mask_polygon_selects_both_image_edges() -> None:
    polygon = ((350.0, 30.0), (10.0, 30.0), (10.0, 70.0), (350.0, 70.0))
    detection = _detection(
        _box(seam=True),
        {'mask_polygons_panorama_uv': (polygon,)},
    )
    uv = np.array([[355.0, 50.0], [5.0, 50.0], [20.0, 50.0], [340.0, 50.0]])

    result = select_detection_points(detection, _projection(uv))

    assert result.selected_projection_indices.tolist() == [0, 1]
