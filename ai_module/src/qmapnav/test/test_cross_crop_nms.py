"""Tests for panorama projection and seam-aware cross-crop NMS."""

import numpy as np

from qmapnav.perception import cross_crop_nms
from qmapnav.perception import Detection2D
from qmapnav.perception import panorama_box_intersection_over_smaller
from qmapnav.perception import panorama_box_iou
from qmapnav.perception import PanoramaBox


def _box(
    intervals: tuple[tuple[float, float], ...],
    y_min: float = 20.0,
    y_max: float = 60.0,
) -> PanoramaBox:
    boundary = np.array(
        ((intervals[0][0], y_min), (intervals[0][1], y_min),
         (intervals[0][1], y_max), (intervals[0][0], y_max)),
        dtype=np.float64,
    )
    return PanoramaBox(100, 80, intervals, y_min, y_max, boundary)


def _detection(
    detection_id: str,
    crop_id: int,
    panorama_box: PanoramaBox,
    confidence: float,
    *,
    class_name: str = 'chair',
    ray: tuple[float, float, float] = (1.0, 0.0, 0.0),
    metadata=None,
) -> Detection2D:
    normalized_ray = np.asarray(ray, dtype=np.float64)
    normalized_ray /= np.linalg.norm(normalized_ray)
    return Detection2D(
        detection_id=detection_id,
        class_name=class_name,
        prompt_used=class_name,
        confidence=confidence,
        panorama_box=panorama_box,
        crop_ids=(crop_id,),
        crop_boxes_xyxy=((1.0, 2.0, 10.0, 20.0),),
        centre_panorama_uv=(50.0, 40.0),
        centre_camera_ray=normalized_ray,
        metadata=metadata or {},
    )


def test_seam_aware_iou_matches_split_intervals() -> None:
    first = _box(((0.0, 10.0), (90.0, 100.0)))
    second = _box(((0.0, 8.0), (92.0, 100.0)))

    assert panorama_box_iou(first, second) == 0.8
    assert panorama_box_intersection_over_smaller(first, second) == 1.0


def test_nested_boxes_merge_when_iou_is_low_but_containment_is_high() -> None:
    large = _detection('large', 0, _box(((10.0, 40.0),)), 0.9)
    small = _detection('small', 1, _box(((15.0, 25.0),)), 0.8)

    result = cross_crop_nms((large, small), iou_threshold=0.4)

    assert len(result) == 1


def test_same_class_overlap_from_two_crops_merges_and_preserves_sources() -> None:
    first = _detection(
        'frame:chair:0:0',
        0,
        _box(((10.0, 30.0),)),
        0.9,
        metadata={
            'mask_polygons_panorama_uv': (((10.0, 20.0), (30.0, 20.0), (20.0, 60.0)),),
            'mask_source_crop_ids': (0,),
        },
    )
    second = _detection(
        'frame:chair:1:0',
        1,
        _box(((12.0, 32.0),)),
        0.7,
        metadata={
            'mask_polygons_panorama_uv': (((12.0, 20.0), (32.0, 20.0), (22.0, 60.0)),),
            'mask_source_crop_ids': (1,),
        },
    )

    merged = cross_crop_nms((second, first), iou_threshold=0.4)

    assert len(merged) == 1
    assert merged[0].detection_id == first.detection_id
    assert merged[0].crop_ids == (0, 1)
    assert merged[0].seam_merged is True
    assert merged[0].metadata['premerge_count'] == 2
    assert len(merged[0].metadata['mask_polygons_panorama_uv']) == 2
    assert merged[0].metadata['mask_source_crop_ids'] == (0, 1)


def test_seam_crossing_duplicates_merge_without_bad_centre_average() -> None:
    first = _detection(
        'frame:chair:3:0',
        3,
        _box(((0.0, 8.0), (92.0, 100.0))),
        0.9,
        ray=(-1.0, 0.02, 0.0),
    )
    second = _detection(
        'frame:chair:4:0',
        4,
        _box(((0.0, 7.0), (93.0, 100.0))),
        0.8,
        ray=(-1.0, -0.02, 0.0),
    )

    result = cross_crop_nms((first, second), iou_threshold=0.4)[0]

    assert result.panorama_box.crosses_seam
    assert min(result.centre_panorama_uv[0], 100.0 - result.centre_panorama_uv[0]) < 1.0


def test_nearby_distinct_objects_and_different_classes_do_not_merge() -> None:
    first = _detection('first', 0, _box(((10.0, 24.0),)), 0.9)
    nearby = _detection('nearby', 1, _box(((21.0, 35.0),)), 0.8)
    other_class = _detection(
        'table',
        1,
        _box(((11.0, 25.0),)),
        0.7,
        class_name='table',
    )

    assert len(cross_crop_nms((first, nearby, other_class), iou_threshold=0.4)) == 3


def test_strong_same_crop_duplicates_are_suppressed() -> None:
    first = _detection('first', 2, _box(((10.0, 30.0),)), 0.9)
    second = _detection('second', 2, _box(((10.0, 30.0),)), 0.8)

    assert len(cross_crop_nms((first, second))) == 1


def test_nearby_same_crop_detections_remain_distinct() -> None:
    first = _detection('first', 2, _box(((10.0, 24.0),)), 0.9)
    second = _detection('second', 2, _box(((21.0, 35.0),)), 0.8)

    assert len(cross_crop_nms((first, second))) == 2
