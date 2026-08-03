"""Seam-aware panorama projection and cross-crop non-maximum suppression."""

from dataclasses import replace

import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.panorama_projection import camera_rays_to_panorama_pixels
from qmapnav.perception.panorama_projection import crop_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.panorama_projection import project_crop_box_to_panorama


def panorama_box_area(box: PanoramaBox) -> float:
    """Return the area of a wrap-aware panorama box in square pixels."""
    width = sum(end - start for start, end in box.x_intervals)
    return width * (box.y_max - box.y_min)


def panorama_box_iou(first: PanoramaBox, second: PanoramaBox) -> float:
    """Compute IoU directly over one- or two-interval horizontal support."""
    intersection = _panorama_box_intersection(first, second)
    union = panorama_box_area(first) + panorama_box_area(second) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def panorama_box_intersection_over_smaller(
    first: PanoramaBox,
    second: PanoramaBox,
) -> float:
    """Return intersection divided by the smaller box area."""
    intersection = _panorama_box_intersection(first, second)
    smaller = min(panorama_box_area(first), panorama_box_area(second))
    return 0.0 if smaller <= 0.0 else intersection / smaller


def _panorama_box_intersection(first: PanoramaBox, second: PanoramaBox) -> float:
    if (
        first.panorama_width != second.panorama_width
        or first.panorama_height != second.panorama_height
    ):
        raise ValueError('panorama boxes must use the same image dimensions')
    horizontal_intersection = sum(
        max(0.0, min(first_end, second_end) - max(first_start, second_start))
        for first_start, first_end in first.x_intervals
        for second_start, second_end in second.x_intervals
    )
    vertical_intersection = max(
        0.0,
        min(first.y_max, second.y_max) - max(first.y_min, second.y_min),
    )
    return horizontal_intersection * vertical_intersection


def project_crop_detections(
    image_id: str,
    view: PerspectiveView,
    detections: tuple[CropDetection, ...],
    panorama_model: PanoramaCameraModel,
) -> tuple[Detection2D, ...]:
    """Map one crop's normalized boxes into detector-independent panorama data."""
    mapped = []
    for index, detection in enumerate(detections):
        if detection.crop_id != view.geometry.crop_id:
            raise ValueError('detection and perspective view crop IDs differ')
        x_min, y_min, x_max, y_max = detection.bbox_xyxy
        centre_crop = np.array(
            ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
            dtype=np.float64,
        )
        centre_ray = crop_pixels_to_camera_rays(centre_crop, view.geometry)
        centre_uv, centre_valid = camera_rays_to_panorama_pixels(
            centre_ray,
            panorama_model,
        )
        if not bool(centre_valid):
            raise ValueError('detection centre lies outside the panorama span')
        panorama_box = project_crop_box_to_panorama(
            detection.bbox_xyxy,
            view.geometry,
            panorama_model,
        )
        mapped.append(
            Detection2D(
                detection_id=(
                    f'{image_id}:{detection.canonical_name}:'
                    f'{detection.crop_id}:{index}'
                ),
                class_name=detection.canonical_name,
                prompt_used=detection.prompt_used,
                confidence=detection.confidence,
                panorama_box=panorama_box,
                crop_ids=(detection.crop_id,),
                crop_boxes_xyxy=(detection.bbox_xyxy,),
                centre_panorama_uv=tuple(float(value) for value in centre_uv),
                centre_camera_ray=centre_ray,
                metadata=dict(detection.metadata),
            )
        )
    return tuple(mapped)


def cross_crop_nms(
    detections: tuple[Detection2D, ...],
    *,
    iou_threshold: float = 0.4,
    containment_threshold: float = 0.3,
    same_crop_iou_threshold: float = 0.6,
    same_crop_containment_threshold: float = 0.85,
) -> tuple[Detection2D, ...]:
    """Merge duplicate boxes with stricter suppression inside one crop."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError('iou_threshold must lie in (0, 1]')
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError('containment_threshold must lie in (0, 1]')
    if not 0.0 < same_crop_iou_threshold <= 1.0:
        raise ValueError('same_crop_iou_threshold must lie in (0, 1]')
    if not 0.0 < same_crop_containment_threshold <= 1.0:
        raise ValueError('same_crop_containment_threshold must lie in (0, 1]')
    pending = sorted(
        detections,
        key=lambda item: (-item.confidence, item.detection_id),
    )
    if not all(isinstance(item, Detection2D) for item in pending):
        raise TypeError('detections must contain Detection2D values')

    merged = []
    while pending:
        representative = pending.pop(0)
        duplicates = []
        survivors = []
        for candidate in pending:
            separate_crop = set(representative.crop_ids).isdisjoint(
                candidate.crop_ids,
            )
            overlap = panorama_box_iou(
                representative.panorama_box,
                candidate.panorama_box,
            )
            containment = panorama_box_intersection_over_smaller(
                representative.panorama_box,
                candidate.panorama_box,
            )
            overlap_threshold = (
                iou_threshold if separate_crop else same_crop_iou_threshold
            )
            containment_limit = (
                containment_threshold
                if separate_crop
                else same_crop_containment_threshold
            )
            duplicate = (
                candidate.class_name == representative.class_name
                and (
                    overlap >= overlap_threshold
                    or containment >= containment_limit
                )
            )
            if duplicate:
                duplicates.append(candidate)
            else:
                survivors.append(candidate)
        pending = survivors
        merged.append(_merge_with_representative(representative, duplicates))
    return tuple(sorted(merged, key=lambda item: item.detection_id))


def _merge_with_representative(
    representative: Detection2D,
    duplicates: list[Detection2D],
) -> Detection2D:
    if not duplicates:
        return representative
    group = [representative, *duplicates]
    crop_evidence_by_id = {}
    for detection in group:
        for crop_id, crop_box in zip(
            detection.crop_ids,
            detection.crop_boxes_xyxy,
        ):
            crop_evidence_by_id.setdefault(crop_id, crop_box)
    crop_evidence = sorted(crop_evidence_by_id.items())
    weights = np.asarray(
        [max(detection.confidence, 1e-6) for detection in group],
        dtype=np.float64,
    )
    rays = np.stack([detection.centre_camera_ray for detection in group])
    centre_ray = np.average(rays, axis=0, weights=weights)
    centre_ray /= np.linalg.norm(centre_ray)
    centre_uv = _ray_to_panorama_centre(
        centre_ray,
        representative.panorama_box,
    )
    metadata = dict(representative.metadata)
    metadata['merged_detection_ids'] = tuple(
        detection.detection_id for detection in duplicates
    )
    metadata['premerge_count'] = len(group)
    return replace(
        representative,
        crop_ids=tuple(item[0] for item in crop_evidence),
        crop_boxes_xyxy=tuple(item[1] for item in crop_evidence),
        centre_panorama_uv=centre_uv,
        centre_camera_ray=centre_ray,
        seam_merged=True,
        metadata=metadata,
    )


def _ray_to_panorama_centre(
    ray: np.ndarray,
    reference_box: PanoramaBox,
) -> tuple[float, float]:
    model = PanoramaCameraModel(
        width=reference_box.panorama_width,
        height=reference_box.panorama_height,
    )
    pixel, valid = camera_rays_to_panorama_pixels(ray, model)
    if not bool(valid):
        raise ValueError('merged centre ray lies outside the panorama')
    return tuple(float(value) for value in pixel)
