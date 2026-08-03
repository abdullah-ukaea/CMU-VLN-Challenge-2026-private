"""Deterministic Day 4 crop, detection, merge, and ray visualisations."""

import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.panorama_projection import crop_pixels_to_panorama_pixels
from qmapnav.perception.panorama_projection import PanoramaCameraModel


_PALETTE = (
    (255, 80, 80),
    (80, 220, 120),
    (80, 160, 255),
    (255, 200, 70),
    (210, 100, 255),
    (70, 230, 230),
)


def draw_crop_layout(
    panorama_rgb: np.ndarray,
    views: tuple[PerspectiveView, ...],
    panorama_model: PanoramaCameraModel,
) -> np.ndarray:
    """Overlay crop centres, horizontal FOV boundaries, and crop IDs."""
    canvas = np.asarray(panorama_rgb).copy()
    for view in views:
        geometry = view.geometry
        samples = np.array(
            (
                (0.0, geometry.height / 2.0),
                (geometry.width / 2.0, geometry.height / 2.0),
                (geometry.width, geometry.height / 2.0),
            )
        )
        panorama_uv, valid = crop_pixels_to_panorama_pixels(
            samples,
            geometry,
            panorama_model,
        )
        if not np.all(valid):
            continue
        color = _color_for(str(geometry.crop_id))
        for boundary_u in (panorama_uv[0, 0], panorama_uv[2, 0]):
            x = int(round(boundary_u)) % panorama_model.width
            cv2.line(canvas, (x, 0), (x, panorama_model.height - 1), color, 1)
        centre_x = int(round(panorama_uv[1, 0])) % panorama_model.width
        cv2.line(
            canvas,
            (centre_x, 0),
            (centre_x, panorama_model.height - 1),
            color,
            3,
        )
        cv2.putText(
            canvas,
            f'crop {geometry.crop_id}',
            (max(0, centre_x - 30), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_crop_detection_contact_sheet(
    views: tuple[PerspectiveView, ...],
    detections_by_crop: tuple[tuple[CropDetection, ...], ...],
    *,
    cell_width: int = 320,
) -> np.ndarray:
    """Draw every crop and its model-normalized boxes in a contact sheet."""
    if len(views) != len(detections_by_crop):
        raise ValueError('one detection tuple is required for every crop')
    rendered = []
    for view, detections in zip(views, detections_by_crop):
        image = view.image_rgb.copy()
        for detection in detections:
            _draw_crop_box(image, detection)
        cv2.putText(
            image,
            f'crop {view.geometry.crop_id}  yaw={np.degrees(view.geometry.yaw_rad):.0f}',
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        scale = cell_width / image.shape[1]
        cell_height = int(round(image.shape[0] * scale))
        rendered.append(
            cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        )
    columns = 4
    rows = (len(rendered) + columns - 1) // columns
    cell_height = rendered[0].shape[0]
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(rendered):
        row, column = divmod(index, columns)
        sheet[
            row * cell_height:(row + 1) * cell_height,
            column * cell_width:(column + 1) * cell_width,
        ] = image
    return sheet


def draw_panorama_detections(
    panorama_rgb: np.ndarray,
    detections: tuple[Detection2D, ...],
    *,
    include_sources: bool,
) -> np.ndarray:
    """Overlay wrap-aware panorama boxes before or after cross-crop NMS."""
    canvas = np.asarray(panorama_rgb).copy()
    for detection in detections:
        color = _color_for(detection.class_name)
        y_min = int(round(detection.panorama_box.y_min))
        y_max = int(round(detection.panorama_box.y_max))
        for x_min, x_max in detection.panorama_box.x_intervals:
            left = int(round(x_min))
            right = min(canvas.shape[1] - 1, int(round(x_max)))
            cv2.rectangle(canvas, (left, y_min), (right, y_max), color, 2)
        label = f'{detection.class_name} {detection.confidence:.2f}'
        if include_sources:
            label += f' crops={list(detection.crop_ids)}'
        anchor_x = int(round(detection.panorama_box.x_intervals[-1][0]))
        cv2.putText(
            canvas,
            label,
            (anchor_x, max(18, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_debug_bundle(
    output_directory: Path,
    panorama_rgb: np.ndarray,
    views: tuple[PerspectiveView, ...],
    crop_detections: tuple[tuple[CropDetection, ...], ...],
    raw_detections: tuple[Detection2D, ...],
    final_detections: tuple[Detection2D, ...],
    panorama_model: PanoramaCameraModel,
) -> tuple[Path, ...]:
    """Save the four required images plus a camera-ray JSON trace."""
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_directory / 'crop_layout.png',
        output_directory / 'crop_detections.png',
        output_directory / 'panorama_before_nms.png',
        output_directory / 'panorama_after_nms.png',
    )
    images = (
        draw_crop_layout(panorama_rgb, views, panorama_model),
        draw_crop_detection_contact_sheet(views, crop_detections),
        draw_panorama_detections(
            panorama_rgb,
            raw_detections,
            include_sources=True,
        ),
        draw_panorama_detections(
            panorama_rgb,
            final_detections,
            include_sources=True,
        ),
    )
    for path, image in zip(outputs, images):
        if not cv2.imwrite(str(path), np.ascontiguousarray(image[..., ::-1])):
            raise RuntimeError(f'failed to write debug image: {path}')
    ray_path = output_directory / 'camera_rays.json'
    ray_path.write_text(
        json.dumps(
            [
                {
                    'detection_id': item.detection_id,
                    'class_name': item.class_name,
                    'centre_panorama_uv': list(item.centre_panorama_uv),
                    'centre_camera_ray': item.centre_camera_ray.tolist(),
                    'confidence': item.confidence,
                    'crop_ids': list(item.crop_ids),
                    'panorama_box': {
                        'x_intervals': [
                            list(interval)
                            for interval in item.panorama_box.x_intervals
                        ],
                        'y_min': item.panorama_box.y_min,
                        'y_max': item.panorama_box.y_max,
                    },
                    'seam_merged': item.seam_merged,
                }
                for item in final_detections
            ],
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    return (*outputs, ray_path)


def _draw_crop_box(image: np.ndarray, detection: CropDetection) -> None:
    color = _color_for(detection.canonical_name)
    x_min, y_min, x_max, y_max = (
        int(round(value)) for value in detection.bbox_xyxy
    )
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        image,
        f'{detection.canonical_name} {detection.confidence:.2f}',
        (x_min, max(18, y_min - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _color_for(label: str) -> tuple[int, int, int]:
    index = sum(ord(character) for character in label) % len(_PALETTE)
    return _PALETTE[index]
