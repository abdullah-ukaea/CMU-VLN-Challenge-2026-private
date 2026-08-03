"""Run Day 4 detections over a saved Day 5 case and draw LiDAR support."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.mapping import load_projection_regression_case
from qmapnav.mapping.lidar_camera_projection import project_result_into_crops
from qmapnav.mapping.projection_quality import summarize_detections
from qmapnav.mapping.projection_visualisation import draw_crop_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_detection_projection_overlay
from qmapnav.perception import DetectorClass
from qmapnav.perception import eight_view_layout
from qmapnav.perception import make_day4_baseline_worker
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerceptionRequest
from qmapnav.perception import PerspectiveCropGenerator


DEFAULT_CLASSES = (
    'chair',
    'table',
    'computer monitor',
    'bookshelf',
    'potted plant',
    'clock',
)


def main() -> None:
    """Load one regression case, run YOLOE, and save global/crop overlays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory', type=Path)
    parser.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES)
    parser.add_argument('--output-prefix', default='')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('/home/docker/models/yoloe-11s-seg.pt'),
    )
    arguments = parser.parse_args()
    panorama, projection, manifest = load_projection_regression_case(
        arguments.case_directory
    )
    classes = tuple(
        DetectorClass(canonical_name=name, prompts=(name,))
        for name in arguments.classes
    )
    request = PerceptionRequest(
        image_id=projection.image_id,
        timestamp_ns=projection.image_timestamp_ns,
        panorama_rgb=panorama,
        detector_classes=classes,
        task_type='object_reference',
    )
    worker = make_day4_baseline_worker(
        panorama.shape[1],
        panorama.shape[0],
        checkpoint=arguments.checkpoint,
    )
    perception = worker.process(request)
    summaries = summarize_detections(perception.detections, projection)
    overlay = draw_detection_projection_overlay(
        panorama,
        projection,
        perception.detections,
        summaries,
    )
    _save_rgb(
        arguments.case_directory
        / f'{arguments.output_prefix}detection_overlay.png',
        overlay,
    )

    model = PanoramaCameraModel(
        width=int(manifest['image']['width']),
        height=int(manifest['image']['height']),
        vertical_fov_rad=float(manifest['image']['vertical_fov_rad']),
        u_yaw_sign=int(manifest['image']['u_yaw_sign']),
    )
    crop_generator = PerspectiveCropGenerator(model, eight_view_layout())
    views = crop_generator.generate(
        panorama,
        source_image_id=projection.image_id,
    )
    crop_projections = project_result_into_crops(
        projection,
        crop_generator.geometries(),
    )
    crop_directory = arguments.case_directory / f'{arguments.output_prefix}crops'
    crop_directory.mkdir(parents=True, exist_ok=True)
    for view, crop_projection in zip(views, crop_projections):
        matching_box = _first_box_for_crop(
            perception.detections,
            crop_projection.crop_id,
        )
        crop_overlay = draw_crop_projection_overlay(
            view.image_rgb,
            crop_projection,
            bbox_xyxy=matching_box,
        )
        _save_rgb(
            crop_directory / f'crop_{crop_projection.crop_id:02d}.png',
            crop_overlay,
        )
    payload = {
        'detector': worker.detector_name,
        'requested_classes': list(arguments.classes),
        'image_scan_delta_ms': projection.diagnostics.image_scan_delta_ms,
        'pose_mode': projection.diagnostics.pose_mode,
        'raw_detection_count': len(perception.raw_detections),
        'final_detection_count': len(perception.detections),
        'detections': [
            {
                'detection_id': detection.detection_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'crop_ids': list(detection.crop_ids),
                'x_intervals': [
                    list(interval)
                    for interval in detection.panorama_box.x_intervals
                ],
                'y_min': detection.panorama_box.y_min,
                'y_max': detection.panorama_box.y_max,
                'support': _support_dict(summary),
            }
            for detection, summary in zip(perception.detections, summaries)
        ],
    }
    (
        arguments.case_directory
        / f'{arguments.output_prefix}detections.json'
    ).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(
        f'detector={worker.detector_name} '
        f'raw={len(perception.raw_detections)} '
        f'final={len(perception.detections)}'
    )


def _first_box_for_crop(detections, crop_id: int):
    for detection in detections:
        if crop_id in detection.crop_ids:
            index = detection.crop_ids.index(crop_id)
            return detection.crop_boxes_xyxy[index]
    return None


def _support_dict(summary) -> dict[str, object]:
    payload = asdict(summary)
    payload.pop('projection_indices')
    payload['warnings'] = list(payload['warnings'])
    return payload


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(image_rgb[..., ::-1])):
        raise RuntimeError(f'failed to save {path}')


if __name__ == '__main__':
    main()
