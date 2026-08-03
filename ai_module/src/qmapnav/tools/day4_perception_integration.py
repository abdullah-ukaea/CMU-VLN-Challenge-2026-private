"""Run the selected Day 4 worker from a real parsed question and panorama."""

import argparse
import json
from pathlib import Path
from time import time_ns

import cv2
import numpy as np

from qmapnav.language import parse_question
from qmapnav.perception import detector_classes_from_task_specification
from qmapnav.perception import make_day4_baseline_worker
from qmapnav.perception import PerceptionRequest
import torch


def main() -> None:
    """Parse arguments, process one keyframe, and save plain detector output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panorama_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--question', required=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('/home/docker/models/yoloe-11s-seg.pt'),
    )
    arguments = parser.parse_args()
    if arguments.repeat <= 0:
        raise ValueError('--repeat must be positive')

    bgr = cv2.imread(str(arguments.panorama_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'failed to load panorama: {arguments.panorama_path}')
    panorama_rgb = np.ascontiguousarray(bgr[..., ::-1])
    task = parse_question(arguments.question)
    detector_classes = detector_classes_from_task_specification(task)
    request = PerceptionRequest(
        image_id=arguments.panorama_path.stem,
        timestamp_ns=time_ns(),
        panorama_rgb=panorama_rgb,
        detector_classes=detector_classes,
        task_type=task.task_type,
    )
    worker = make_day4_baseline_worker(
        panorama_rgb.shape[1],
        panorama_rgb.shape[0],
        checkpoint=arguments.checkpoint,
    )
    allocated_vram_mib = []
    result = None
    for _ in range(arguments.repeat):
        result = worker.process(request)
        if torch.cuda.is_available():
            allocated_vram_mib.append(
                torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            )
    payload = {
        'question': arguments.question,
        'task_type': task.task_type,
        'parse_mode': task.parse_mode,
        'requested_classes': [
            item.canonical_name for item in detector_classes
        ],
        'detector': worker.detector_name,
        'crop_count': result.crop_count,
        'repeat_count': arguments.repeat,
        'allocated_vram_mib': allocated_vram_mib,
        'raw_detection_count': len(result.raw_detections),
        'final_detection_count': len(result.detections),
        'detections': [_detection_dict(item) for item in result.detections],
    }
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(
        f'detector={worker.detector_name} crops={result.crop_count} '
        f'raw={len(result.raw_detections)} final={len(result.detections)}'
    )


def _detection_dict(detection):
    return {
        'detection_id': detection.detection_id,
        'class_name': detection.class_name,
        'prompt_used': detection.prompt_used,
        'confidence': detection.confidence,
        'panorama_box': {
            'x_intervals': [
                list(interval)
                for interval in detection.panorama_box.x_intervals
            ],
            'y_min': detection.panorama_box.y_min,
            'y_max': detection.panorama_box.y_max,
        },
        'crop_ids': list(detection.crop_ids),
        'centre_panorama_uv': list(detection.centre_panorama_uv),
        'centre_camera_ray': detection.centre_camera_ray.tolist(),
        'seam_merged': detection.seam_merged,
    }


if __name__ == '__main__':
    main()
