"""Run one Day 4 detector candidate on all perspective crops of one panorama."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.evaluation import DetectorBenchmarkCase
from qmapnav.evaluation import TwoCandidateDetectorBenchmark
from qmapnav.perception import DetectorClass
from qmapnav.perception import eight_view_layout
from qmapnav.perception import GroundingDinoTinyDetector
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerspectiveCropGenerator
from qmapnav.perception import YOLOEDetector
from qmapnav.perception.panorama_projection import crop_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import project_crop_box_to_panorama


def run_smoke(
    panorama_path: Path,
    output_path: Path,
    *,
    candidate_name: str,
    model_path: Path,
    requested_classes: tuple[str, ...],
    confidence_threshold: float,
) -> dict[str, object]:
    """Run one candidate and save raw crop and transformed detections."""
    panorama_bgr = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama_bgr is None:
        raise ValueError(f'failed to read panorama: {panorama_path}')
    panorama_rgb = np.ascontiguousarray(panorama_bgr[..., ::-1])
    height, width = panorama_rgb.shape[:2]
    model = PanoramaCameraModel(width=width, height=height)
    layout = eight_view_layout()
    views = PerspectiveCropGenerator(model, layout).generate(
        panorama_rgb,
        source_image_id=panorama_path.stem,
    )
    detector_classes = tuple(
        DetectorClass(
            canonical_name=class_name,
            prompts=(class_name.replace('_', ' '),),
        )
        for class_name in requested_classes
    )
    if candidate_name == 'yoloe':
        detector = YOLOEDetector(checkpoint=model_path)
    elif candidate_name == 'grounding_dino_tiny':
        detector = GroundingDinoTinyDetector(model_name_or_path=model_path)
    else:
        raise ValueError(f'unsupported candidate: {candidate_name!r}')
    benchmark = TwoCandidateDetectorBenchmark((detector,))
    predictions = benchmark.run_case(
        DetectorBenchmarkCase(
            image_id=panorama_path.stem,
            views=views,
            detector_classes=detector_classes,
        ),
        confidence_threshold=confidence_threshold,
    )[0]

    transformed = []
    for view, crop_detections in zip(views, predictions.detections_by_crop):
        for detection in crop_detections:
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            centre_crop = np.array(
                ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                dtype=np.float64,
            )
            centre_ray = crop_pixels_to_camera_rays(
                centre_crop,
                view.geometry,
            )
            panorama_box = project_crop_box_to_panorama(
                detection.bbox_xyxy,
                view.geometry,
                model,
            )
            transformed.append(
                {
                    'crop_id': detection.crop_id,
                    'canonical_name': detection.canonical_name,
                    'prompt_used': detection.prompt_used,
                    'confidence': detection.confidence,
                    'crop_bbox_xyxy': list(detection.bbox_xyxy),
                    'panorama_x_intervals': [
                        list(interval) for interval in panorama_box.x_intervals
                    ],
                    'panorama_y_min': panorama_box.y_min,
                    'panorama_y_max': panorama_box.y_max,
                    'crosses_panorama_seam': panorama_box.crosses_seam,
                    'centre_camera_ray': centre_ray.tolist(),
                }
            )
    result = {
        'candidate': {
            'name': detector.identity.candidate_name,
            'framework': detector.identity.framework,
            'checkpoint': detector.identity.checkpoint,
            'version': detector.identity.version,
        },
        'panorama_path': str(panorama_path),
        'panorama_width': width,
        'panorama_height': height,
        'crop_count': len(views),
        'crop_horizontal_fov_deg': 60.0,
        'crop_vertical_fov_deg': 90.0,
        'horizontal_overlap_fraction': layout.horizontal_overlap_fraction,
        'requested_classes': list(requested_classes),
        'confidence_threshold': confidence_threshold,
        'raw_detection_count': predictions.detection_count,
        'detections': transformed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(output_path)
    return result


def main() -> None:
    """Parse arguments and run one raw candidate smoke benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panorama_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument(
        '--candidate',
        required=True,
        choices=('yoloe', 'grounding_dino_tiny'),
    )
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--confidence-threshold', type=float, default=0.2)
    arguments = parser.parse_args()
    result = run_smoke(
        arguments.panorama_path,
        arguments.output_path,
        candidate_name=arguments.candidate,
        model_path=arguments.model_path,
        requested_classes=tuple(arguments.classes),
        confidence_threshold=arguments.confidence_threshold,
    )
    print(
        f"candidate={result['candidate']['name']} "
        f"crops={result['crop_count']} "
        f"raw_detections={result['raw_detection_count']}"
    )


if __name__ == '__main__':
    main()
