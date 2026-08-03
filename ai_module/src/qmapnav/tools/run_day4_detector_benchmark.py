"""Run one pinned detector through the measured Day 4 benchmark and sweep."""

import argparse
from contextlib import chdir
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np

from qmapnav.evaluation import empty_metric_counts
from qmapnav.evaluation import load_detector_dataset
from qmapnav.evaluation import roll_visible_instance
from qmapnav.evaluation import score_panorama_detections
from qmapnav.perception import cross_crop_nms
from qmapnav.perception import eight_view_layout
from qmapnav.perception import GroundingDinoTinyDetector
from qmapnav.perception import panorama_box_iou
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerspectiveCropGenerator
from qmapnav.perception import project_crop_detections
from qmapnav.perception import save_debug_bundle
from qmapnav.perception import YOLOEDetector
import torch


def run_benchmark(
    manifest_path: Path,
    output_directory: Path,
    *,
    candidate_name: str,
    model_path: Path,
    thresholds: tuple[float, ...],
    nms_iou_threshold: float,
    debug_threshold: float,
) -> dict[str, object]:
    """Measure one candidate in a fresh process and save its complete report."""
    dataset = load_detector_dataset(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_cases = _load_source_cases(dataset)
    _save_ground_truth_overlays(output_directory / 'ground_truth', source_cases)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = _vram_snapshot()
    load_start = perf_counter()
    with chdir(model_path.parent):
        detector = _make_detector(candidate_name, model_path)
        torch.cuda.synchronize()
        model_load_ms = (perf_counter() - load_start) * 1000.0
        vram_after_load = _vram_snapshot()
        result = _run_thresholds(
            detector,
            dataset,
            source_cases,
            output_directory,
            thresholds,
            nms_iou_threshold,
            debug_threshold,
        )
        vram_after_benchmark = _vram_snapshot()
    peak_process_bytes = torch.cuda.max_memory_allocated()
    result.update(
        {
            'candidate': asdict(detector.identity),
            'model_load_ms': model_load_ms,
            'vram_before_load': vram_before,
            'vram_after_load': vram_after_load,
            'vram_after_benchmark': vram_after_benchmark,
            'peak_process_vram_mib': _mib(peak_process_bytes),
            'manifest': str(manifest_path),
            'source_scene_count': len({case['scene'] for case in source_cases}),
            'source_panorama_count': len(source_cases),
            'scored_panorama_count': len(source_cases) * 2,
            'crop_layout': {
                'count': 8,
                'width': 640,
                'height': 640,
                'horizontal_fov_deg': 60.0,
                'vertical_fov_deg': 90.0,
                'horizontal_overlap_fraction': 0.25,
            },
            'cross_crop_nms_iou': nms_iou_threshold,
            'cross_crop_nms_intersection_over_smaller': 0.3,
            'same_crop_nms_iou': 0.6,
            'same_crop_nms_intersection_over_smaller': 0.85,
            'matching_iou': dataset.matching_iou_threshold,
        }
    )
    report_path = output_directory / f'{detector.identity.candidate_name}.json'
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return result


def _run_thresholds(
    detector,
    dataset,
    source_cases,
    output_directory,
    thresholds,
    nms_iou_threshold,
    debug_threshold,
) -> dict[str, object]:
    panorama_model = PanoramaCameraModel(dataset.width, dataset.height)
    generator = PerspectiveCropGenerator(panorama_model, eight_view_layout())
    threshold_reports = []
    first_run_ms = None
    first_run_recorded = False
    for threshold in thresholds:
        base_metrics = empty_metric_counts()
        roll_metrics = empty_metric_counts()
        per_class_false_positives = {}
        rare_visible_classes = set()
        rare_matched_classes = set()
        seam_visible = 0
        seam_matched = 0
        latencies = []
        crop_latencies = []
        detector_latencies = []
        projection_latencies = []
        preprocess_latencies = []
        inference_latencies = []
        postprocess_latencies = []
        raw_detection_count = 0
        final_detection_count = 0
        merged_output_count = 0
        debug_variants_saved = set()
        seam_duplicate_excess = 0
        seam_case_details = []
        for source_case in source_cases:
            variants = (
                (
                    source_case['image_id'],
                    source_case['image_rgb'],
                    source_case['instances'],
                    False,
                ),
                (
                    f"{source_case['image_id']}_roll_{dataset.roll_shift_pixels}",
                    np.roll(
                        source_case['image_rgb'],
                        dataset.roll_shift_pixels,
                        axis=1,
                    ),
                    tuple(
                        roll_visible_instance(item, dataset.roll_shift_pixels)
                        for item in source_case['instances']
                    ),
                    True,
                ),
            )
            for image_id, panorama_rgb, instances, is_rolled in variants:
                torch.cuda.synchronize()
                total_start = perf_counter()
                crop_start = total_start
                views = generator.generate(panorama_rgb, source_image_id=image_id)
                crop_end = perf_counter()
                crop_detections = []
                adapter_timing = {
                    'preprocess': 0.0,
                    'inference': 0.0,
                    'postprocess': 0.0,
                }
                detector_start = crop_end
                for view in views:
                    detections = detector.detect(
                        view,
                        source_case['detector_classes'],
                        confidence_threshold=threshold,
                    )
                    crop_detections.append(detections)
                    for name, value in detector.last_timing_ms.items():
                        adapter_timing[name] += value
                torch.cuda.synchronize()
                detector_end = perf_counter()
                raw = tuple(
                    detection
                    for view, detections in zip(views, crop_detections)
                    for detection in project_crop_detections(
                        image_id,
                        view,
                        detections,
                        panorama_model,
                    )
                )
                final = cross_crop_nms(raw, iou_threshold=nms_iou_threshold)
                projection_end = perf_counter()
                total_ms = (projection_end - total_start) * 1000.0
                if not first_run_recorded:
                    first_run_ms = total_ms
                    first_run_recorded = True
                latencies.append(total_ms)
                crop_latencies.append((crop_end - crop_start) * 1000.0)
                detector_latencies.append((detector_end - detector_start) * 1000.0)
                projection_latencies.append((projection_end - detector_end) * 1000.0)
                preprocess_latencies.append(adapter_timing['preprocess'])
                inference_latencies.append(adapter_timing['inference'])
                postprocess_latencies.append(adapter_timing['postprocess'])
                metrics = score_panorama_detections(
                    instances,
                    final,
                    iou_threshold=dataset.matching_iou_threshold,
                )
                if is_rolled:
                    roll_metrics += metrics
                else:
                    base_metrics += metrics
                raw_detection_count += len(raw)
                final_detection_count += len(final)
                merged_output_count += sum(item.seam_merged for item in final)
                for class_name in {
                    item.class_name for item in (*instances, *final)
                }:
                    class_metrics = score_panorama_detections(
                        tuple(item for item in instances if item.class_name == class_name),
                        tuple(item for item in final if item.class_name == class_name),
                        iou_threshold=dataset.matching_iou_threshold,
                    )
                    per_class_false_positives[class_name] = (
                        per_class_false_positives.get(class_name, 0)
                        + class_metrics.false_positives
                    )
                    rare_truth = tuple(
                        item
                        for item in instances
                        if item.class_name == class_name and item.is_rare
                    )
                    if rare_truth:
                        rare_visible_classes.add(class_name)
                        rare_score = score_panorama_detections(
                            rare_truth,
                            tuple(item for item in final if item.class_name == class_name),
                            iou_threshold=dataset.matching_iou_threshold,
                        )
                        if rare_score.matched_instances:
                            rare_matched_classes.add(class_name)
                seam_truth = tuple(item for item in instances if item.seam_case)
                if seam_truth:
                    seam_score = score_panorama_detections(
                        seam_truth,
                        final,
                        iou_threshold=dataset.matching_iou_threshold,
                    )
                    seam_visible += seam_score.visible_instances
                    seam_matched += seam_score.matched_instances
                    for truth in seam_truth:
                        matching_predictions = tuple(
                            detection
                            for detection in final
                            if detection.class_name == truth.class_name
                            and panorama_box_iou(
                                detection.panorama_box,
                                truth.panorama_box,
                            )
                            >= dataset.matching_iou_threshold
                        )
                        seam_duplicate_excess += max(
                            0,
                            len(matching_predictions) - 1,
                        )
                        seam_case_details.append(
                            {
                                'image_id': image_id,
                                'instance_id': truth.instance_id,
                                'is_rolled': is_rolled,
                                'truth_box': _box_dict(truth.panorama_box),
                                'matching_predictions': [
                                    {
                                        'detection_id': item.detection_id,
                                        'confidence': item.confidence,
                                        'crop_ids': list(item.crop_ids),
                                        'box': _box_dict(item.panorama_box),
                                        'truth_iou': panorama_box_iou(
                                            item.panorama_box,
                                            truth.panorama_box,
                                        ),
                                    }
                                    for item in matching_predictions
                                ],
                            }
                        )
                should_save_debug = (
                    is_rolled not in debug_variants_saved
                    and source_case['image_id'] == 'office_1_spawn'
                    and np.isclose(threshold, debug_threshold)
                )
                if should_save_debug:
                    variant_name = 'rolled' if is_rolled else 'original'
                    save_debug_bundle(
                        output_directory
                        / f'debug_{variant_name}_threshold_{threshold:.2f}',
                        panorama_rgb,
                        views,
                        tuple(crop_detections),
                        raw,
                        final,
                        panorama_model,
                    )
                    debug_variants_saved.add(is_rolled)
        warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
        threshold_reports.append(
            {
                'threshold': threshold,
                'base_metrics': _metric_dict(base_metrics, len(source_cases)),
                'rolled_metrics': _metric_dict(roll_metrics, len(source_cases)),
                'rare_class_recall': (
                    len(rare_matched_classes) / len(rare_visible_classes)
                    if rare_visible_classes
                    else 0.0
                ),
                'rare_visible_classes': sorted(rare_visible_classes),
                'rare_matched_classes': sorted(rare_matched_classes),
                'false_positives_by_class': dict(
                    sorted(per_class_false_positives.items())
                ),
                'warm_median_total_ms': median(warm_latencies),
                'warm_p90_total_ms': _percentile(warm_latencies, 90),
                'median_crop_generation_ms': median(crop_latencies),
                'median_detector_total_ms': median(detector_latencies),
                'median_projection_nms_ms': median(projection_latencies),
                'median_preprocess_ms': median(preprocess_latencies),
                'median_inference_ms': median(inference_latencies),
                'median_detector_postprocess_ms': median(postprocess_latencies),
                'raw_detection_count': raw_detection_count,
                'final_detection_count': final_detection_count,
                'detections_removed_by_cross_crop_nms': (
                    raw_detection_count - final_detection_count
                ),
                'merged_output_count': merged_output_count,
                'seam_visible': seam_visible,
                'seam_matched': seam_matched,
                'seam_recall': seam_matched / seam_visible if seam_visible else 0.0,
                'seam_duplicate_excess': seam_duplicate_excess,
                'seam_case_details': seam_case_details,
            }
        )
    return {
        'first_run_total_ms': first_run_ms,
        'thresholds': threshold_reports,
    }


def _load_source_cases(dataset):
    cases = []
    for case in dataset.cases:
        bgr = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (dataset.height, dataset.width, 3):
            raise ValueError(f'failed to load benchmark panorama: {case.image_path}')
        cases.append(
            {
                'image_id': case.image_id,
                'scene': case.scene,
                'image_rgb': np.ascontiguousarray(bgr[..., ::-1]),
                'detector_classes': case.detector_classes,
                'instances': case.instances,
            }
        )
    return tuple(cases)


def _make_detector(candidate_name, model_path):
    if candidate_name == 'yoloe':
        return YOLOEDetector(model_path)
    if candidate_name == 'grounding_dino_tiny':
        return GroundingDinoTinyDetector(model_path, text_threshold=0.20)
    raise ValueError(f'unsupported candidate: {candidate_name}')


def _metric_dict(metrics, panorama_count):
    values = asdict(metrics)
    values.update(
        {
            'target_recall': metrics.target_recall,
            'anchor_recall': metrics.anchor_recall,
            'rare_instance_recall': metrics.rare_recall,
            'small_recall': metrics.small_recall,
            'false_positives_per_panorama': metrics.false_positives / panorama_count,
        }
    )
    return values


def _vram_snapshot():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        'process_allocated_mib': _mib(torch.cuda.memory_allocated()),
        'global_used_mib': _mib(total_bytes - free_bytes),
        'total_mib': _mib(total_bytes),
    }


def _mib(value):
    return value / (1024.0 * 1024.0)


def _box_dict(box):
    return {
        'x_intervals': [list(item) for item in box.x_intervals],
        'y_min': box.y_min,
        'y_max': box.y_max,
    }


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _save_ground_truth_overlays(output_directory, source_cases):
    output_directory.mkdir(parents=True, exist_ok=True)
    for case in source_cases:
        image = case['image_rgb'].copy()
        for instance in case['instances']:
            box = instance.panorama_box
            for start, end in box.x_intervals:
                cv2.rectangle(
                    image,
                    (int(round(start)), int(round(box.y_min))),
                    (min(image.shape[1] - 1, int(round(end))), int(round(box.y_max))),
                    (80, 255, 80),
                    2,
                )
            cv2.putText(
                image,
                instance.class_name,
                (
                    int(round(box.x_intervals[-1][0])),
                    max(18, int(round(box.y_min)) - 4),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 255, 80),
                1,
                cv2.LINE_AA,
            )
        path = output_directory / f"{case['image_id']}.png"
        if not cv2.imwrite(str(path), np.ascontiguousarray(image[..., ::-1])):
            raise RuntimeError(f'failed to save ground-truth overlay: {path}')


def main() -> None:
    """Parse CLI arguments and run one detector in an isolated process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest_path', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument(
        '--candidate',
        required=True,
        choices=('yoloe', 'grounding_dino_tiny'),
    )
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=(0.10, 0.20, 0.30, 0.40),
    )
    parser.add_argument('--nms-iou-threshold', type=float, default=0.4)
    parser.add_argument('--debug-threshold', type=float, default=0.2)
    arguments = parser.parse_args()
    result = run_benchmark(
        arguments.manifest_path,
        arguments.output_directory,
        candidate_name=arguments.candidate,
        model_path=arguments.model_path,
        thresholds=tuple(arguments.thresholds),
        nms_iou_threshold=arguments.nms_iou_threshold,
        debug_threshold=arguments.debug_threshold,
    )
    print(
        f"candidate={result['candidate']['candidate_name']} "
        f"scored={result['scored_panorama_count']} "
        f"thresholds={len(result['thresholds'])}"
    )


if __name__ == '__main__':
    main()
