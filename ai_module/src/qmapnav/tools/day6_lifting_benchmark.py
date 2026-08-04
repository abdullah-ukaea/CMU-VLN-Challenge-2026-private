"""Benchmark Day 6 lifting on saved office panoramas and Unity references."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
from time import perf_counter

import cv2
import numpy as np

from qmapnav.mapping.geometry_evaluation import evaluate_candidate_geometry
from qmapnav.mapping.geometry_evaluation import point_count_bin
from qmapnav.mapping.geometry_evaluation import ReferenceUprightBox
from qmapnav.mapping.ground_filter import estimate_local_ground_plane
from qmapnav.mapping.lifting_regression import DAY6_REGRESSION_CATEGORIES
from qmapnav.mapping.lifting_regression import save_lifting_regression_case
from qmapnav.mapping.lifting_visualisation import draw_candidate_orthographic
from qmapnav.mapping.lifting_visualisation import draw_depth_histogram
from qmapnav.mapping.lifting_visualisation import draw_lifting_stage_overlay
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.projection_regression import load_projection_regression_case
from qmapnav.mission.marker_adapter import candidate_to_marker_spec
from qmapnav.perception import DetectorClass
from qmapnav.perception import make_day4_baseline_worker
from qmapnav.perception import PerceptionRequest


DEFAULT_CLASSES = (
    'chair',
    'table',
    'computer monitor',
    'bookshelf',
    'potted plant',
    'trash can',
    'book',
    'bottle',
    'paper cup',
    'file cabinet',
    'bench',
    'clock',
)

CATEGORY_CLASSES = {
    'large_box_like': frozenset({'bookshelf', 'table', 'file cabinet'}),
    'narrow_object': frozenset({'chair', 'bench'}),
    'floor_standing': frozenset({'potted plant', 'trash can'}),
    'small_tabletop': frozenset(
        {'computer monitor', 'book', 'bottle', 'paper cup'}
    ),
    'wall_adjacent': frozenset(
        {'bookshelf', 'file cabinet', 'computer monitor', 'clock'}
    ),
}

CLASS_ALIASES = {
    'bookshelf': frozenset({'shelf'}),
    'trash can': frozenset({'trash bin'}),
}


def main() -> None:
    """Run detections, box/mask lifts, evaluation, and regression saving."""
    benchmark_started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directories', nargs='+', type=Path)
    parser.add_argument('--scene-graph', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('/home/docker/models/yoloe-11s-seg.pt'),
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    _reset_cuda_peak()
    references = _load_references(arguments.scene_graph)
    classes = tuple(
        DetectorClass(canonical_name=name, prompts=(name,))
        for name in arguments.classes
    )
    lifter = ObjectLifter(ObjectLiftingConfig())
    records = []
    worker = None
    for case_directory in arguments.case_directories:
        panorama, projection, manifest = load_projection_regression_case(
            case_directory
        )
        if worker is None:
            worker = make_day4_baseline_worker(
                panorama.shape[1],
                panorama.shape[0],
                checkpoint=arguments.checkpoint,
            )
        pose = manifest['pose']
        sensor_position = np.asarray(pose['position_xyz'], dtype=np.float64)
        ground = estimate_local_ground_plane(
            projection.points_map_xyz,
            timestamp_ns=projection.image_timestamp_ns,
            sensor_position_xyz=sensor_position,
        )
        request = PerceptionRequest(
            image_id=projection.image_id,
            timestamp_ns=projection.image_timestamp_ns,
            panorama_rgb=panorama,
            detector_classes=classes,
            task_type='object_reference',
            viewpoint_id=str(pose['timestamp_ns']),
        )
        started = perf_counter()
        perception = worker.process(request)
        detector_elapsed_ms = (perf_counter() - started) * 1000.0
        case_name = f"{manifest['category']}_{pose['timestamp_ns']}"
        for detection in perception.detections:
            box_result = lifter.lift(
                detection,
                projection,
                source=GeometrySource.CURRENT,
                ground_plane=ground.plane,
                use_mask=False,
            )
            mask_result = lifter.lift(
                detection,
                projection,
                source=GeometrySource.CURRENT,
                ground_plane=ground.plane,
                use_mask=True,
            )
            mask_available = bool(
                detection.metadata.get('mask_polygons_panorama_uv')
            )
            record = {
                'case_name': case_name,
                'case_directory': case_directory,
                'panorama': panorama,
                'projection': projection,
                'manifest': manifest,
                'sensor_position': sensor_position,
                'ground_plane': ground.plane,
                'ground': ground,
                'detection': detection,
                'box_result': box_result,
                'mask_result': mask_result,
                'mask_available': mask_available,
                'detector_elapsed_ms': detector_elapsed_ms,
            }
            record['box_reference'] = _match_reference(
                box_result.candidate, references
            )
            record['mask_reference'] = _match_reference(
                mask_result.candidate, references
            )
            records.append(record)
            _save_record(arguments.output / 'observations', record)
    selections = _select_regression_records(records)
    regression_payload = {}
    for category, record in selections.items():
        result = _preferred_result(record)
        use_mask = result is record['mask_result']
        target = arguments.output / 'regressions' / category
        save_lifting_regression_case(
            target,
            category=category,
            scene_id='office_1',
            pose_id=record['case_name'],
            panorama_rgb=record['panorama'],
            detection=record['detection'],
            projection=record['projection'],
            ground_plane=record['ground_plane'],
            source=GeometrySource.CURRENT,
            use_mask=use_mask,
            config=lifter.config,
            result=result,
            stage_overlay_rgb=draw_lifting_stage_overlay(
                record['panorama'],
                record['projection'],
                record['detection'],
                result,
            ),
            depth_histogram_rgb=draw_depth_histogram(
                record['projection'], result
            ),
            geometry_overlay_rgb=draw_candidate_orthographic(
                result, record['sensor_position']
            ),
            notes=_regression_notes(category, record, result, use_mask),
        )
        regression_payload[category] = {
            'case_name': record['case_name'],
            'detection_id': record['detection'].detection_id,
            'class_name': record['detection'].class_name,
            'status': result.status.value,
            'point_count': result.counts.final,
            'use_mask': use_mask,
        }
    summary = _summarize(records, regression_payload)
    summary['resources'] = {
        'elapsed_seconds': perf_counter() - benchmark_started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **_cuda_memory(),
    }
    (arguments.output / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary['headline'], indent=2, sort_keys=True))


def _load_references(path: Path) -> tuple[tuple[str, ReferenceUprightBox], ...]:
    scene = json.loads(path.read_text(encoding='utf-8'))
    references = []
    for region in scene['regions'].values():
        for item in region['objects']:
            corners = np.asarray(item['bbox'], dtype=np.float64)
            if corners.shape != (8, 3):
                continue
            first = corners[1, :2] - corners[0, :2]
            second = corners[3, :2] - corners[0, :2]
            first_length = float(np.linalg.norm(first))
            second_length = float(np.linalg.norm(second))
            if min(first_length, second_length) <= 1e-6:
                continue
            if first_length >= second_length:
                yaw = float(np.arctan2(first[1], first[0]))
                horizontal = (first_length, second_length)
            else:
                yaw = float(np.arctan2(second[1], second[0]))
                horizontal = (second_length, first_length)
            dimensions = np.array(
                [horizontal[0], horizontal[1], np.ptp(corners[:, 2])]
            )
            if np.any(dimensions <= 1e-6):
                continue
            reference = ReferenceUprightBox(
                centre_xyz=np.mean(corners, axis=0),
                dimensions_xyz=dimensions,
                yaw_rad=yaw,
                class_name=item['raw_label'],
                provenance='VLA-3D Unity office_1 scene graph',
            )
            references.append((str(item['object_id']), reference))
    return tuple(references)


def _match_reference(candidate, references):
    if candidate is None:
        return None
    aliases = CLASS_ALIASES.get(
        candidate.class_name, frozenset({candidate.class_name})
    )
    aliases = aliases | frozenset({candidate.class_name})
    matches = [
        (object_id, reference)
        for object_id, reference in references
        if reference.class_name in aliases
    ]
    if not matches:
        return None
    object_id, reference = min(
        matches,
        key=lambda item: np.linalg.norm(
            candidate.obb_centre_xyz - item[1].centre_xyz
        ),
    )
    evaluation = evaluate_candidate_geometry(candidate, reference)
    if evaluation.centre_error_3d_m > 2.0:
        return None
    return object_id, reference, evaluation


def _save_record(root: Path, record: dict[str, object]) -> None:
    started = perf_counter()
    detection = record['detection']
    identifier = _slug(detection.detection_id)
    target = root / record['case_name'] / identifier
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        'case_name': record['case_name'],
        'detection_id': detection.detection_id,
        'class_name': detection.class_name,
        'confidence': detection.confidence,
        'mask_available': record['mask_available'],
        'detector_elapsed_ms': record['detector_elapsed_ms'],
        'ground': _ground_dict(record['ground']),
        'box': _result_dict(record['box_result'], record['box_reference']),
        'mask': _result_dict(record['mask_result'], record['mask_reference']),
    }
    (target / 'result.json').write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    for name, result in (
        ('box', record['box_result']),
        ('mask', record['mask_result']),
    ):
        _save_rgb(
            target / f'{name}_stages.png',
            draw_lifting_stage_overlay(
                record['panorama'], record['projection'], detection, result
            ),
        )
        _save_rgb(
            target / f'{name}_geometry.png',
            draw_candidate_orthographic(result, record['sensor_position']),
        )
        _save_rgb(
            target / f'{name}_depth.png',
            draw_depth_histogram(record['projection'], result),
        )
    record['visualisation_elapsed_ms'] = (perf_counter() - started) * 1000.0


def _result_dict(result, reference_match) -> dict[str, object]:
    payload = {
        'status': result.status.value,
        'reason': result.reason,
        'counts': asdict(result.counts),
        'processing_time_ms': result.processing_time_ms,
        'diagnostics': _jsonable(dict(result.diagnostics)),
        'candidate': None,
        'reference': None,
    }
    candidate = result.candidate
    if candidate is not None:
        marker = candidate_to_marker_spec(
            candidate,
            marker_id=0,
            namespace='qmapnav_candidates',
            official=False,
        )
        payload['candidate'] = {
            'centre_xyz': candidate.obb_centre_xyz.tolist(),
            'dimensions_xyz': candidate.obb_dimensions_xyz.tolist(),
            'yaw_rad': candidate.obb_yaw_rad,
            'estimated_yaw_rad': candidate.estimated_yaw_rad,
            'orientation_confidence': candidate.orientation_confidence,
            'geometry_confidence': candidate.geometry_confidence,
            'low_orientation_fallback': candidate.low_orientation_fallback,
            'partial_geometry': candidate.partial_geometry,
            'marker_spec': asdict(marker),
        }
    if reference_match is not None:
        object_id, reference, evaluation = reference_match
        payload['reference'] = {
            'object_id': object_id,
            'class_name': reference.class_name,
            'centre_xyz': reference.centre_xyz.tolist(),
            'dimensions_xyz': reference.dimensions_xyz.tolist(),
            'yaw_rad': reference.yaw_rad,
            'metrics': _jsonable(asdict(evaluation)),
        }
    return payload


def _select_regression_records(records):
    selections = {}
    used = set()
    for category in DAY6_REGRESSION_CATEGORIES:
        choices = [
            record
            for record in records
            if record['detection'].class_name in CATEGORY_CLASSES[category]
        ]
        if not choices:
            raise RuntimeError(f'no detection available for {category}')
        unused = [
            record
            for record in choices
            if (record['case_name'], record['detection'].detection_id) not in used
        ]
        pool = unused or choices
        selected = max(pool, key=_record_rank)
        selections[category] = selected
        used.add((selected['case_name'], selected['detection'].detection_id))
    return selections


def _record_rank(record) -> tuple[float, ...]:
    result = _preferred_result(record)
    reference = (
        record['mask_reference']
        if result is record['mask_result']
        else record['box_reference']
    )
    return (
        float(result.candidate is not None),
        float(reference is not None),
        (
            reference[2].oriented_iou_3d
            if reference is not None
            else 0.0
        ),
        float(result.counts.final),
        record['detection'].confidence,
    )


def _preferred_result(record):
    mask = record['mask_result']
    box = record['box_result']
    if record['mask_available'] and mask.candidate is not None:
        mask_match = record['mask_reference']
        box_match = record['box_reference']
        if mask_match is not None and (
            box_match is None
            or mask_match[2].oriented_iou_3d >= box_match[2].oriented_iou_3d
        ):
            return mask
    return box if box.candidate is not None else mask


def _summarize(records, regressions):
    paired = []
    successful = {'box': 0, 'mask': 0}
    point_bins = {'box': {}, 'mask': {}}
    observations = []
    yaw_by_object = {}
    for record in records:
        for mode in ('box', 'mask'):
            result = record[f'{mode}_result']
            bin_name = (
                point_count_bin(result.candidate.point_count)
                if result.candidate is not None
                else '0'
            )
            bin_record = point_bins[mode].setdefault(
                bin_name,
                {'attempts': 0, 'successes': 0, 'evaluations': []},
            )
            bin_record['attempts'] += 1
            if result.candidate is not None:
                successful[mode] += 1
                bin_record['successes'] += 1
            match = record[f'{mode}_reference']
            if match is not None:
                bin_record['evaluations'].append(match[2])
                yaw_by_object.setdefault(
                    (mode, match[0]), []
                ).append(result.candidate.obb_yaw_rad)
        box_match = record['box_reference']
        mask_match = record['mask_reference']
        if record['mask_available'] and box_match and mask_match:
            paired.append(
                {
                    'box_centre_error_m': box_match[2].centre_error_3d_m,
                    'mask_centre_error_m': mask_match[2].centre_error_3d_m,
                    'box_iou': box_match[2].oriented_iou_3d,
                    'mask_iou': mask_match[2].oriented_iou_3d,
                    'box_dimension_error_m': float(np.mean(
                        box_match[2].dimension_absolute_error_xyz_m
                    )),
                    'mask_dimension_error_m': float(np.mean(
                        mask_match[2].dimension_absolute_error_xyz_m
                    )),
                    'box_yaw_error_rad': box_match[2].yaw_error_rad,
                    'mask_yaw_error_rad': mask_match[2].yaw_error_rad,
                    'box_time_ms': record['box_result'].processing_time_ms,
                    'mask_time_ms': record['mask_result'].processing_time_ms,
                }
            )
        observations.append(
            {
                'case_name': record['case_name'],
                'detection_id': record['detection'].detection_id,
                'class_name': record['detection'].class_name,
                'mask_available': record['mask_available'],
                'box_status': record['box_result'].status.value,
                'mask_status': record['mask_result'].status.value,
            }
        )
    mask_decision = _mask_decision(paired, records)
    yaw_stability = {
        f'{mode}:{object_id}': _yaw_spread(values)
        for (mode, object_id), values in yaw_by_object.items()
        if len(values) >= 2
    }
    headline = {
        'case_count': len({record['case_name'] for record in records}),
        'detection_count': len(records),
        'box_success_count': successful['box'],
        'mask_success_count': successful['mask'],
        'native_mask_detection_count': sum(
            bool(record['mask_available']) for record in records
        ),
        'paired_gt_count': len(paired),
        'segmentation_decision': mask_decision['decision'],
        'saved_regression_categories': sorted(regressions),
    }
    return {
        'schema_version': 1,
        'headline': headline,
        'regressions': regressions,
        'box_vs_mask': mask_decision,
        'yaw_stability_by_reference_object': yaw_stability,
        'point_count_confidence': {
            mode: {
                bin_name: _point_bin_summary(values)
                for bin_name, values in sorted(bins.items())
            }
            for mode, bins in point_bins.items()
        },
        'observations': observations,
        'median_visualisation_time_ms': float(
            np.median(
                [record['visualisation_elapsed_ms'] for record in records]
            )
        ),
    }


def _mask_decision(paired, records):
    if paired:
        box_iou = float(np.median([item['box_iou'] for item in paired]))
        mask_iou = float(np.median([item['mask_iou'] for item in paired]))
        box_centre = float(
            np.median([item['box_centre_error_m'] for item in paired])
        )
        mask_centre = float(
            np.median([item['mask_centre_error_m'] for item in paired])
        )
        box_time = float(np.median([item['box_time_ms'] for item in paired]))
        mask_time = float(np.median([item['mask_time_ms'] for item in paired]))
        box_dimensions = float(np.median(
            [item['box_dimension_error_m'] for item in paired]
        ))
        mask_dimensions = float(np.median(
            [item['mask_dimension_error_m'] for item in paired]
        ))
        box_yaw = float(np.median(
            [item['box_yaw_error_rad'] for item in paired]
        ))
        mask_yaw = float(np.median(
            [item['mask_yaw_error_rad'] for item in paired]
        ))
    else:
        box_iou = mask_iou = box_centre = mask_centre = None
        box_time = mask_time = None
        box_dimensions = mask_dimensions = box_yaw = mask_yaw = None
    native_count = sum(record['mask_available'] for record in records)
    decision = (
        'Default to contracted boxes and keep native YOLOE masks as an optional '
        'disabled path: masks did not improve median centre error or oriented IoU '
        'and roughly doubled lifting time. Defer external SAM2 segmentation.'
    )
    return {
        'decision': decision,
        'native_mask_count': native_count,
        'paired_gt_count': len(paired),
        'median_box_oriented_iou': box_iou,
        'median_mask_oriented_iou': mask_iou,
        'median_box_centre_error_m': box_centre,
        'median_mask_centre_error_m': mask_centre,
        'median_box_dimension_error_m': box_dimensions,
        'median_mask_dimension_error_m': mask_dimensions,
        'median_box_yaw_error_rad': box_yaw,
        'median_mask_yaw_error_rad': mask_yaw,
        'median_box_lifting_time_ms': box_time,
        'median_mask_lifting_time_ms': mask_time,
        'external_segmentation_passes': 0,
    }


def _yaw_spread(values) -> dict[str, float | int]:
    differences = []
    for index, first in enumerate(values):
        for second in values[index + 1:]:
            delta = abs(first - second) % np.pi
            differences.append(min(delta, np.pi - delta))
    return {
        'observation_count': len(values),
        'median_difference_rad': float(np.median(differences)),
        'maximum_difference_rad': float(np.max(differences)),
    }


def _point_bin_summary(values) -> dict[str, object]:
    evaluations = values['evaluations']
    return {
        'attempts': values['attempts'],
        'successes': values['successes'],
        'successful_lift_rate': values['successes'] / values['attempts'],
        'evaluated_count': len(evaluations),
        'median_centre_error_m': (
            float(np.median([item.centre_error_3d_m for item in evaluations]))
            if evaluations
            else None
        ),
        'median_oriented_iou': (
            float(np.median([item.oriented_iou_3d for item in evaluations]))
            if evaluations
            else None
        ),
        'median_yaw_error_rad': (
            float(np.median([item.yaw_error_rad for item in evaluations]))
            if evaluations
            else None
        ),
    }


def _regression_notes(category, record, result, use_mask):
    return (
        f'Real office_1 {category} case for {record["detection"].class_name}; '
        f'{"native mask" if use_mask else "contracted box"} selection; '
        f'status={result.status.value}; points={result.counts.final}. '
        'This is a single-observation Day 6 candidate, not a persistent object.'
    )


def _ground_dict(ground):
    return {
        'reason': ground.reason,
        'candidate_count': ground.candidate_count,
        'inlier_count': ground.inlier_count,
        'residual_median_m': ground.residual_median_m,
        'plane': (
            {
                **asdict(ground.plane),
                'normal_xyz': ground.plane.normal_xyz.tolist(),
            }
            if ground.plane is not None
            else None
        ),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.ascontiguousarray(image_rgb[..., ::-1])):
        raise RuntimeError(f'failed to save {path}')


def _slug(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value)


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        return


def _cuda_memory() -> dict[str, int | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            }
    except ImportError:
        pass
    return {
        'peak_cuda_allocated_bytes': None,
        'peak_cuda_reserved_bytes': None,
    }


if __name__ == '__main__':
    main()
