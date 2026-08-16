"""
Annotated-map Day 10 benchmark and report generation.

This backend deliberately bypasses image detection and LiDAR lifting.  It
injects released VLA3D boxes as one perfect observation through the production
ObjectMap, then exercises persistent identity, relation solving, final OBB
adaptation, protocol validation, and terminal result logging.  Its answer
comparison is a development proxy derived from the same released annotations,
not independent competition ground truth.
"""

from collections import Counter
from dataclasses import asdict
import json
from math import cos
from math import sin
from pathlib import Path
from time import perf_counter

import numpy as np

from qmapnav.evaluation.dataset_loader import load_development_scenes
from qmapnav.evaluation.ground_truth import OracleObject
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.evaluation.object_reference_runner import (
    ObjectReferenceBenchmarkRunner,
)
from qmapnav.evaluation.oracle import solve_object_reference
from qmapnav.language import parse_question
from qmapnav.mapping.geometry_evaluation import aabb_iou_3d
from qmapnav.mapping.geometry_evaluation import ReferenceUprightBox
from qmapnav.mapping.geometry_evaluation import upright_box_aabb
from qmapnav.mapping.geometry_evaluation import upright_box_iou_3d
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.mission.episode_reports import build_object_reference_manifest
from qmapnav.mission.episode_reports import classify_primary_failure
from qmapnav.mission.episode_reports import FixCandidate
from qmapnav.mission.episode_reports import ObjectReferenceCase
from qmapnav.mission.episode_reports import ObjectReferenceEpisodeResult
from qmapnav.mission.episode_reports import rank_fix_candidates
from qmapnav.mission.episode_reports import StageEvidence
from qmapnav.mission.marker_adapter import object_instance_to_marker_spec
from qmapnav.mission.marker_adapter import validate_marker_spec
from qmapnav.perception.vocabulary import detector_classes_from_task_specification
from qmapnav.reasoning.colour_types import ColourEstimate
from qmapnav.reasoning.object_reference_solver import (
    resolve_object_reference_from_maps,
)


def run_annotated_map_benchmark(
    *,
    mode: str,
    questions_path: Path,
    simulation_root: Path,
    vla_root: Path,
    output_root: Path,
    run_id: str | None = None,
) -> object:
    """Load released scenes and run the annotated persistent-map backend."""
    scenes = load_development_scenes(
        questions_path=questions_path,
        simulation_root=simulation_root,
        vla_root=vla_root,
    )
    cases = build_object_reference_manifest(
        tuple(question for scene in scenes for question in scene.questions),
        parse_question,
    )
    by_scene = {scene.scene_id: scene for scene in scenes}
    executor = AnnotatedMapCaseExecutor(by_scene)
    run = ObjectReferenceBenchmarkRunner(
        cases, executor, output_root
    ).run(mode=mode, run_id=run_id)
    directory = Path(output_root) / run.summary.run_id
    _write_report(directory, run, cases)
    return run


class AnnotatedMapCaseExecutor:
    """Execute one released question through a perfect one-view object map."""

    def __init__(self, scenes: dict[str, OracleScene]) -> None:
        self._scenes = dict(scenes)

    def __call__(
        self,
        case: ObjectReferenceCase,
        run_id: str,
        directory: Path,
    ) -> ObjectReferenceEpisodeResult:
        started = perf_counter()
        scene = self._scenes[case.scene_id]
        task = parse_question(case.question)
        parser_correct = (
            task.task_type == 'object_reference'
            and bool(task.entities)
            and task.entities[0].class_name == case.expected_target_class
            and tuple(dict.fromkeys(
                item.class_name for item in task.entities[1:]
            )) == case.expected_anchor_classes
            and not task.ordered_route_steps
        )
        object_map, source_ids, lifting, fusion = _annotated_object_map(scene)
        structural_map = StructuralMap()
        resolution = resolve_object_reference_from_maps(
            task, object_map, structural_map
        )
        oracle = solve_object_reference(task, scene)
        expected_id = oracle.selected_object_id
        selected_instance_id = resolution.selected_target_id
        selected_sources = source_ids.get(selected_instance_id, ())
        selected_correct = (
            expected_id is not None and expected_id in selected_sources
        )
        identity_correct = (
            None if selected_instance_id is None
            else len(selected_sources) == 1
        )
        requested_colour = task.entities[0].attributes.get('colour')
        colour_correct = None
        if requested_colour is not None and expected_id is not None:
            colour_correct = selected_correct or _wrong_for_colour(
                scene, selected_sources, expected_id, requested_colour
            ) is False
        generated = resolution.candidate_generation
        target_generated = generated[task.entities[0].entity_id]
        anchors_available = all(
            bool(generated[item.entity_id].retained)
            for item in task.entities[1:]
        )
        marker = None
        marker_errors = ('missing_target_candidate',)
        marker_metrics = None
        if selected_instance_id is not None:
            marker = object_instance_to_marker_spec(
                object_map.get(int(selected_instance_id)), timestamp_ns=1
            )
            marker_errors = validate_marker_spec(marker)
            if expected_id is not None:
                marker_metrics = _marker_metrics(
                    marker,
                    next(item for item in scene.objects
                         if item.object_id == expected_id),
                )
        obb_acceptable = (
            None if marker_metrics is None
            else marker_metrics['obb_iou'] >= 0.25
        )
        protocol_valid = marker is not None and not marker_errors
        proxy_success = (
            selected_correct and obb_acceptable is True and protocol_valid
        )
        evidence = StageEvidence(
            parser_correct=parser_correct,
            target_observed=True,
            target_detected=bool(target_generated.retained),
            anchors_available=anchors_available,
            target_lifted=bool(target_generated.retained),
            identity_correct=identity_correct,
            colour_correct=colour_correct,
            relation_correct=selected_correct,
            target_selected_correctly=selected_correct,
            obb_acceptable=obb_acceptable,
            protocol_valid=protocol_valid,
            detail={
                'answer_provenance': 'derived_from_released_annotations',
                'expected_proxy_object_id': expected_id,
                'selected_source_object_ids': selected_sources,
                'fusion_subtype': (
                    'false_merge' if identity_correct is False else None
                ),
                'relation_subtype': (
                    'annotated_map_ranking_disagreement'
                    if selected_correct is False else None
                ),
            },
        )
        failure = (
            classify_primary_failure(evidence)
            if not proxy_success else classify_primary_failure(StageEvidence())
        )
        relevant_classes = {
            canonicalize_class_name(item.class_name) for item in task.entities
        }
        raw_counts = Counter(
            canonicalize_class_name(item.class_name)
            for item in scene.objects
            if canonicalize_class_name(item.class_name) in relevant_classes
        )
        ranking = resolution.ranked_hypotheses
        duration = perf_counter() - started
        result = ObjectReferenceEpisodeResult(
            run_id=run_id,
            case_id=case.case_id,
            scene_id=case.scene_id,
            question=case.question,
            pipeline_mode='oracle_replay',
            episode_status=(
                'completed' if protocol_valid else 'protocol_failure'
            ),
            parser_mode=task.parse_mode,
            task_specification=_stable(asdict(task)),
            requested_classes=tuple(
                item.canonical_name
                for item in detector_classes_from_task_specification(task)
            ),
            stage_evidence=evidence,
            target_detections=raw_counts[
                canonicalize_class_name(task.entities[0].class_name)
            ],
            anchor_detections={
                item.class_name: raw_counts[
                    canonicalize_class_name(item.class_name)
                ] for item in task.entities[1:]
            },
            lifting_results=tuple(
                item for item in lifting
                if item['class_name'] in relevant_classes
            ),
            object_candidates_3d=len(scene.objects),
            persistent_instances=len(object_map.active_instances()),
            fusion_events=fusion,
            ranked_target_ids=tuple(item.target_id for item in ranking),
            ranked_target_scores=tuple(item.score for item in ranking),
            ranked_score_components=tuple(item.to_dict() for item in ranking),
            confidence_margin=resolution.confidence_margin,
            unresolved_constraints=resolution.unresolved_constraints,
            selected_target_id=selected_instance_id,
            predicted_box=(
                None if marker is None else {
                    'frame_id': marker.frame_id,
                    'centre_xyz': marker.centre_xyz,
                    'orientation_xyzw': marker.orientation_xyzw,
                    'dimensions_xyz': marker.dimensions_xyz,
                }
            ),
            marker_validation_errors=marker_errors,
            marker_published=protocol_valid,
            marker_publish_count=int(protocol_valid),
            marker_publish_time_sec=(duration if protocol_valid else None),
            matching_waypoint_published=protocol_valid,
            target_selection_correct=selected_correct,
            centre_error_m=(
                None if marker_metrics is None
                else marker_metrics['centre_error_m']
            ),
            aabb_iou=(
                None if marker_metrics is None else marker_metrics['aabb_iou']
            ),
            obb_iou=(
                None if marker_metrics is None else marker_metrics['obb_iou']
            ),
            yaw_error_rad=(
                None if marker_metrics is None
                else marker_metrics['yaw_error_rad']
            ),
            marker_success=(
                None if marker_metrics is None
                else proxy_success
            ),
            success=proxy_success,
            primary_failure_category=failure.category,
            failure_subtype=failure.subtype,
            failure_detail=failure.detail,
            manual_review=(
                'Correctness proxy is derived from released VLA3D annotations; '
                '2D detection and LiDAR lifting are bypassed in this run.'
            ),
            episode_duration_sec=duration,
            trace_path=str(directory / 'decision_trace.jsonl'),
            evidence_directory=str(directory),
            final_response_logged=True,
            proxy_score=2.0 * float(proxy_success),
        )
        (directory / 'question.txt').write_text(
            case.question + '\n', encoding='utf-8'
        )
        _write_json(
            directory / 'task_specification.json',
            result.task_specification,
        )
        _write_json(
            directory / 'candidate_ranking.json',
            result.ranked_score_components,
        )
        _write_json(directory / 'lifting_diagnostics.json', lifting)
        _write_json(directory / 'fusion_diagnostics.json', fusion)
        _write_json(directory / 'final_marker.json', result.predicted_box)
        _write_json(directory / 'marker_validation.json', {
            'errors': marker_errors,
            'publish_count': result.marker_publish_count,
            'matching_waypoint_published': result.matching_waypoint_published,
            'publish_time_sec': result.marker_publish_time_sec,
        })
        _write_json(directory / 'evidence_manifest.json', {
            'mode': 'annotated_map_control',
            'perception_limit': (
                '2D detector and real LiDAR association are intentionally '
                'bypassed; use perceived episode records for those claims.'
            ),
            'files': [
                'question.txt',
                'task_specification.json',
                'annotated_map_source_ids.json',
                'lifting_diagnostics.json',
                'fusion_diagnostics.json',
                'candidate_ranking.json',
                'final_marker.json',
                'marker_validation.json',
                'decision_trace.jsonl',
                'episode_result.json',
            ],
        })
        (directory / 'decision_trace.jsonl').write_text(
            json.dumps({
                'event': 'object_reference_committed',
                'case_id': case.case_id,
                'selected_target_id': selected_instance_id,
                'selected_source_object_ids': selected_sources,
                'confidence_margin': resolution.confidence_margin,
                'unresolved_constraints': resolution.unresolved_constraints,
                'marker_errors': marker_errors,
                'primary_failure_category': failure.category,
            }, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        _write_json(directory / 'annotated_map_source_ids.json', source_ids)
        return result


def _annotated_object_map(scene: OracleScene):
    object_map = ObjectMap()
    candidates = [_oracle_candidate(item, index) for index, item in enumerate(
        scene.objects
    )]
    observations = [
        ViewpointObservation(
            'annotated_map_view', np.zeros(4), 1,
            item.detection_id, item.point_count, 1.0, 'full',
        ) for item in candidates
    ]
    instance_ids = object_map.add_viewpoint_candidates(candidates, observations)
    source_ids = {}
    for source, instance_id in zip(scene.objects, instance_ids):
        source_ids.setdefault(str(instance_id), []).append(source.object_id)
        probabilities = _colour_probabilities(source)
        if probabilities:
            dominant = max(sorted(probabilities), key=probabilities.get)
            object_map.update_colour(
                instance_id,
                ColourEstimate(
                    probabilities, dominant, probabilities[dominant], 100,
                    None, None, 'annotated_map_view',
                    f'oracle_{source.object_id}', 'good',
                ),
            )
    lifting = tuple({
        'source_object_id': source.object_id,
        'class_name': canonicalize_class_name(source.class_name),
        'status': 'good',
        'projected': 8,
        'post_ground': 8,
        'clustered': 8,
        'final': 8,
    } for source in scene.objects)
    fusion = tuple(event.to_dict() for event in object_map.last_events)
    return (
        object_map,
        {key: tuple(value) for key, value in sorted(source_ids.items())},
        lifting,
        fusion,
    )


def _oracle_candidate(source: OracleObject, index: int) -> ObjectCandidate3D:
    centre = np.asarray(source.centre_xyz, dtype=np.float64)
    dimensions = np.asarray(source.dimensions_xyz, dtype=np.float64)
    axis = np.array([cos(source.yaw), sin(source.yaw)])
    perpendicular = np.array([-sin(source.yaw), cos(source.yaw)])
    xy = np.asarray([
        centre[:2] + sx * dimensions[0] / 2.0 * axis
        + sy * dimensions[1] / 2.0 * perpendicular
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])
    points = np.asarray([
        (point[0], point[1], centre[2] + z * dimensions[2] / 2.0)
        for z in (-1, 1) for point in xy
    ])
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)
    counts = LiftingCounts(8, 8, 8, 8, 8, 8, 8)
    confidence = ConfidenceComponents(*(1.0 for _ in range(9)))
    identifier = f'oracle_{index:04d}_{source.object_id}'
    return ObjectCandidate3D(
        identifier, identifier, canonicalize_class_name(source.class_name), 1.0,
        GeometrySource.COMBINED, 1, 1, 1, 1, 'exact', 0.0, 0.0, 0.0,
        False, points, np.arange(8), centre, aabb_min, aabb_max, centre,
        dimensions, source.yaw, source.yaw, 1.0, 1.0,
        GeometryStatus.GOOD, False, False, counts, confidence,
        {'oracle_replay': True, 'source_object_id': source.object_id},
    )


def _colour_probabilities(source: OracleObject) -> dict[str, float]:
    values = {
        canonicalize_class_name(item.label): float(item.proportion)
        for item in source.colours if item.proportion > 0.0
    }
    total = sum(values.values())
    return (
        {} if total <= 0.0
        else {key: value / total for key, value in sorted(values.items())}
    )


def _wrong_for_colour(scene, selected_sources, expected_id, requested_colour):
    by_id = {item.object_id: item for item in scene.objects}
    expected = _colour_probabilities(by_id[expected_id]).get(
        requested_colour, 0.0
    )
    selected = max(
        (_colour_probabilities(by_id[item]).get(requested_colour, 0.0)
         for item in selected_sources),
        default=0.0,
    )
    return selected + 1.0e-9 < expected


def _marker_metrics(marker, expected):
    expected_box = ReferenceUprightBox(
        np.asarray(expected.centre_xyz), np.asarray(expected.dimensions_xyz),
        expected.yaw, expected.class_name, 'released_vla3d',
    )
    predicted_centre = np.asarray(marker.centre_xyz)
    predicted_dimensions = np.asarray(marker.dimensions_xyz)
    predicted_yaw = 2.0 * np.arctan2(
        marker.orientation_xyzw[2], marker.orientation_xyzw[3]
    )
    expected_aabb = upright_box_aabb(expected_box)
    predicted_corners = _aabb_from_box(
        predicted_centre, predicted_dimensions, predicted_yaw
    )
    return {
        'centre_error_m': float(np.linalg.norm(
            predicted_centre - expected_box.centre_xyz
        )),
        'aabb_iou': aabb_iou_3d(
            *predicted_corners, *expected_aabb
        ),
        'obb_iou': upright_box_iou_3d(
            predicted_centre, predicted_dimensions, predicted_yaw,
            expected_box.centre_xyz, expected_box.dimensions_xyz,
            expected_box.yaw_rad,
        ),
        'yaw_error_rad': float(abs(
            ((predicted_yaw - expected_box.yaw_rad + np.pi / 2.0)
             % np.pi) - np.pi / 2.0
        )),
    }


def _aabb_from_box(centre, dimensions, yaw):
    box = ReferenceUprightBox(centre, dimensions, yaw)
    return upright_box_aabb(box)


def _write_report(directory, run, cases):
    by_case = {item.case_id: item for item in cases}
    _write_parser_audit(directory, run)
    _write_grouped_metrics(directory, run, by_case)
    failures = run.summary.failure_counts
    fixes = []
    proposals = {
        'duplicate_instance': (
            'tighten same-view overlap and class-aware false-merge gates', 0.75,
            1.5,
        ),
        'incorrect_colour': (
            'rebalance calibrated colour evidence as a soft ranking term', 0.75,
            1.0,
        ),
        'bad_relation': (
            'calibrate joint relation thresholds on the affected tags', 0.70,
            2.0,
        ),
        'incorrect_obb': (
            'use accumulated stable OBB or conservative low-yaw fallback', 0.80,
            1.5,
        ),
        'missed_anchor': (
            'improve structural/object anchor association', 0.65, 2.0,
        ),
        'missed_target': (
            'use the bounded targeted viewpoint for observation misses', 0.60,
            2.0,
        ),
        'parsing': ('extend deterministic grammar pattern coverage', 0.90, 1.0),
        'bad_lifting': (
            'prefer accumulated scan and closer bounded re-observation', 0.65,
            2.0,
        ),
        'protocol_failure': (
            'enforce final marker adapter assertions before publish', 0.95, 0.5,
        ),
    }
    for category, count in failures.items():
        proposal, confidence, effort = proposals[category]
        fixes.append(FixCandidate(
            category, count, proposal, confidence, effort, 1.0
        ))
    ranked = rank_fix_candidates(tuple(fixes)) if fixes else ()
    _write_json(directory / 'fix_priorities.json', [
        item.to_dict() for item in ranked
    ])
    lines = [
        '# Day 10 annotated-map object-reference benchmark', '',
        '> Correctness labels here are development proxies derived from the same '
        'released VLA3D annotations. This run bypasses 2D detection and LiDAR '
        'lifting; it does not claim perceived-scene accuracy.', '',
        f'- Cases scheduled: {run.summary.scheduled_cases}',
        f'- Final responses logged: {run.summary.final_responses_logged}',
        f'- Proxy target selections correct: '
        f'{run.summary.target_selections_correct}/'
        f'{run.summary.target_labels_available}',
        f'- Valid marker publications: {run.summary.protocol_valid}',
        f'- Proxy marker successes: {run.summary.marker_successes}/'
        f'{run.summary.marker_labels_available}',
        '', '## Failure counts', '',
    ]
    lines.extend(
        f'- {name}: {failures.get(name, 0)}' for name in (
            'parsing', 'missed_target', 'missed_anchor', 'incorrect_colour',
            'bad_lifting', 'duplicate_instance', 'bad_relation',
            'incorrect_obb', 'protocol_failure',
        )
    )
    lines.extend(['', '## Ranked fixes', ''])
    lines.extend(
        f'{index}. `{item.failure_source}` — {item.proposed_fix} '
        f'(affected={item.affected_tasks}, priority={item.priority:.3f})'
        for index, item in enumerate(ranked, start=1)
    )
    lines.extend(['', '## Cases', ''])
    for result in run.results:
        lines.append(
            f'- `{result.case_id}` ({", ".join(by_case[result.case_id].tags)}): '
            f'response={result.final_response_logged}, '
            f'target_proxy={result.target_selection_correct}, '
            f'marker={result.marker_success}, '
            f'failure={result.primary_failure_category or "none"}'
        )
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_parser_audit(directory, run):
    records = []
    for result in run.results:
        task = parse_question(result.question)
        target = task.entities[0]
        records.append({
            'case_id': result.case_id,
            'parse_mode': task.parse_mode,
            'parser_correct': result.stage_evidence.parser_correct,
            'target_class': target.class_name,
            'target_attributes': dict(target.attributes),
            'anchor_classes': [item.class_name for item in task.entities[1:]],
            'relations': [asdict(item) for item in task.relations],
            'ordered_route_step_count': len(task.ordered_route_steps),
        })
    _write_json(directory / 'parser_audit.json', records)


def _write_grouped_metrics(directory, run, by_case):
    def aggregate(results):
        failures = Counter(
            item.primary_failure_category for item in results
            if item.primary_failure_category is not None
        )
        return {
            'tasks': len(results),
            'final_responses_logged': sum(
                item.final_response_logged for item in results
            ),
            'target_selections_correct': sum(
                item.target_selection_correct is True for item in results
            ),
            'marker_successes': sum(
                item.marker_success is True for item in results
            ),
            'failure_counts': dict(sorted(failures.items())),
        }

    scenes = {}
    for scene_id in sorted({item.scene_id for item in run.results}):
        scenes[scene_id] = aggregate([
            item for item in run.results if item.scene_id == scene_id
        ])
    tags = {}
    all_tags = sorted({
        tag for case in by_case.values() for tag in case.tags
    })
    for tag in all_tags:
        tags[tag] = aggregate([
            item for item in run.results if tag in by_case[item.case_id].tags
        ])
    _write_json(directory / 'grouped_metrics.json', {
        'overall': aggregate(run.results),
        'scenes': scenes,
        'tags': tags,
    })


def _stable(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def main(argv=None) -> int:
    """Run the annotated-map benchmark from the installed console script."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), required=True)
    parser.add_argument('--questions-path', type=Path, required=True)
    parser.add_argument('--simulation-root', type=Path, required=True)
    parser.add_argument('--vla-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    run = run_annotated_map_benchmark(
        mode=args.mode,
        questions_path=args.questions_path,
        simulation_root=args.simulation_root,
        vla_root=args.vla_root,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(asdict(run.summary), indent=2, sort_keys=True))
    return 0 if run.summary.final_responses_logged == run.summary.scheduled_cases else 1


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['AnnotatedMapCaseExecutor', 'run_annotated_map_benchmark']
