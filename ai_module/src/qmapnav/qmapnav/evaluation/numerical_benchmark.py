"""Released numerical benchmark over annotated persistent ObjectMaps."""

import argparse
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from qmapnav.counting import CountStabilityMachine
from qmapnav.counting import resolve_numerical_from_maps
from qmapnav.evaluation.dataset_loader import load_development_scenes
from qmapnav.evaluation.object_reference_replay import _annotated_object_map
from qmapnav.evaluation.oracle import solve_numerical
from qmapnav.language import parse_question
from qmapnav.mapping import StructuralMap


NUMERICAL_FAILURE_CATEGORIES = frozenset({
    'bad_relation',
    'duplicate_instances',
    'exploration_failure',
    'false_merge',
    'incorrect_colour',
    'missed_anchor',
    'missed_target_instances',
    'parsing',
    'protocol_failure',
    'timeout',
    'unstable_count',
})


@dataclass(frozen=True)
class NumericalEpisodeResult:
    """Terminal record for one released numerical question."""

    case_id: str
    scene_id: str
    question: str
    predicted_count: int
    expected_count: int
    correct: bool
    definite_instance_ids: tuple[int, ...]
    probable_instance_ids: tuple[int, ...]
    rejected_instance_ids: tuple[int, ...]
    unresolved_instance_ids: tuple[int, ...]
    stable: bool
    observations_used: int
    independent_viewpoints: int
    runtime_sec: float
    failure_category: str | None
    final_response_logged: bool
    result_trace: dict[str, object]

    def __post_init__(self) -> None:
        for name in ('case_id', 'scene_id', 'question'):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise ValueError(f'{name} must be non-empty')
        for name in (
            'predicted_count', 'expected_count', 'observations_used',
            'independent_viewpoints',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be non-negative')
        if not isfinite(self.runtime_sec) or self.runtime_sec < 0.0:
            raise ValueError('runtime_sec must be finite and non-negative')
        if self.failure_category not in NUMERICAL_FAILURE_CATEGORIES | {None}:
            raise ValueError('unsupported numerical failure category')
        if self.correct != (self.predicted_count == self.expected_count):
            raise ValueError('correct flag does not match exact count equality')

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe terminal evidence."""
        data = asdict(self)
        for name in (
            'definite_instance_ids',
            'probable_instance_ids',
            'rejected_instance_ids',
            'unresolved_instance_ids',
        ):
            data[name] = list(data[name])
        return data


def run_annotated_numerical_benchmark(
    *,
    questions_path: Path,
    simulation_root: Path,
    vla_root: Path,
    output_directory: Path,
) -> tuple[NumericalEpisodeResult, ...]:
    """Run all 15 questions and persist a final answer for every case."""
    scenes = load_development_scenes(
        questions_path=questions_path,
        simulation_root=simulation_root,
        vla_root=vla_root,
    )
    results = []
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    cases_root = output_directory / 'cases'
    cases_root.mkdir(exist_ok=True)
    for scene in scenes:
        question = next(
            item for item in scene.questions if item.task_type == 'numerical'
        )
        started = perf_counter()
        task = parse_question(question.question_text)
        object_map, source_ids, _, _ = _annotated_object_map(scene)
        persistent = resolve_numerical_from_maps(
            task, object_map, StructuralMap()
        )
        oracle = solve_numerical(task, scene)
        stability = CountStabilityMachine()
        state = None
        for index, viewpoint in enumerate(
            ('annotated_view_a', 'annotated_view_a', 'annotated_view_b')
        ):
            state = stability.update(
                persistent,
                viewpoint_id=viewpoint,
                time_remaining_sec=500.0 - index,
                episode_time_sec=float(index),
            )
        if not state.should_publish:
            state = stability.force_best_available(
                'annotated_benchmark_observation_budget_exhausted'
            )
        final = state.result
        failure = _failure_category(task, final, oracle.count)
        result = NumericalEpisodeResult(
            question.question_id,
            scene.scene_id,
            question.question_text,
            final.count,
            oracle.count,
            final.count == oracle.count,
            final.definite_instance_ids,
            final.probable_instance_ids,
            final.rejected_instance_ids,
            final.unresolved_instance_ids,
            final.stable,
            3,
            state.independent_viewpoints,
            perf_counter() - started,
            failure,
            True,
            {
                **final.to_dict(),
                'persistent_to_source_ids': source_ids,
                'expected_source_object_ids': list(
                    oracle.matching_object_ids
                ),
            },
        )
        case_directory = cases_root / question.question_id
        case_directory.mkdir(exist_ok=True)
        _write_json(case_directory / 'episode_result.json', result.to_dict())
        _write_json(case_directory / 'numerical_trace.json', result.result_trace)
        results.append(result)
    _write_outputs(output_directory, tuple(results))
    return tuple(results)


def _failure_category(task, result, expected_count):
    if task.parse_mode not in {'full', 'degraded'} or not task.entities:
        return 'parsing'
    if result.count != expected_count:
        if task.entities[0].attributes.get('colour') is not None:
            return 'incorrect_colour'
        if not result.anchor_ambiguity.hypotheses:
            return 'missed_anchor'
        if result.count < expected_count:
            return 'missed_target_instances'
        return 'duplicate_instances'
    if not result.stable:
        return 'unstable_count'
    return None


def _write_outputs(directory, results):
    failures = Counter(
        item.failure_category for item in results
        if item.failure_category is not None
    )
    summary = {
        'scheduled_cases': len(results),
        'final_responses_logged': sum(
            item.final_response_logged for item in results
        ),
        'correct': sum(item.correct for item in results),
        'exact_accuracy': (
            sum(item.correct for item in results) / len(results)
        ),
        'stable': sum(item.stable for item in results),
        'mean_runtime_sec': (
            sum(item.runtime_sec for item in results) / len(results)
        ),
        'failure_counts': dict(sorted(failures.items())),
        'answer_provenance': 'derived_from_released_vla3d_annotations',
        'perception_limit': (
            'Annotated-map control bypasses detector and real LiDAR lifting.'
        ),
    }
    _write_json(directory / 'summary.json', summary)
    _write_json(
        directory / 'per_case.json',
        [item.to_dict() for item in results],
    )


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    """Run the released benchmark from explicit development-data paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--simulation-root', type=Path, required=True)
    parser.add_argument('--vla-root', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    arguments = parser.parse_args()
    results = run_annotated_numerical_benchmark(
        questions_path=arguments.questions,
        simulation_root=arguments.simulation_root,
        vla_root=arguments.vla_root,
        output_directory=arguments.output_directory,
    )
    print(json.dumps({
        'scheduled_cases': len(results),
        'final_responses_logged': sum(
            item.final_response_logged for item in results
        ),
        'correct': sum(item.correct for item in results),
        'stable': sum(item.stable for item in results),
    }, indent=2, sort_keys=True))


__all__ = [
    'NUMERICAL_FAILURE_CATEGORIES',
    'NumericalEpisodeResult',
    'main',
    'run_annotated_numerical_benchmark',
]
