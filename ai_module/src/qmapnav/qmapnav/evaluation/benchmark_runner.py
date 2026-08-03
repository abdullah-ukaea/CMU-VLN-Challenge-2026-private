"""Offline quick and full regression harness for the Day 3 oracle system."""

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from qmapnav.evaluation.dataset_loader import load_development_scenes
from qmapnav.evaluation.ground_truth import ground_truth_to_data
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.evaluation.ground_truth import QuestionRecord
from qmapnav.evaluation.metrics import count_accuracy_metric
from qmapnav.evaluation.metrics import object_selection_metric
from qmapnav.evaluation.metrics import relation_metrics
from qmapnav.evaluation.metrics import RelationKey
from qmapnav.evaluation.metrics import semantic_route_metric
from qmapnav.evaluation.metrics import TimingMetric
from qmapnav.language import parse_question
from qmapnav.reasoning import geometric_relation_holds
from qmapnav.reasoning import OracleReasoningError
from qmapnav.reasoning import plan_semantic_route
from qmapnav.reasoning import RoutePlanningError
from qmapnav.reasoning import solve_numerical
from qmapnav.reasoning import solve_object_reference


BENCHMARK_SCHEMA_VERSION = 1
SUPPORTED_RELATION_METRICS = (
    'above',
    'below',
    'between',
    'inside',
    'near',
    'on',
)
QUICK_QUESTION_IDS = (
    'home_building_2_numerical_01',
    'japanese_room_object_reference_01',
    'loft_object_reference_01',
    'chinese_room_instruction_following_02',
    'loft_instruction_following_01',
    'office_1_instruction_following_02',
)


class FailureCategory(str, Enum):
    """Specific, stable categories used by machine-readable reports."""

    AMBIGUOUS_REFERENCE = 'AMBIGUOUS_REFERENCE'
    ANSWER_MAPPING_MISSING = 'ANSWER_MAPPING_MISSING'
    ATTRIBUTE_MISMATCH = 'ATTRIBUTE_MISMATCH'
    CONSTRAINT_ORDER_FAILURE = 'CONSTRAINT_ORDER_FAILURE'
    DATASET_LOAD_FAILURE = 'DATASET_LOAD_FAILURE'
    ENTITY_NOT_FOUND = 'ENTITY_NOT_FOUND'
    EVALUATION_ERROR = 'EVALUATION_ERROR'
    FORBIDDEN_REGION_VIOLATION = 'FORBIDDEN_REGION_VIOLATION'
    INVALID_GATE = 'INVALID_GATE'
    NO_REACHABLE_GOAL = 'NO_REACHABLE_GOAL'
    PARSE_FAILURE = 'PARSE_FAILURE'
    PATH_NOT_FOUND = 'PATH_NOT_FOUND'
    RELATION_MISMATCH = 'RELATION_MISMATCH'
    TERMINAL_GOAL_MISSED = 'TERMINAL_GOAL_MISSED'


@dataclass(frozen=True)
class QuestionBenchmarkResult:
    """One question's oracle prediction, metrics, timing, and diagnostics."""

    question_id: str
    scene_id: str
    task_type: str
    question_text: str
    parse_mode: str | None
    structural_success: bool
    answer_label_available: bool
    prediction: Any
    metrics: Any
    timing: TimingMetric
    proxy_score: float | None
    proxy_maximum: float | None
    failure_categories: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    route: Any | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark counts without hiding unavailable answer labels."""

    mode: str
    question_count: int
    task_counts: dict[str, int]
    structural_successes: int
    structural_failures: int
    answer_labels_available: int
    answer_labels_missing: int
    proxy_score_earned: float
    proxy_score_available_maximum: float
    proxy_score_theoretical_maximum: float
    normalized_available_proxy_score: float | None
    instruction_successes: int
    instruction_count: int
    total_runtime_seconds: float
    mean_question_seconds: float
    failure_category_counts: dict[str, int]


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Complete in-memory result mirrored by the JSON report files."""

    schema_version: int
    summary: BenchmarkSummary
    questions: tuple[QuestionBenchmarkResult, ...]
    relation_metrics: tuple[Any, ...]


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _timing(
    parser_seconds: float,
    reasoning_seconds: float,
    planning_seconds: float,
    started_at: float,
) -> TimingMetric:
    return TimingMetric(
        parser_seconds=parser_seconds,
        reasoning_seconds=reasoning_seconds,
        planning_seconds=planning_seconds,
        execution_seconds=None,
        total_seconds=perf_counter() - started_at,
    )


def _route_failures(metric: Any) -> tuple[str, ...]:
    categories = []
    if not metric.order_correct:
        categories.append(FailureCategory.CONSTRAINT_ORDER_FAILURE.value)
    if metric.forbidden_violation_count:
        categories.append(FailureCategory.FORBIDDEN_REGION_VIOLATION.value)
    if not metric.terminal_goal_reached:
        categories.append(FailureCategory.TERMINAL_GOAL_MISSED.value)
    return tuple(categories)


def _reasoning_failure_category(message: str) -> str:
    if 'no scene object' in message or 'no candidate' in message:
        return FailureCategory.ENTITY_NOT_FOUND.value
    return FailureCategory.EVALUATION_ERROR.value


def _planning_failure_category(message: str) -> str:
    lowered = message.lower()
    if 'gate' in lowered:
        return FailureCategory.INVALID_GATE.value
    if 'reachable' in lowered or 'free cell' in lowered:
        return FailureCategory.NO_REACHABLE_GOAL.value
    if 'path' in lowered:
        return FailureCategory.PATH_NOT_FOUND.value
    if 'candidate' in lowered or 'object' in lowered:
        return FailureCategory.ENTITY_NOT_FOUND.value
    return FailureCategory.EVALUATION_ERROR.value


def _failed_question(
    question: QuestionRecord,
    *,
    started_at: float,
    parser_seconds: float,
    reasoning_seconds: float,
    planning_seconds: float,
    parse_mode: str | None,
    category: str,
    message: str,
) -> QuestionBenchmarkResult:
    return QuestionBenchmarkResult(
        question_id=question.question_id,
        scene_id=question.scene_id,
        task_type=question.task_type,
        question_text=question.question_text,
        parse_mode=parse_mode,
        structural_success=False,
        answer_label_available=False,
        prediction=None,
        metrics=None,
        timing=_timing(
            parser_seconds,
            reasoning_seconds,
            planning_seconds,
            started_at,
        ),
        proxy_score=None,
        proxy_maximum=None,
        failure_categories=(category,),
        warnings=(message,),
    )


def evaluate_question(
    scene: OracleScene,
    question: QuestionRecord,
) -> QuestionBenchmarkResult:
    """Parse and evaluate one question against a normalized oracle scene."""
    if question.scene_id != scene.scene_id:
        raise ValueError('question and scene IDs do not match')
    started_at = perf_counter()
    parser_seconds = 0.0
    reasoning_seconds = 0.0
    planning_seconds = 0.0
    parse_mode = None
    parse_started = perf_counter()
    try:
        task = parse_question(question.question_text)
        parser_seconds = perf_counter() - parse_started
        parse_mode = task.parse_mode
    except (TypeError, ValueError) as exc:
        parser_seconds = perf_counter() - parse_started
        return _failed_question(
            question,
            started_at=started_at,
            parser_seconds=parser_seconds,
            reasoning_seconds=reasoning_seconds,
            planning_seconds=planning_seconds,
            parse_mode=parse_mode,
            category=FailureCategory.PARSE_FAILURE.value,
            message=str(exc),
        )

    try:
        if question.task_type == 'numerical':
            reasoning_started = perf_counter()
            answer = solve_numerical(task, scene)
            reasoning_seconds = perf_counter() - reasoning_started
            metric = count_accuracy_metric(answer.count, question.expected_count)
            failures = []
            if not metric.label_available:
                failures.append(FailureCategory.ANSWER_MAPPING_MISSING.value)
            score = (
                float(metric.exact_match) if metric.label_available else None
            )
            return QuestionBenchmarkResult(
                question_id=question.question_id,
                scene_id=question.scene_id,
                task_type=question.task_type,
                question_text=question.question_text,
                parse_mode=parse_mode,
                structural_success=True,
                answer_label_available=metric.label_available,
                prediction=answer,
                metrics=metric,
                timing=_timing(
                    parser_seconds,
                    reasoning_seconds,
                    planning_seconds,
                    started_at,
                ),
                proxy_score=score,
                proxy_maximum=(1.0 if metric.label_available else None),
                failure_categories=tuple(failures),
                warnings=answer.warnings,
            )

        if question.task_type == 'object_reference':
            reasoning_started = perf_counter()
            answer = solve_object_reference(task, scene)
            reasoning_seconds = perf_counter() - reasoning_started
            metric = object_selection_metric(
                answer.selected_object_id,
                question.expected_object_id,
            )
            failures = []
            if not metric.label_available:
                failures.append(FailureCategory.ANSWER_MAPPING_MISSING.value)
            if answer.selected_object_id is None:
                failures.append(FailureCategory.ENTITY_NOT_FOUND.value)
            elif answer.confidence_margin == 0.0:
                failures.append(FailureCategory.AMBIGUOUS_REFERENCE.value)
            score = (
                2.0 * float(metric.correct) if metric.label_available else None
            )
            return QuestionBenchmarkResult(
                question_id=question.question_id,
                scene_id=question.scene_id,
                task_type=question.task_type,
                question_text=question.question_text,
                parse_mode=parse_mode,
                structural_success=answer.selected_object_id is not None,
                answer_label_available=metric.label_available,
                prediction=answer,
                metrics=metric,
                timing=_timing(
                    parser_seconds,
                    reasoning_seconds,
                    planning_seconds,
                    started_at,
                ),
                proxy_score=score,
                proxy_maximum=(2.0 if metric.label_available else None),
                failure_categories=tuple(failures),
                warnings=answer.warnings,
            )

        trajectory = scene.trajectory_for(question.question_id)
        start_xy = trajectory.points_xyz[0][:2]
        planning_started = perf_counter()
        route = plan_semantic_route(task, scene, start_xy)
        planning_seconds = perf_counter() - planning_started
        metric = semantic_route_metric(
            route.waypoints_xy,
            route.required_regions,
            route.forbidden_regions,
        )
        route_report = {
            'waypoints_xy': route.waypoints_xy,
            'required_region_ids': tuple(
                region.region_id for region in route.required_regions
            ),
            'forbidden_region_ids': tuple(
                region.region_id for region in route.forbidden_regions
            ),
            'resolved_step_object_ids': route.resolved_step_object_ids,
        }
        return QuestionBenchmarkResult(
            question_id=question.question_id,
            scene_id=question.scene_id,
            task_type=question.task_type,
            question_text=question.question_text,
            parse_mode=parse_mode,
            structural_success=metric.success,
            answer_label_available=True,
            prediction={
                'waypoint_count': len(route.waypoints_xy),
                'resolved_step_object_ids': route.resolved_step_object_ids,
            },
            metrics=metric,
            timing=_timing(
                parser_seconds,
                reasoning_seconds,
                planning_seconds,
                started_at,
            ),
            proxy_score=metric.proxy_score,
            proxy_maximum=6.0,
            failure_categories=_route_failures(metric),
            warnings=route.warnings,
            route=route_report,
        )
    except KeyError as exc:
        return _failed_question(
            question,
            started_at=started_at,
            parser_seconds=parser_seconds,
            reasoning_seconds=reasoning_seconds,
            planning_seconds=planning_seconds,
            parse_mode=parse_mode,
            category=FailureCategory.DATASET_LOAD_FAILURE.value,
            message=f'missing linked development data: {exc}',
        )
    except OracleReasoningError as exc:
        return _failed_question(
            question,
            started_at=started_at,
            parser_seconds=parser_seconds,
            reasoning_seconds=reasoning_seconds,
            planning_seconds=planning_seconds,
            parse_mode=parse_mode,
            category=_reasoning_failure_category(str(exc)),
            message=str(exc),
        )
    except RoutePlanningError as exc:
        planning_seconds = perf_counter() - planning_started
        return _failed_question(
            question,
            started_at=started_at,
            parser_seconds=parser_seconds,
            reasoning_seconds=reasoning_seconds,
            planning_seconds=planning_seconds,
            parse_mode=parse_mode,
            category=_planning_failure_category(str(exc)),
            message=str(exc),
        )
    except (TypeError, ValueError) as exc:
        return _failed_question(
            question,
            started_at=started_at,
            parser_seconds=parser_seconds,
            reasoning_seconds=reasoning_seconds,
            planning_seconds=planning_seconds,
            parse_mode=parse_mode,
            category=FailureCategory.EVALUATION_ERROR.value,
            message=str(exc),
        )


def _relation_key(
    scene_id: str,
    relation: str,
    subject_id: str,
    object_ids: tuple[str, ...],
) -> RelationKey:
    return (
        relation,
        f'{scene_id}:{subject_id}',
        tuple(f'{scene_id}:{object_id}' for object_id in object_ids),
    )


def _negative_relation(
    scene: OracleScene,
    relation: str,
    subject_id: str,
    object_ids: tuple[str, ...],
    known: set[tuple[str, str, tuple[str, ...]]],
    used: set[tuple[str, str, tuple[str, ...]]],
) -> tuple[str, str, tuple[str, ...]] | None:
    excluded = {subject_id, *object_ids}
    for replacement in sorted(obj.object_id for obj in scene.objects):
        if replacement in excluded:
            continue
        candidate_anchors = object_ids[:-1] + (replacement,)
        candidate = (relation, subject_id, candidate_anchors)
        reverse = (
            relation,
            subject_id,
            tuple(reversed(candidate_anchors)),
        )
        if candidate not in known and reverse not in known and candidate not in used:
            return candidate
    return None


def evaluate_relation_geometry(
    scenes: tuple[OracleScene, ...],
    *,
    maximum_positives_per_relation: int | None = None,
) -> tuple[Any, ...]:
    """Compare deterministic predicates with sampled GT positives/negatives."""
    positives: dict[str, list[tuple[OracleScene, str, tuple[str, ...]]]] = {
        relation: [] for relation in SUPPORTED_RELATION_METRICS
    }
    known_by_scene: dict[str, set[tuple[str, str, tuple[str, ...]]]] = {}
    for scene in scenes:
        known = {
            (item.relation, item.subject_id, item.object_ids)
            for item in scene.relations
        }
        known_by_scene[scene.scene_id] = known
        for item in scene.relations:
            if item.relation in positives:
                positives[item.relation].append(
                    (scene, item.subject_id, item.object_ids)
                )

    reports = []
    for relation in SUPPORTED_RELATION_METRICS:
        observations = positives[relation]
        if maximum_positives_per_relation is not None:
            observations = observations[:maximum_positives_per_relation]
        expected: set[RelationKey] = set()
        predicted: set[RelationKey] = set()
        negative_keys_by_scene: dict[
            str,
            set[tuple[str, str, tuple[str, ...]]],
        ] = {}
        for scene, subject_id, object_ids in observations:
            key = _relation_key(
                scene.scene_id,
                relation,
                subject_id,
                object_ids,
            )
            expected.add(key)
            subject = scene.object_by_id(subject_id)
            anchors = tuple(scene.object_by_id(item) for item in object_ids)
            if geometric_relation_holds(relation, subject, anchors):
                predicted.add(key)

            scene_negative_keys = negative_keys_by_scene.setdefault(
                scene.scene_id,
                set(),
            )
            negative = _negative_relation(
                scene,
                relation,
                subject_id,
                object_ids,
                known_by_scene[scene.scene_id],
                scene_negative_keys,
            )
            if negative is None:
                continue
            scene_negative_keys.add(negative)
            _, negative_subject_id, negative_object_ids = negative
            negative_key = _relation_key(
                scene.scene_id,
                relation,
                negative_subject_id,
                negative_object_ids,
            )
            subject = scene.object_by_id(negative_subject_id)
            anchors = tuple(
                scene.object_by_id(item) for item in negative_object_ids
            )
            if geometric_relation_holds(relation, subject, anchors):
                predicted.add(negative_key)
        reports.append(
            relation_metrics(
                expected,
                predicted,
                relation=relation,
                negatives_evaluated=sum(
                    len(items) for items in negative_keys_by_scene.values()
                ),
            )
        )
    return tuple(reports)


def _summarize(
    mode: str,
    questions: tuple[QuestionBenchmarkResult, ...],
    total_runtime_seconds: float,
) -> BenchmarkSummary:
    task_counts = {
        task_type: sum(item.task_type == task_type for item in questions)
        for task_type in (
            'numerical',
            'object_reference',
            'instruction_following',
        )
    }
    category_counts: dict[str, int] = {}
    for item in questions:
        for category in item.failure_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    available_scores = [
        item.proxy_score for item in questions if item.proxy_score is not None
    ]
    available_maxima = [
        item.proxy_maximum
        for item in questions
        if item.proxy_maximum is not None
    ]
    theoretical = (
        task_counts['numerical']
        + 2.0 * task_counts['object_reference']
        + 6.0 * task_counts['instruction_following']
    )
    available_maximum = sum(available_maxima)
    return BenchmarkSummary(
        mode=mode,
        question_count=len(questions),
        task_counts=task_counts,
        structural_successes=sum(item.structural_success for item in questions),
        structural_failures=sum(
            not item.structural_success for item in questions
        ),
        answer_labels_available=sum(
            item.answer_label_available for item in questions
        ),
        answer_labels_missing=sum(
            not item.answer_label_available for item in questions
        ),
        proxy_score_earned=sum(available_scores),
        proxy_score_available_maximum=available_maximum,
        proxy_score_theoretical_maximum=theoretical,
        normalized_available_proxy_score=(
            sum(available_scores) / available_maximum
            if available_maximum
            else None
        ),
        instruction_successes=sum(
            item.structural_success
            for item in questions
            if item.task_type == 'instruction_following'
        ),
        instruction_count=task_counts['instruction_following'],
        total_runtime_seconds=total_runtime_seconds,
        mean_question_seconds=(
            sum(item.timing.total_seconds for item in questions) / len(questions)
            if questions
            else 0.0
        ),
        failure_category_counts=dict(sorted(category_counts.items())),
    )


def run_benchmark_on_scenes(
    scenes: tuple[OracleScene, ...],
    *,
    mode: str,
    question_ids: tuple[str, ...] | None = None,
) -> BenchmarkRunResult:
    """Run one deterministic benchmark over already normalized scenes."""
    if mode not in {'quick', 'full'}:
        raise ValueError("mode must be 'quick' or 'full'")
    all_questions = {
        question.question_id: (scene, question)
        for scene in scenes
        for question in scene.questions
    }
    selected_ids = question_ids
    if selected_ids is None:
        selected_ids = (
            QUICK_QUESTION_IDS if mode == 'quick' else tuple(all_questions)
        )
    missing = [item for item in selected_ids if item not in all_questions]
    if missing:
        raise ValueError(f'benchmark question IDs are unavailable: {missing}')

    started_at = perf_counter()
    question_results = tuple(
        evaluate_question(*all_questions[question_id])
        for question_id in selected_ids
    )
    selected_scene_ids = {
        all_questions[question_id][0].scene_id for question_id in selected_ids
    }
    selected_scenes = tuple(
        scene for scene in scenes if scene.scene_id in selected_scene_ids
    )
    relations = evaluate_relation_geometry(
        selected_scenes,
        maximum_positives_per_relation=(25 if mode == 'quick' else None),
    )
    total_runtime = perf_counter() - started_at
    return BenchmarkRunResult(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        summary=_summarize(mode, question_results, total_runtime),
        questions=question_results,
        relation_metrics=relations,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(ground_truth_to_data(value), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def write_benchmark_reports(
    result: BenchmarkRunResult,
    output_directory: Path,
) -> None:
    """Atomically write stable summary, question, relation, and route reports."""
    output = Path(output_directory)
    failures = tuple(
        {
            'question_id': item.question_id,
            'categories': item.failure_categories,
            'warnings': item.warnings,
        }
        for item in result.questions
        if item.failure_categories
    )
    _write_json(output / 'summary.json', result.summary)
    _write_json(output / 'per_question.json', result.questions)
    _write_json(output / 'relation_metrics.json', result.relation_metrics)
    _write_json(output / 'failures.json', failures)
    routes = output / 'routes'
    for item in result.questions:
        if item.route is not None:
            _write_json(routes / f'{item.question_id}.json', item.route)


def _find_repository_root() -> Path:
    candidates = (Path.cwd(), *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / 'questions' / 'questions.json').is_file():
            return candidate
    raise FileNotFoundError(
        'cannot locate repository root containing questions/questions.json'
    )


def run_benchmark(
    *,
    mode: str,
    questions_path: Path,
    simulation_root: Path,
    vla_root: Path,
    output_directory: Path,
    answers_path: Path | None = None,
) -> BenchmarkRunResult:
    """Load released development data, run a mode, and save its reports."""
    scenes = load_development_scenes(
        questions_path=questions_path,
        simulation_root=simulation_root,
        vla_root=vla_root,
        answers_path=answers_path,
    )
    result = run_benchmark_on_scenes(scenes, mode=mode)
    write_benchmark_reports(result, output_directory)
    return result


def _format_summary(summary: BenchmarkSummary) -> str:
    normalized = summary.normalized_available_proxy_score
    score = f'{normalized:.1%}' if normalized is not None else 'unavailable'
    return '\n'.join(
        (
            f'Q-MapNav Oracle Benchmark ({summary.mode})',
            f'Questions: {summary.question_count}',
            f'Structural success: {summary.structural_successes}/'
            f'{summary.question_count}',
            f'Instruction success: {summary.instruction_successes}/'
            f'{summary.instruction_count}',
            f'Available-label proxy: {summary.proxy_score_earned:.1f}/'
            f'{summary.proxy_score_available_maximum:.1f} ({score})',
            f'Missing structured labels: {summary.answer_labels_missing}',
            f'Runtime: {summary.total_runtime_seconds:.3f} seconds',
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line oracle benchmark and return a process status."""
    import argparse

    try:
        repository = _find_repository_root()
    except FileNotFoundError:
        repository = Path.cwd()
    workspace = repository.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), required=True)
    parser.add_argument(
        '--questions-path',
        type=Path,
        default=repository / 'questions' / 'questions.json',
    )
    parser.add_argument(
        '--simulation-root',
        type=Path,
        default=Path(
            os.environ.get('QMAPNAV_SIMULATION_ROOT', workspace / 'simulation')
        ),
    )
    parser.add_argument(
        '--vla-root',
        type=Path,
        default=Path(
            os.environ.get('QMAPNAV_VLA_ROOT', workspace / 'data' / 'vla3d')
        ),
    )
    parser.add_argument('--answers-path', type=Path)
    parser.add_argument(
        '--output-directory',
        type=Path,
        default=None,
    )
    arguments = parser.parse_args(argv)
    output = arguments.output_directory or (
        repository / 'reports' / 'oracle' / arguments.mode / 'latest'
    )
    try:
        result = run_benchmark(
            mode=arguments.mode,
            questions_path=arguments.questions_path,
            simulation_root=arguments.simulation_root,
            vla_root=arguments.vla_root,
            answers_path=arguments.answers_path,
            output_directory=output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_format_summary(result.summary))
    print(f'Reports: {output}')
    return 0 if result.summary.structural_failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'BENCHMARK_SCHEMA_VERSION',
    'BenchmarkRunResult',
    'BenchmarkSummary',
    'FailureCategory',
    'QUICK_QUESTION_IDS',
    'QuestionBenchmarkResult',
    'evaluate_question',
    'evaluate_relation_geometry',
    'main',
    'run_benchmark',
    'run_benchmark_on_scenes',
    'write_benchmark_reports',
]
