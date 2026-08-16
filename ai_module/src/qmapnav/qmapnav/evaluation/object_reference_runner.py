"""Resumable quick/full Day 10 object-reference benchmark runner."""

from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from qmapnav.mission.episode_reports import classify_primary_failure
from qmapnav.mission.episode_reports import manifest_digest
from qmapnav.mission.episode_reports import ObjectReferenceCase
from qmapnav.mission.episode_reports import ObjectReferenceEpisodeResult
from qmapnav.mission.episode_reports import QUICK_OBJECT_REFERENCE_IDS
from qmapnav.mission.episode_reports import StageEvidence


CaseExecutor = Callable[
    [ObjectReferenceCase, str, Path], ObjectReferenceEpisodeResult
]


@dataclass(frozen=True)
class ObjectReferenceBenchmarkSummary:
    """Aggregate counts retaining response and label denominators."""

    run_id: str
    mode: str
    manifest_digest: str
    scheduled_cases: int
    terminal_records: int
    final_responses_logged: int
    markers_published: int
    protocol_valid: int
    target_labels_available: int
    target_selections_correct: int
    marker_labels_available: int
    marker_successes: int
    targeted_viewpoints_used: int
    total_proxy_score: float
    maximum_proxy_score: float
    mean_duration_sec: float
    failure_counts: dict[str, int]
    failure_subtype_counts: dict[str, int]
    tag_counts: dict[str, int]
    tag_success_counts: dict[str, int]


@dataclass(frozen=True)
class ObjectReferenceBenchmarkRun:
    """One immutable run mirrored by the report directory."""

    summary: ObjectReferenceBenchmarkSummary
    results: tuple[ObjectReferenceEpisodeResult, ...]


class ObjectReferenceBenchmarkRunner:
    """Run stable manifest cases and persist every terminal attempt."""

    def __init__(
        self,
        cases: tuple[ObjectReferenceCase, ...],
        executor: CaseExecutor,
        output_root: Path,
    ) -> None:
        if not cases:
            raise ValueError('cases must not be empty')
        if len({item.case_id for item in cases}) != len(cases):
            raise ValueError('case IDs must be unique')
        if not callable(executor):
            raise TypeError('executor must be callable')
        self._cases = tuple(cases)
        self._executor = executor
        self._output_root = Path(output_root)

    def run(
        self,
        *,
        mode: str,
        run_id: str | None = None,
        resume: bool = False,
    ) -> ObjectReferenceBenchmarkRun:
        """Run quick or full cases, containing each backend failure."""
        cases = self._select(mode)
        identifier = run_id or _run_id(mode)
        run_directory = self._output_root / identifier
        run_directory.mkdir(parents=True, exist_ok=resume)
        case_root = run_directory / 'cases'
        case_root.mkdir(exist_ok=True)
        self._write_metadata(run_directory, identifier, mode, cases)
        results = []
        for case in cases:
            directory = case_root / case.case_id
            result_path = directory / 'episode_result.json'
            if resume and result_path.is_file():
                results.append(_load_result(result_path))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            result = self._execute_case(case, identifier, directory)
            _atomic_json(result_path, result.to_dict())
            results.append(result)
        summary = summarize_results(
            identifier,
            mode,
            tuple(cases),
            tuple(results),
        )
        _atomic_json(run_directory / 'summary.json', asdict(summary))
        _atomic_json(
            run_directory / 'per_case.json',
            [item.to_dict() for item in results],
        )
        return ObjectReferenceBenchmarkRun(summary, tuple(results))

    def _select(self, mode: str) -> tuple[ObjectReferenceCase, ...]:
        if mode == 'full':
            if len(self._cases) != 30:
                raise ValueError('full mode requires exactly 30 released cases')
            return self._cases
        if mode != 'quick':
            raise ValueError('mode must be quick or full')
        by_id = {item.case_id: item for item in self._cases}
        missing = [
            item for item in QUICK_OBJECT_REFERENCE_IDS if item not in by_id
        ]
        if missing:
            raise ValueError(f'quick manifest is missing case IDs: {missing}')
        return tuple(by_id[item] for item in QUICK_OBJECT_REFERENCE_IDS)

    def _execute_case(self, case, run_id, directory):
        started = perf_counter()
        try:
            result = self._executor(case, run_id, directory)
            if not isinstance(result, ObjectReferenceEpisodeResult):
                raise TypeError('executor returned an invalid result')
            if result.case_id != case.case_id or result.run_id != run_id:
                raise ValueError('executor result identity does not match case')
        except Exception as error:
            evidence = StageEvidence(
                protocol_valid=False,
                detail={
                    'protocol_subtype': 'runtime_exception',
                    'protocol_detail': str(error),
                },
            )
            result = ObjectReferenceEpisodeResult(
                run_id=run_id,
                case_id=case.case_id,
                scene_id=case.scene_id,
                question=case.question,
                pipeline_mode='synthetic',
                episode_status='runtime_failure',
                parser_mode=None,
                task_specification={},
                requested_classes=(),
                stage_evidence=evidence,
                primary_failure_category='protocol_failure',
                failure_subtype='runtime_exception',
                failure_detail=str(error),
                episode_duration_sec=perf_counter() - started,
                evidence_directory=str(directory),
                final_response_logged=True,
            )
        classification = classify_primary_failure(result.stage_evidence)
        if (
            result.primary_failure_category is None
            and classification.category is not None
        ):
            result = replace(
                result,
                primary_failure_category=classification.category,
                failure_subtype=classification.subtype,
                failure_detail=classification.detail,
            )
        return result

    def _write_metadata(self, directory, run_id, mode, cases):
        metadata = {
            'run_id': run_id,
            'mode': mode,
            'manifest_digest': manifest_digest(self._cases),
            'selected_case_ids': [item.case_id for item in cases],
            'case_count': len(cases),
        }
        _atomic_json(directory / 'run_metadata.json', metadata)
        _atomic_json(
            directory / 'manifest.json',
            [asdict(item) for item in cases],
        )


def summarize_results(
    run_id: str,
    mode: str,
    cases: tuple[ObjectReferenceCase, ...],
    results: tuple[ObjectReferenceEpisodeResult, ...],
) -> ObjectReferenceBenchmarkSummary:
    """Aggregate results without hiding unavailable correctness labels."""
    if len(cases) != len(results):
        raise ValueError('cases and results must have equal length')
    by_case = {item.case_id: item for item in cases}
    if set(by_case) != {item.case_id for item in results}:
        raise ValueError('result case IDs do not match scheduled cases')
    failures = Counter(
        item.primary_failure_category
        for item in results if item.primary_failure_category is not None
    )
    subtypes = Counter(
        item.failure_subtype for item in results if item.failure_subtype
    )
    tag_counts = Counter()
    tag_successes = Counter()
    for result in results:
        for tag in by_case[result.case_id].tags:
            tag_counts[tag] += 1
            if result.success is True:
                tag_successes[tag] += 1
    durations = [item.episode_duration_sec for item in results]
    return ObjectReferenceBenchmarkSummary(
        run_id=run_id,
        mode=mode,
        manifest_digest=manifest_digest(cases),
        scheduled_cases=len(cases),
        terminal_records=len(results),
        final_responses_logged=sum(
            item.final_response_logged for item in results
        ),
        markers_published=sum(item.marker_published for item in results),
        protocol_valid=sum(
            item.stage_evidence.protocol_valid is True for item in results
        ),
        target_labels_available=sum(
            item.target_selection_correct is not None for item in results
        ),
        target_selections_correct=sum(
            item.target_selection_correct is True for item in results
        ),
        marker_labels_available=sum(
            item.marker_success is not None for item in results
        ),
        marker_successes=sum(item.marker_success is True for item in results),
        targeted_viewpoints_used=sum(
            item.targeted_viewpoint_used for item in results
        ),
        total_proxy_score=sum(item.proxy_score for item in results),
        maximum_proxy_score=2.0 * len(results),
        mean_duration_sec=(sum(durations) / len(durations) if durations else 0.0),
        failure_counts=dict(sorted(failures.items())),
        failure_subtype_counts=dict(sorted(subtypes.items())),
        tag_counts=dict(sorted(tag_counts.items())),
        tag_success_counts=dict(sorted(tag_successes.items())),
    )


def write_episode_result(
    path: Path,
    result: ObjectReferenceEpisodeResult,
) -> None:
    """Atomically persist one live or offline terminal episode record."""
    if not isinstance(result, ObjectReferenceEpisodeResult):
        raise TypeError('result must be ObjectReferenceEpisodeResult')
    _atomic_json(Path(path), result.to_dict())


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_result(path: Path) -> ObjectReferenceEpisodeResult:
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    data['stage_evidence'] = StageEvidence(**data['stage_evidence'])
    for name in (
        'requested_classes', 'lifting_results', 'fusion_events',
        'ranked_target_ids', 'ranked_target_scores',
        'ranked_score_components', 'unresolved_constraints',
        'marker_validation_errors', 'targeted_viewpoint_pose',
    ):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return ObjectReferenceEpisodeResult(**data)


def _run_id(mode: str) -> str:
    return f'{mode}_{uuid4().hex[:12]}'


__all__ = [
    'CaseExecutor',
    'ObjectReferenceBenchmarkRun',
    'ObjectReferenceBenchmarkRunner',
    'ObjectReferenceBenchmarkSummary',
    'summarize_results',
    'write_episode_result',
]
