"""Tests for resumable Day 10 quick/full benchmark execution."""

from pathlib import Path

from qmapnav.evaluation import build_object_reference_manifest
from qmapnav.evaluation import load_questions
from qmapnav.evaluation import ObjectReferenceBenchmarkRunner
from qmapnav.evaluation import ObjectReferenceEpisodeResult
from qmapnav.evaluation import StageEvidence
from qmapnav.language import parse_question


FIXTURE = Path(__file__).parent / 'fixtures' / 'released_questions.json'


def _cases():
    return build_object_reference_manifest(load_questions(FIXTURE), parse_question)


def _successful_executor(case, run_id, directory):
    del directory
    return ObjectReferenceEpisodeResult(
        run_id=run_id,
        case_id=case.case_id,
        scene_id=case.scene_id,
        question=case.question,
        pipeline_mode='synthetic',
        episode_status='completed',
        parser_mode='full',
        task_specification={},
        requested_classes=(case.expected_target_class,),
        stage_evidence=StageEvidence(
            parser_correct=True,
            target_observed=True,
            target_detected=True,
            anchors_available=True,
            target_lifted=True,
            identity_correct=True,
            colour_correct=True,
            relation_correct=True,
            target_selected_correctly=True,
            obb_acceptable=True,
            protocol_valid=True,
        ),
        selected_target_id='7',
        marker_published=True,
        marker_publish_count=1,
        marker_publish_time_sec=1.0,
        matching_waypoint_published=True,
        target_selection_correct=True,
        marker_success=True,
        success=True,
        proxy_score=2.0,
    )


def test_quick_runner_writes_six_terminal_results(tmp_path) -> None:
    runner = ObjectReferenceBenchmarkRunner(
        _cases(), _successful_executor, tmp_path
    )

    result = runner.run(mode='quick', run_id='quick_test')

    assert result.summary.scheduled_cases == 6
    assert result.summary.terminal_records == 6
    assert result.summary.final_responses_logged == 6
    assert result.summary.markers_published == 6
    assert result.summary.total_proxy_score == 12.0
    assert (tmp_path / 'quick_test' / 'summary.json').is_file()
    assert (tmp_path / 'quick_test' / 'manifest.json').is_file()


def test_full_runner_contains_executor_exception_and_logs_result(tmp_path) -> None:
    failing_id = _cases()[4].case_id

    def executor(case, run_id, directory):
        if case.case_id == failing_id:
            raise RuntimeError('injected failure')
        return _successful_executor(case, run_id, directory)

    runner = ObjectReferenceBenchmarkRunner(_cases(), executor, tmp_path)

    result = runner.run(mode='full', run_id='full_test')

    assert result.summary.scheduled_cases == 30
    assert result.summary.terminal_records == 30
    assert result.summary.final_responses_logged == 30
    assert result.summary.failure_counts == {'protocol_failure': 1}
    failed = next(item for item in result.results if item.case_id == failing_id)
    assert failed.failure_subtype == 'runtime_exception'
    assert 'injected failure' in failed.failure_detail


def test_resume_reuses_existing_terminal_results(tmp_path) -> None:
    calls = []

    def executor(case, run_id, directory):
        calls.append(case.case_id)
        return _successful_executor(case, run_id, directory)

    runner = ObjectReferenceBenchmarkRunner(_cases(), executor, tmp_path)
    first = runner.run(mode='quick', run_id='resume_test')
    second = runner.run(mode='quick', run_id='resume_test', resume=True)

    assert len(calls) == 6
    assert second.summary == first.summary
