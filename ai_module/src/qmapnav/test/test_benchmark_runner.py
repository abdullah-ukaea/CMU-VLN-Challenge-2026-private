"""Tests for quick/full oracle benchmark orchestration and JSON reports."""

import json
from pathlib import Path

from qmapnav.evaluation import ColourAttribute
from qmapnav.evaluation import OracleObject
from qmapnav.evaluation import OracleRelation
from qmapnav.evaluation import OracleScene
from qmapnav.evaluation import OracleTrajectory
from qmapnav.evaluation import QuestionRecord
from qmapnav.evaluation.benchmark_runner import evaluate_question
from qmapnav.evaluation.benchmark_runner import evaluate_relation_geometry
from qmapnav.evaluation.benchmark_runner import run_benchmark_on_scenes
from qmapnav.evaluation.benchmark_runner import write_benchmark_reports


def _object(
    object_id: str,
    class_name: str,
    x: float,
    y: float,
    *,
    colour: str | None = None,
) -> OracleObject:
    colours = ()
    if colour is not None:
        colours = (ColourAttribute(colour, (80, 80, 80), 1.0),)
    return OracleObject(
        object_id=object_id,
        class_name=class_name,
        centre_xyz=(x, y, 0.5),
        dimensions_xyz=(0.5, 0.5, 1.0),
        yaw=0.0,
        colours=colours,
    )


def _benchmark_scene() -> OracleScene:
    questions = (
        QuestionRecord(
            question_id='test_numerical_01',
            scene_id='test',
            question_text='How many red pillows are there?',
            task_type='numerical',
            question_number=1,
            task_index=1,
            expected_count=1,
            answer_provenance='machine_readable',
        ),
        QuestionRecord(
            question_id='test_object_reference_01',
            scene_id='test',
            question_text='Find the blue chair closest to the window.',
            task_type='object_reference',
            question_number=2,
            task_index=1,
            expected_object_id='chair_near',
            answer_provenance='machine_readable',
        ),
        QuestionRecord(
            question_id='test_instruction_following_01',
            scene_id='test',
            question_text=(
                'First, go near the potted plant, then stop near the window.'
            ),
            task_type='instruction_following',
            question_number=3,
            task_index=1,
            trajectory_path=Path('trajectory.ply'),
        ),
    )
    return OracleScene(
        scene_id='test',
        objects=(
            _object('pillow', 'pillow', -2.0, -2.0, colour='red'),
            _object('chair_far', 'chair', -3.0, 2.0, colour='blue'),
            _object('chair_near', 'chair', 3.0, 2.0, colour='blue'),
            _object('plant', 'potted_plant', 1.0, 2.0),
            _object('window', 'window', 4.0, 2.0),
        ),
        relations=(OracleRelation('near', 'chair_near', ('window',)),),
        regions=(),
        questions=questions,
        trajectories=(
            (
                'test_instruction_following_01',
                OracleTrajectory(
                    points_xyz=((0.0, 2.0, 0.0), (4.0, 2.0, 0.0)),
                    source_path=Path('trajectory.ply'),
                ),
            ),
        ),
    )


def test_evaluate_question_preserves_missing_answer_label() -> None:
    scene = _benchmark_scene()
    unlabeled = QuestionRecord(
        question_id='unlabeled',
        scene_id='test',
        question_text='How many red pillows are there?',
        task_type='numerical',
        question_number=4,
        task_index=2,
    )

    result = evaluate_question(scene, unlabeled)

    assert result.structural_success
    assert not result.answer_label_available
    assert result.proxy_score is None
    assert 'ANSWER_MAPPING_MISSING' in result.failure_categories


def test_full_benchmark_scores_all_three_task_types_and_writes_reports(
    tmp_path: Path,
) -> None:
    scene = _benchmark_scene()
    question_ids = tuple(item.question_id for item in scene.questions)

    result = run_benchmark_on_scenes(
        (scene,),
        mode='full',
        question_ids=question_ids,
    )
    write_benchmark_reports(result, tmp_path)

    assert result.summary.question_count == 3
    assert result.summary.structural_successes == 3
    assert result.summary.proxy_score_earned == 9.0
    assert result.summary.proxy_score_available_maximum == 9.0
    assert result.summary.normalized_available_proxy_score == 1.0
    assert result.summary.instruction_successes == 1
    assert (tmp_path / 'summary.json').is_file()
    assert (tmp_path / 'per_question.json').is_file()
    assert (tmp_path / 'relation_metrics.json').is_file()
    assert (tmp_path / 'failures.json').is_file()
    assert (
        tmp_path / 'routes' / 'test_instruction_following_01.json'
    ).is_file()
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['mode'] == 'full'
    assert summary['structural_failures'] == 0


def test_relation_geometry_uses_gt_positives_and_sampled_negatives() -> None:
    scene = _benchmark_scene()

    reports = {
        item.relation: item for item in evaluate_relation_geometry((scene,))
    }

    near = reports['near']
    assert near.support == 1
    assert near.true_positive == 1
    assert near.false_negative == 0
    assert near.sample_count == 2
