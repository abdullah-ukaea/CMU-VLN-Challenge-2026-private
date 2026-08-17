"""Tests for evaluation data, reasoning, metrics, and benchmark orchestration."""

import csv
import json
from math import pi
from pathlib import Path
from zipfile import ZipFile

import pytest

from qmapnav.evaluation import build_object_reference_manifest
from qmapnav.evaluation import ColourAttribute
from qmapnav.evaluation import DatasetLoadError
from qmapnav.evaluation import ground_truth_to_json
from qmapnav.evaluation import load_ascii_trajectory_ply
from qmapnav.evaluation import load_oracle_scene
from qmapnav.evaluation import load_questions
from qmapnav.evaluation import load_unity_object_list
from qmapnav.evaluation import load_unity_scene_objects
from qmapnav.evaluation import load_vla3d_objects
from qmapnav.evaluation import load_vla3d_regions
from qmapnav.evaluation import load_vla3d_relations
from qmapnav.evaluation import merge_unity_and_vla_objects
from qmapnav.evaluation import ObjectReferenceBenchmarkRunner
from qmapnav.evaluation import ObjectReferenceEpisodeResult
from qmapnav.evaluation import OracleObject
from qmapnav.evaluation import OracleRelation
from qmapnav.evaluation import OracleScene
from qmapnav.evaluation import OracleTrajectory
from qmapnav.evaluation import QuestionRecord
from qmapnav.evaluation import StageEvidence
from qmapnav.evaluation.benchmark_runner import (
    evaluate_question as _evaluate_question_impl,
)
from qmapnav.evaluation.benchmark_runner import (
    evaluate_relation_geometry as _evaluate_relation_geometry_impl,
)
from qmapnav.evaluation.benchmark_runner import (
    run_benchmark_on_scenes as _run_benchmark_on_scenes_impl,
)
from qmapnav.evaluation.benchmark_runner import (
    write_benchmark_reports as _write_benchmark_reports_impl,
)
from qmapnav.evaluation.metrics import count_accuracy_metric
from qmapnav.evaluation.metrics import forbidden_region_metrics
from qmapnav.evaluation.metrics import object_selection_metric
from qmapnav.evaluation.metrics import relation_metrics
from qmapnav.evaluation.metrics import semantic_route_metric
from qmapnav.evaluation.metrics import terminal_goal_distance
from qmapnav.evaluation.metrics import TimingMetric
from qmapnav.evaluation.oracle import OracleReasoningError
from qmapnav.evaluation.oracle import solve_numerical
from qmapnav.evaluation.oracle import solve_object_reference
from qmapnav.evaluation.oracle_route_planner import build_planning_grid
from qmapnav.evaluation.oracle_route_planner import OraclePlannerConfig
from qmapnav.evaluation.oracle_route_planner import plan_semantic_route
from qmapnav.evaluation.oracle_route_planner import RoutePlanningError
from qmapnav.language import parse_question
from qmapnav.reasoning import make_approach_region
from qmapnav.reasoning import make_between_gate
from qmapnav.reasoning import make_near_region
from qmapnav.reasoning import object_footprint
from qmapnav.reasoning import Polygon2D
from qmapnav.reasoning import SemanticRegion

FIXTURE_ROOT = Path(__file__).parent / 'fixtures'
FIXTURE = FIXTURE_ROOT / 'released_questions.json'

VLA_OBJECT_FIELDS = (
    'object_id',
    'region_id',
    'raw_label',
    'nyu_id',
    'nyu40_id',
    'nyu_label',
    'nyu40_label',
    'object_bbox_cx',
    'object_bbox_cy',
    'object_bbox_cz',
    'object_bbox_xlength',
    'object_bbox_ylength',
    'object_bbox_zlength',
    'object_bbox_heading',
    'object_front_heading',
    'object_color_r1',
    'object_color_g1',
    'object_color_b1',
    'object_color_scheme1',
    'object_color_scheme_percentage1',
    'object_color_scheme_average_dist1',
    'object_color_r2',
    'object_color_g2',
    'object_color_b2',
    'object_color_scheme2',
    'object_color_scheme_percentage2',
    'object_color_scheme_average_dist2',
    'object_color_r3',
    'object_color_g3',
    'object_color_b3',
    'object_color_scheme3',
    'object_color_scheme_percentage3',
    'object_color_scheme_average_dist3',
)
VLA_REGION_FIELDS = (
    'region_id',
    'region_label',
    'region_bbox_cx',
    'region_bbox_cy',
    'region_bbox_cz',
    'region_bbox_xlength',
    'region_bbox_ylength',
    'region_bbox_zlength',
    'region_bbox_heading',
)


def _benchmark_object(
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
            _benchmark_object('pillow', 'pillow', -2.0, -2.0, colour='red'),
            _benchmark_object('chair_far', 'chair', -3.0, 2.0, colour='blue'),
            _benchmark_object('chair_near', 'chair', 3.0, 2.0, colour='blue'),
            _benchmark_object('plant', 'potted_plant', 1.0, 2.0),
            _benchmark_object('window', 'window', 4.0, 2.0),
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

    result = _evaluate_question_impl(scene, unlabeled)

    assert result.structural_success
    assert not result.answer_label_available
    assert result.proxy_score is None
    assert 'ANSWER_MAPPING_MISSING' in result.failure_categories


def test_full_benchmark_scores_all_three_task_types_and_writes_reports(
    tmp_path: Path,
) -> None:
    scene = _benchmark_scene()
    question_ids = tuple(item.question_id for item in scene.questions)

    result = _run_benchmark_on_scenes_impl(
        (scene,),
        mode='full',
        question_ids=question_ids,
    )
    _write_benchmark_reports_impl(result, tmp_path)

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
        item.relation: item
        for item in _evaluate_relation_geometry_impl((scene,))
    }

    near = reports['near']
    assert near.support == 1
    assert near.true_positive == 1
    assert near.false_negative == 0
    assert near.sample_count == 2


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


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _vla_object_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(VLA_OBJECT_FIELDS, '_')
    row.update(
        {
            'object_id': '7',
            'region_id': '0',
            'raw_label': 'garbage can',
            'nyu_id': '12',
            'nyu40_id': '40',
            'nyu_label': 'garbage bin',
            'nyu40_label': 'otherprop',
            'object_bbox_cx': '1.0',
            'object_bbox_cy': '2.0',
            'object_bbox_cz': '0.5',
            'object_bbox_xlength': '0.4',
            'object_bbox_ylength': '0.5',
            'object_bbox_zlength': '1.0',
            'object_bbox_heading': '3.5',
            'object_color_r1': '105',
            'object_color_g1': '105',
            'object_color_b1': '105',
            'object_color_scheme1': 'gray',
            'object_color_scheme_percentage1': '0.75',
            'object_color_scheme_average_dist1': '3.25',
            'object_color_r2': '0',
            'object_color_g2': '0',
            'object_color_b2': '0',
            'object_color_scheme2': 'black',
            'object_color_scheme_percentage2': '0.25',
            'object_color_scheme_average_dist2': '0.0',
        }
    )
    row.update(overrides)
    return row


def _write_minimal_questions(root: Path) -> Path:
    scene_root = root / 'demo_scene'
    scene_root.mkdir(parents=True)
    (scene_root / 'questions.pdf').write_bytes(b'%PDF-1.4\n')
    (scene_root / 'trajectory_q3.ply').write_text(
        'ply\n'
        'format ascii 1.0\n'
        'element vertex 2\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'end_header\n'
        '0 0 0.75\n'
        '1 2 0.75\n',
        encoding='utf-8',
    )
    questions = [
        {
            'scene': 'demo_scene',
            'questions': {
                'numerical': ['How many chairs are there?'],
                'object_reference': ['Find the chair.'],
                'instruction_following': ['Go to the chair.'],
            },
        }
    ]
    path = root / 'questions.json'
    path.write_text(json.dumps(questions), encoding='utf-8')
    return path


def test_all_released_questions_receive_stable_ids() -> None:
    records = load_questions(FIXTURE_ROOT / 'released_questions.json')

    assert len(records) == 75
    assert len({item.question_id for item in records}) == 75
    assert records[0].question_id == 'arabic_room_numerical_01'
    assert records[-1].question_id == 'studio_instruction_following_02'
    assert all(item.answer_provenance == 'unavailable' for item in records)


def test_question_loader_links_answers_and_trajectory(tmp_path: Path) -> None:
    questions_path = _write_minimal_questions(tmp_path)
    answers_path = tmp_path / 'answers.json'
    answers_path.write_text(
        json.dumps(
            {
                'answers': {
                    'demo_scene_numerical_01': {'expected_count': 3},
                    'demo_scene_object_reference_01': {
                        'expected_object_id': 'chair_7'
                    },
                }
            }
        ),
        encoding='utf-8',
    )

    records = load_questions(
        questions_path,
        answers_path=answers_path,
        require_released_distribution=False,
    )

    assert records[0].expected_count == 3
    assert records[0].answer_provenance == 'machine_readable'
    assert records[1].expected_object_id == 'chair_7'
    assert records[2].trajectory_path.name == 'trajectory_q3.ply'
    assert records[2].answer_provenance == 'visualization_only'
    assert [item.answer_visualization_index for item in records] == [1, 2, 3]


def test_question_loader_rejects_unknown_answer_mapping(tmp_path: Path) -> None:
    questions_path = _write_minimal_questions(tmp_path)
    answers_path = tmp_path / 'answers.json'
    answers_path.write_text(
        json.dumps({'unknown_question': {'expected_count': 1}}),
        encoding='utf-8',
    )

    with pytest.raises(DatasetLoadError, match='unknown questions'):
        load_questions(
            questions_path,
            answers_path=answers_path,
            require_released_distribution=False,
        )


def test_unity_object_list_handles_quoted_multiword_labels(tmp_path: Path) -> None:
    path = tmp_path / 'object_list.txt'
    path.write_text(
        '7 1.0 2.0 0.5 0.4 0.5 1.0 3.5 "garbage can"\n',
        encoding='utf-8',
    )

    objects = load_unity_object_list(path)

    assert len(objects) == 1
    assert objects[0].object_id == '7'
    assert objects[0].class_name == 'trash_can'
    assert objects[0].raw_class_name == 'garbage can'
    assert -3.1416 < objects[0].yaw < 3.1416


def test_unity_objects_load_directly_from_scene_zip(tmp_path: Path) -> None:
    archive = tmp_path / 'demo_scene.zip'
    with ZipFile(archive, mode='w') as zip_file:
        zip_file.writestr(
            'demo_scene/object_list.txt',
            '1 0 0 1 1 2 3 0 "table"\n',
        )

    objects, source_archive = load_unity_scene_objects(tmp_path, 'demo_scene')

    assert [item.class_name for item in objects] == ['table']
    assert source_archive == archive


def test_unity_loader_rejects_nonpositive_dimensions(tmp_path: Path) -> None:
    path = tmp_path / 'object_list.txt'
    path.write_text('1 0 0 1 1 0 3 0 "table"\n', encoding='utf-8')

    with pytest.raises(DatasetLoadError, match='positive'):
        load_unity_object_list(path)


def test_vla_object_loader_normalizes_colours_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / 'demo_object_result.csv'
    _write_csv(path, VLA_OBJECT_FIELDS, [_vla_object_row()])

    objects = load_vla3d_objects(path)

    assert len(objects) == 1
    assert objects[0].class_name == 'trash_can'
    assert objects[0].region_id == '0'
    assert objects[0].nyu_label == 'garbage bin'
    assert [item.label for item in objects[0].colours] == ['grey', 'black']
    assert objects[0].colours[0].rgb == (105, 105, 105)
    assert objects[0].colours[0].proportion == 0.75


def test_vla_object_loader_rejects_invalid_rgb(tmp_path: Path) -> None:
    path = tmp_path / 'demo_object_result.csv'
    _write_csv(
        path,
        VLA_OBJECT_FIELDS,
        [_vla_object_row(object_color_r1='256')],
    )

    with pytest.raises(DatasetLoadError, match=r'\[0, 255\]'):
        load_vla3d_objects(path)


def test_vla_region_loader_normalizes_region(tmp_path: Path) -> None:
    path = tmp_path / 'demo_region_result.csv'
    _write_csv(
        path,
        VLA_REGION_FIELDS,
        [
            {
                'region_id': '0',
                'region_label': 'Living Room',
                'region_bbox_cx': '1',
                'region_bbox_cy': '2',
                'region_bbox_cz': '1.5',
                'region_bbox_xlength': '4',
                'region_bbox_ylength': '5',
                'region_bbox_zlength': '3',
                'region_bbox_heading': '0',
            }
        ],
    )

    regions = load_vla3d_regions(path)

    assert len(regions) == 1
    assert regions[0].label == 'living_room'
    assert regions[0].dimensions_xyz == (4.0, 5.0, 3.0)


def test_scene_graph_flattens_binary_and_between_relations(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'demo_scene_graph.json'
    path.write_text(
        json.dumps(
            {
                'scene_name': 'demo_scene',
                'regions': {
                    '0': {
                        'relationships': {
                            'above': {'1': ['2']},
                            'beside': {'2': ['1']},
                            'between': {'3': [['1', '2']]},
                            'closest': {'1': []},
                            'farthest': {'1': []},
                            'hanging_on': {'2': []},
                            'in': {'1': []},
                            'near': {'1': []},
                            'on': {'1': []},
                            'below': {'1': []},
                        }
                    }
                },
            }
        ),
        encoding='utf-8',
    )

    relations = load_vla3d_relations(
        path,
        known_object_ids={'1', '2', '3'},
        expected_scene_id='demo_scene',
    )

    assert relations == (
        OracleRelation('above', '1', ('2',), '0'),
        OracleRelation('between', '3', ('1', '2'), '0'),
        OracleRelation('near', '2', ('1',), '0'),
    )


def test_scene_graph_rejects_unknown_object_reference(tmp_path: Path) -> None:
    path = tmp_path / 'demo_scene_graph.json'
    path.write_text(
        json.dumps(
            {
                'scene_name': 'demo_scene',
                'regions': {
                    '0': {
                        'relationships': {
                            'above': {'1': ['missing']},
                        }
                    }
                },
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(DatasetLoadError, match='unknown object IDs'):
        load_vla3d_relations(path, known_object_ids={'1'})


def test_ascii_trajectory_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / 'trajectory.ply'
    path.write_text(
        'ply\n'
        'format ascii 1.0\n'
        'element vertex 3\n'
        'property float z\n'
        'property double x\n'
        'property float y\n'
        'end_header\n'
        '0.75 0 0\n'
        '0.75 1 2\n'
        '0.75 3 4\n',
        encoding='utf-8',
    )

    trajectory = load_ascii_trajectory_ply(path)

    assert trajectory.points_xyz == (
        (0.0, 0.0, 0.75),
        (1.0, 2.0, 0.75),
        (3.0, 4.0, 0.75),
    )


def test_ascii_trajectory_rejects_vertex_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / 'trajectory.ply'
    path.write_text(
        'ply\n'
        'format ascii 1.0\n'
        'element vertex 2\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'end_header\n'
        '0 0 0.75\n',
        encoding='utf-8',
    )

    with pytest.raises(DatasetLoadError, match='declared 2 vertices'):
        load_ascii_trajectory_ply(path)


def test_merge_uses_vla_attributes_after_id_and_class_validation() -> None:
    unity = (
        OracleObject(
            '7',
            'trash_can',
            (1.0, 2.0, 0.5),
            (0.4, 0.5, 1.0),
            0.0,
            raw_class_name='garbage can',
            sources=('unity_object_list',),
        ),
    )
    vla = (
        OracleObject(
            '7',
            'trash_can',
            (1.1, 2.0, 0.5),
            (0.45, 0.55, 1.0),
            0.1,
            colours=(ColourAttribute('black', (0, 0, 0), 1.0),),
            region_id='0',
            raw_class_name='garbage can',
            sources=('vla3d_object_result',),
        ),
    )

    merged = merge_unity_and_vla_objects(unity, vla)

    assert merged[0].centre_xyz == vla[0].centre_xyz
    assert merged[0].colours == vla[0].colours
    assert merged[0].sources == ('unity_object_list', 'vla3d_object_result')


def test_merge_can_enforce_geometry_tolerance() -> None:
    unity = (
        OracleObject('1', 'chair', (0, 0, 0), (1, 1, 1), 0),
    )
    vla = (
        OracleObject('1', 'chair', (0.2, 0, 0), (1, 1, 1), 0),
    )

    with pytest.raises(DatasetLoadError, match='centre differs'):
        merge_unity_and_vla_objects(unity, vla, geometry_tolerance=0.05)


def test_complete_scene_loads_through_adapter_boundary(tmp_path: Path) -> None:
    questions_path = _write_minimal_questions(tmp_path / 'questions')
    questions = load_questions(
        questions_path,
        require_released_distribution=False,
    )
    simulation_root = tmp_path / 'simulation'
    unity_scene = simulation_root / 'demo_scene'
    unity_scene.mkdir(parents=True)
    (unity_scene / 'object_list.txt').write_text(
        '7 1 2 0.5 0.4 0.5 1 0 "garbage can"\n'
        '8 2 2 0.5 0.5 0.5 1 0 "chair"\n',
        encoding='utf-8',
    )
    vla_scene = tmp_path / 'vla' / 'Unity' / 'demo_scene'
    vla_scene.mkdir(parents=True)
    _write_csv(
        vla_scene / 'demo_scene_object_result.csv',
        VLA_OBJECT_FIELDS,
        [
            _vla_object_row(object_bbox_heading='0'),
            _vla_object_row(
                object_id='8',
                raw_label='chair',
                nyu_label='chair',
                nyu40_label='chair',
                object_bbox_cx='2',
                object_bbox_heading='0',
            ),
        ],
    )
    _write_csv(
        vla_scene / 'demo_scene_region_result.csv',
        VLA_REGION_FIELDS,
        [
            {
                'region_id': '0',
                'region_label': 'office',
                'region_bbox_cx': '1',
                'region_bbox_cy': '1',
                'region_bbox_cz': '1.5',
                'region_bbox_xlength': '4',
                'region_bbox_ylength': '4',
                'region_bbox_zlength': '3',
                'region_bbox_heading': '0',
            }
        ],
    )
    (vla_scene / 'demo_scene_scene_graph.json').write_text(
        json.dumps(
            {
                'scene_name': 'demo_scene',
                'regions': {
                    '0': {
                        'relationships': {
                            'near': {'7': ['8']},
                        }
                    }
                },
            }
        ),
        encoding='utf-8',
    )

    scene = load_oracle_scene(
        'demo_scene',
        questions=questions,
        simulation_root=simulation_root,
        vla_root=tmp_path / 'vla',
    )

    assert len(scene.objects) == 2
    assert scene.objects[0].colours[0].label == 'grey'
    assert scene.regions[0].label == 'office'
    assert scene.relations == (OracleRelation('near', '7', ('8',), '0'),)
    assert len(scene.questions) == 3
    assert len(scene.trajectories) == 1
    assert scene.trajectory_for('demo_scene_instruction_following_01').points_xyz


def test_ground_truth_json_serialization_is_deterministic() -> None:
    colour = ColourAttribute('blue', (0, 0, 255), 0.8, 1.25)

    first = ground_truth_to_json(colour)
    second = ground_truth_to_json(colour)

    assert first == second
    assert json.loads(first) == {
        'average_lab_distance': 1.25,
        'label': 'blue',
        'proportion': 0.8,
        'rgb': [0, 0, 255],
    }


def _object(
    object_id: str,
    class_name: str,
    x: float,
    y: float,
    *,
    dimensions: tuple[float, float, float] = (0.5, 0.5, 1.0),
    yaw: float = 0.0,
    colour: str | None = None,
) -> OracleObject:
    colours = ()
    if colour is not None:
        colours = (
            ColourAttribute(
                label=colour,
                rgb=(128, 128, 128),
                proportion=1.0,
            ),
        )
    return OracleObject(
        object_id=object_id,
        class_name=class_name,
        centre_xyz=(x, y, dimensions[2] / 2.0),
        dimensions_xyz=dimensions,
        yaw=yaw,
        colours=colours,
    )


def _scene(
    objects: tuple[OracleObject, ...],
    relations: tuple[OracleRelation, ...] = (),
) -> OracleScene:
    return OracleScene(
        scene_id='test_scene',
        objects=objects,
        relations=relations,
        regions=(),
        questions=(),
    )


def test_object_footprint_preserves_oriented_box_geometry() -> None:
    obj = _object(
        'table',
        'table',
        1.0,
        2.0,
        dimensions=(2.0, 1.0, 0.8),
        yaw=pi / 2.0,
    )

    footprint = object_footprint(obj)

    assert footprint.centre == pytest.approx((1.0, 2.0))
    assert footprint.bounds == pytest.approx((0.5, 1.0, 1.5, 3.0))
    assert footprint.contains((1.0, 2.9))
    assert not footprint.contains((1.6, 2.0))


def test_near_and_approach_regions_exclude_object_interior() -> None:
    obj = _object('sofa', 'sofa', 0.0, 0.0, dimensions=(2.0, 1.0, 1.0))

    near = make_near_region(obj, min_distance=0.5, max_distance=1.5)
    approach = make_approach_region(
        obj,
        minimum_clearance=0.4,
        maximum_distance=1.0,
    )

    assert near.contains((2.0, 0.0))
    assert not near.contains((0.0, 0.0))
    assert not near.contains((3.0, 0.0))
    assert approach.contains((1.8, 0.0))
    assert not approach.contains((1.2, 0.0))


def test_between_gate_reports_clearance_and_invalid_narrow_gap() -> None:
    left = _object('left', 'table', 0.0, 0.0, dimensions=(1.0, 1.0, 1.0))
    right = _object('right', 'table', 2.0, 0.0, dimensions=(1.0, 1.0, 1.0))
    narrow = _object(
        'narrow',
        'table',
        1.4,
        0.0,
        dimensions=(1.0, 1.0, 1.0),
    )

    valid = make_between_gate(left, right, robot_diameter=0.6)
    invalid = make_between_gate(left, narrow, robot_diameter=0.6)

    assert valid.valid
    assert valid.gap_width == pytest.approx(1.0)
    assert valid.region is not None
    assert valid.region.contains((1.0, 0.0))
    assert not invalid.valid
    assert invalid.region is None
    assert 'narrower' in invalid.reason


def test_numerical_solver_propagates_inbound_support_relation() -> None:
    chair_1 = _object('1', 'chair', 0.0, 0.0)
    chair_2 = _object('2', 'chair', 2.0, 0.0)
    pillow = _object('3', 'pillow', 0.0, 0.0)
    scene = _scene(
        (chair_1, chair_2, pillow),
        (OracleRelation('on', '3', ('1',)),),
    )
    task = parse_question('Count the number of chairs with pillows on them.')

    result = solve_numerical(task, scene)

    assert result.count == 1
    assert result.matching_object_ids == ('1',)
    decisions = {item.object_id: item for item in result.candidate_decisions}
    assert decisions['1'].accepted
    assert not decisions['2'].accepted
    assert 'failed on' in decisions['2'].reasons[0]


def test_numerical_solver_filters_dominant_colour() -> None:
    scene = _scene(
        (
            _object('1', 'pillow', 0.0, 0.0, colour='red'),
            _object('2', 'pillow', 1.0, 0.0, colour='blue'),
        )
    )
    task = parse_question('How many red pillows are there?')

    result = solve_numerical(task, scene)

    assert result.count == 1
    assert result.matching_object_ids == ('1',)
    rejected = result.candidate_decisions[1]
    assert rejected.reasons == ('colour is not red',)


def test_colour_solver_recovers_released_maroon_and_dark_rgb_aliases() -> None:
    maroon = OracleObject(
        object_id='1',
        class_name='pillow',
        centre_xyz=(0.0, 0.0, 0.5),
        dimensions_xyz=(0.5, 0.5, 0.5),
        yaw=0.0,
        colours=(ColourAttribute('maroon', (178, 34, 34), 1.0),),
    )
    dark_grey = OracleObject(
        object_id='2',
        class_name='pillow',
        centre_xyz=(1.0, 0.0, 0.5),
        dimensions_xyz=(0.5, 0.5, 0.5),
        yaw=0.0,
        colours=(ColourAttribute('grey', (47, 79, 79), 1.0),),
    )
    scene = _scene((maroon, dark_grey))

    red = solve_numerical(parse_question('How many red pillows are there?'), scene)
    black = solve_numerical(
        parse_question('How many black pillows are there?'),
        scene,
    )

    assert red.matching_object_ids == ('1',)
    assert black.matching_object_ids == ('2',)


def test_object_solver_applies_between_then_closest_ranking() -> None:
    objects = (
        _object('1', 'chair', 0.0, 0.0, colour='orange'),
        _object('2', 'chair', 5.0, 0.0, colour='orange'),
        _object('3', 'table', 4.0, -1.0),
        _object('4', 'sink', 4.0, 1.0),
        _object('5', 'window', 5.2, 0.0),
    )
    scene = _scene(
        objects,
        (OracleRelation('between', '2', ('3', '4')),),
    )
    task = parse_question(
        'Find the orange chair between the table and sink that is closest '
        'to the window.'
    )

    result = solve_object_reference(task, scene)

    assert result.selected_object_id == '2'
    assert result.selected_object == objects[1]
    assert result.confidence_margin == 1.0
    assert result.warnings == ()


def test_object_solver_reports_ambiguity_and_uses_stable_id() -> None:
    scene = _scene(
        (
            _object('10', 'vase', 0.0, 0.0),
            _object('2', 'vase', 1.0, 0.0),
        )
    )

    result = solve_object_reference(parse_question('Find the vase.'), scene)

    assert result.selected_object_id == '2'
    assert result.confidence_margin == 0.0
    assert 'found 2 candidates' in result.warnings[-1]


def test_oracle_solvers_reject_wrong_task_type() -> None:
    scene = _scene((_object('1', 'chair', 0.0, 0.0),))
    task = parse_question('Find the chair.')

    with pytest.raises(OracleReasoningError, match='numerical task'):
        solve_numerical(task, scene)


def test_grid_inflates_objects_and_preserves_free_space() -> None:
    scene = _scene((_object('1', 'table', 1.0, 0.0),))
    config = OraclePlannerConfig(resolution=0.2, robot_radius=0.3)

    grid = build_planning_grid(scene, (0.0, 0.0), config)

    assert not grid.is_free(grid.point_to_cell((1.0, 0.0)))
    assert grid.is_free(grid.point_to_cell((0.0, 0.0)))


def test_semantic_route_selects_regions_in_textual_order() -> None:
    scene = _scene(
        (
            _object('1', 'plant', 2.0, 0.0),
            _object('2', 'table', 4.0, -1.0),
            _object('3', 'table', 4.0, 1.0),
            _object('4', 'window', 6.0, 0.0),
            _object('5', 'sofa', 3.0, 2.5, dimensions=(1.5, 0.8, 1.0)),
        )
    )
    task = parse_question(
        'First go near the plant, then pass between the two tables, avoid '
        'the sofa and stop near the window.'
    )

    plan = plan_semantic_route(task, scene, (0.0, 0.0))

    assert len(plan.required_regions) == 3
    assert [item.region_type for item in plan.required_regions] == [
        'near',
        'between_gate',
        'near',
    ]
    assert plan.resolved_step_object_ids == (
        (0, ('1',)),
        (1, ('2', '3')),
        (2, ('4',)),
    )
    assert [item.region_type for item in plan.forbidden_regions] == [
        'forbidden_near'
    ]
    assert all(
        plan.grid.is_free(plan.grid.point_to_cell(point))
        for point in plan.waypoints_xy
    )
    assert all(
        not plan.forbidden_regions[0].contains(point)
        for point in plan.waypoints_xy
    )


def test_route_planner_rejects_non_instruction_and_missing_candidates() -> None:
    scene = _scene((_object('1', 'chair', 0.0, 0.0),))

    with pytest.raises(RoutePlanningError, match='instruction-following'):
        plan_semantic_route(parse_question('Find the chair.'), scene, (1.0, 0.0))

    missing_task = parse_question('Go to the window.')
    with pytest.raises(RoutePlanningError, match='no oracle object candidates'):
        plan_semantic_route(missing_task, scene, (1.0, 0.0))


def test_route_planner_reports_unreachable_inferred_between_pair() -> None:
    scene = _scene(
        (
            _object('1', 'table', 1.0, 0.0, dimensions=(1.0, 1.0, 1.0)),
            _object('2', 'table', 1.5, 0.0, dimensions=(1.0, 1.0, 1.0)),
            _object('3', 'window', 3.0, 0.0),
        )
    )
    task = parse_question(
        'Pass between the two tables and stop at the window.'
    )

    with pytest.raises(RoutePlanningError, match='route step 0'):
        plan_semantic_route(task, scene, (0.0, 0.0))


def _square_region(
    region_id: str,
    centre_x: float,
    centre_y: float = 0.0,
    *,
    required: bool = True,
) -> SemanticRegion:
    half_size = 0.2
    return SemanticRegion(
        region_id=region_id,
        region_type='test',
        polygon=Polygon2D(
            (
                (centre_x - half_size, centre_y - half_size),
                (centre_x + half_size, centre_y - half_size),
                (centre_x + half_size, centre_y + half_size),
                (centre_x - half_size, centre_y + half_size),
            )
        ),
        source_object_ids=(region_id,),
        required=required,
    )


def test_object_selection_distinguishes_wrong_and_unavailable_labels() -> None:
    wrong = object_selection_metric('chair_2', 'chair_1')
    unavailable = object_selection_metric('chair_2', None)

    assert wrong.label_available
    assert wrong.correct is False
    assert not unavailable.label_available
    assert unavailable.correct is None


def test_count_accuracy_reports_exact_match_and_absolute_error() -> None:
    exact = count_accuracy_metric(3, 3)
    wrong = count_accuracy_metric(1, 3)
    unavailable = count_accuracy_metric(2, None)

    assert exact.exact_match is True
    assert exact.absolute_error == 0
    assert wrong.exact_match is False
    assert wrong.absolute_error == 2
    assert unavailable.exact_match is None
    assert unavailable.absolute_error is None


def test_relation_metric_reports_complete_confusion_matrix() -> None:
    expected = {
        ('near', 'chair_1', ('table_1',)),
        ('near', 'chair_2', ('table_1',)),
    }
    predicted = {
        ('near', 'chair_1', ('table_1',)),
        ('near', 'chair_3', ('table_1',)),
    }

    metric = relation_metrics(
        expected,
        predicted,
        relation='near',
        negatives_evaluated=2,
    )

    assert metric.true_positive == 1
    assert metric.false_positive == 1
    assert metric.false_negative == 1
    assert metric.true_negative == 1
    assert metric.precision == pytest.approx(0.5)
    assert metric.recall == pytest.approx(0.5)
    assert metric.f1 == pytest.approx(0.5)


def test_semantic_route_metric_scores_order_avoidance_and_terminal_goal() -> None:
    required = (
        _square_region('plant', 1.0),
        _square_region('gate', 2.0),
        _square_region('window', 3.0),
    )
    forbidden = (_square_region('sofa', 2.0, 2.0, required=False),)
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))

    metric = semantic_route_metric(trajectory, required, forbidden)

    assert metric.required_intersection_fraction == 1.0
    assert metric.ordered_constraints_completed == 3
    assert all(item is not None for item in metric.ordered_hit_indices)
    assert metric.order_correct
    assert metric.forbidden_violation_count == 0
    assert metric.terminal_goal_distance == 0.0
    assert metric.terminal_goal_reached
    assert metric.proxy_score == 6.0
    assert metric.success


def test_semantic_route_metric_detects_wrong_order_and_forbidden_entry() -> None:
    required = (
        _square_region('first', 2.0),
        _square_region('second', 1.0),
    )
    forbidden = (_square_region('forbidden', 1.5, required=False),)
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    metric = semantic_route_metric(trajectory, required, forbidden)

    assert metric.required_intersection_fraction == 1.0
    assert not metric.order_correct
    assert metric.ordered_hit_indices[1] is None
    assert metric.forbidden_violation_count == 1
    assert not metric.success


def test_forbidden_length_and_terminal_region_distance_are_geometric() -> None:
    forbidden = _square_region('forbidden', 1.0, required=False)
    result = forbidden_region_metrics(
        ((0.0, 0.0), (2.0, 0.0)),
        (forbidden,),
        sampling_resolution=0.01,
    )[0]

    assert result.violated
    assert result.approximate_length_inside == pytest.approx(0.4, abs=0.02)
    assert terminal_goal_distance((2.0, 0.0), forbidden) == pytest.approx(0.8)


def test_timing_rejects_negative_stage_values() -> None:
    with pytest.raises(ValueError, match='parser_seconds'):
        TimingMetric(-0.1, 0.0, 0.0, None, 0.0)
