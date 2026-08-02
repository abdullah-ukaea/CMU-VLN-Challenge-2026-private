"""Tests for normalized released ground-truth loading."""

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

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
from qmapnav.evaluation import OracleObject
from qmapnav.evaluation import OracleRelation


FIXTURE_ROOT = Path(__file__).parent / 'fixtures'
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
