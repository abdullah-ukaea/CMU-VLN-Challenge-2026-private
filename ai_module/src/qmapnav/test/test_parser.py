"""Regression tests for full and degraded deterministic parsing."""

import json
from pathlib import Path

import pytest

from qmapnav.common import TaskSpecification
from qmapnav.language import FullParseError
from qmapnav.language import INSTRUCTION_FOLLOWING
from qmapnav.language import NUMERICAL
from qmapnav.language import OBJECT_REFERENCE
from qmapnav.language import parse_question
from qmapnav.language import parse_question_degraded
from qmapnav.language import parse_question_full


QUESTION_PATH = Path(__file__).parent / 'fixtures' / 'released_questions.json'


def _released_questions() -> list[tuple[str, str, str]]:
    records = json.loads(QUESTION_PATH.read_text(encoding='utf-8'))
    return [
        (record['scene'], task_type, question)
        for record in records
        for task_type, questions in record['questions'].items()
        for question in questions
    ]


RELEASED_QUESTIONS = _released_questions()
RELEASED_INSTRUCTIONS = [
    question
    for _, task_type, question in RELEASED_QUESTIONS
    if task_type == INSTRUCTION_FOLLOWING
]


@pytest.mark.parametrize(
    ('scene', 'expected_type', 'question'),
    RELEASED_QUESTIONS,
)
def test_every_released_question_has_a_full_task_specification(
    scene: str,
    expected_type: str,
    question: str,
) -> None:
    task = parse_question(question)

    assert isinstance(task, TaskSpecification), scene
    assert task.task_type == expected_type, scene
    assert task.parse_mode == 'full', scene
    assert task.parse_confidence == 1.0, scene
    assert task.entities, scene
    assert len({entity.entity_id for entity in task.entities}) == len(
        task.entities
    )
    assert task == parse_question(question)

    if expected_type in {NUMERICAL, OBJECT_REFERENCE}:
        assert task.ordered_route_steps == []
        assert task.forbidden_constraints == []
        assert task.terminal_target is None
    else:
        assert task.ordered_route_steps
        assert task.terminal_target is not None
        assert [step.step_index for step in task.ordered_route_steps] == list(
            range(len(task.ordered_route_steps))
        )


EXPECTED_INSTRUCTION_ROUTES = [
    (['go_near', 'stop_at'], 'small_table'),
    (['go_to', 'pass_between', 'stop_at'], 'tray'),
    (['go_near', 'stop_at'], 'painting'),
    (['go_near', 'stop_at'], 'table'),
    (['go_to', 'stop_at'], 'dining_table'),
    (['go_to', 'pass_between', 'stop_at'], 'trash_can'),
    (['go_near', 'go_to'], 'potted_plant'),
    (['pass_between', 'go_to', 'go_to'], 'potted_plant'),
    (['go_to', 'stop_at'], 'chair'),
    (['go_near', 'pass_between', 'go_to'], 'picture'),
    (['pass_between', 'stop_at'], 'lamp'),
    (['go_to', 'pass_between', 'stop_at'], 'curtain'),
    (['go_near', 'go_to'], 'flower'),
    (['go_to', 'pass_near', 'go_to'], 'flower'),
    (['go_to', 'stop_at'], 'vase'),
    (['go_near', 'pass_between', 'stop_at'], 'cabinet'),
    (['go_to', 'go_to'], 'crystal_ball_decoration'),
    (['go_to', 'stop_at'], 'soccer_ball'),
    (['pass_near', 'go_to'], 'pillow'),
    (['go_near', 'pass_near', 'stop_at'], 'bowl'),
    (['go_near', 'stop_at'], 'table'),
    (['go_near', 'go_to', 'stop_at'], 'chair'),
    (['go_to'], 'cup'),
    (['go_near', 'pass_by', 'stop_at'], 'sphere_decoration'),
    (['go_to', 'stop_at'], 'water_cooler'),
    (['go_near', 'pass_between', 'stop_at'], 'bench'),
    (['go_near', 'stop_at'], 'window'),
    (['go_to', 'go_to', 'go_to'], 'door'),
    (['go_to', 'stop_at'], 'guitar'),
    (['go_to', 'pass_between', 'stop_at'], 'window'),
]


@pytest.mark.parametrize(
    ('question', 'expected_route'),
    zip(RELEASED_INSTRUCTIONS, EXPECTED_INSTRUCTION_ROUTES),
)
def test_every_released_instruction_preserves_route_and_terminal_target(
    question: str,
    expected_route: tuple[list[str], str],
) -> None:
    expected_actions, expected_terminal = expected_route
    task = parse_question_full(question)

    assert [step.action for step in task.ordered_route_steps] == expected_actions
    assert task.terminal_target is not None
    assert task.terminal_target.class_name == expected_terminal


@pytest.mark.parametrize(
    ('question', 'expected_type', 'expected_classes'),
    [
        (
            'First, go near the tea table with the elephant figurine on it, '
            'then stop at the table with the horse figurine on it, avoiding '
            'the path between the chair and the folding screen.',
            'avoid_between',
            ['chair', 'folding_screen'],
        ),
        (
            'First, go to the chair near the window, then stop at the soccer '
            'ball near the couch, avoiding the path between the TV and the '
            'tea table.',
            'avoid_between',
            ['tv', 'tea_table'],
        ),
        (
            'Go to the cup near the TV remote and avoid the path near the '
            'cabinet.',
            'avoid_near',
            ['cabinet'],
        ),
    ],
)
def test_released_avoidance_constraints_bind_expected_entities(
    question: str,
    expected_type: str,
    expected_classes: list[str],
) -> None:
    task = parse_question_full(question)
    constraint = task.forbidden_constraints[0]
    classes_by_id = {entity.entity_id: entity.class_name for entity in task.entities}

    assert constraint.constraint_type == expected_type
    assert [classes_by_id[item] for item in constraint.entity_ids] == (
        expected_classes
    )


def test_full_parser_binds_attributes_cardinality_and_between_relation() -> None:
    task = parse_question_full(
        'How many blue chairs are between the table and the wall?'
    )
    entities = {entity.class_name: entity for entity in task.entities}
    relation = task.relations[0]

    assert entities['chair'].attributes == {'colour': 'blue'}
    assert relation.relation == 'between'
    assert relation.subject_entity_id == entities['chair'].entity_id
    assert relation.anchor_entity_ids == [
        entities['table'].entity_id,
        entities['wall'].entity_id,
    ]


def test_full_parser_binds_multiple_attributes_to_the_correct_entities() -> None:
    task = parse_question_full(
        'Find the large round wooden table beside the small metal chair.'
    )
    entities = {entity.class_name: entity for entity in task.entities}

    assert entities['table'].attributes == {
        'size': 'large',
        'shape': 'round',
        'material': 'wood',
    }
    assert entities['chair'].attributes == {
        'size': 'small',
        'material': 'metal',
    }


def test_degraded_parser_preserves_non_colour_attributes() -> None:
    task = parse_question_degraded('Seek the tiny leather chair.')

    chair = next(entity for entity in task.entities if entity.class_name == 'chair')
    assert chair.attributes == {'size': 'small', 'material': 'leather'}


def test_full_parser_resolves_pronoun_support_relation() -> None:
    task = parse_question_full(
        'Count the number of chairs with pillows on them.'
    )
    entities = {entity.class_name: entity for entity in task.entities}

    assert task.relations[0].relation == 'on'
    assert task.relations[0].subject_entity_id == entities['pillow'].entity_id
    assert task.relations[0].anchor_entity_ids == [entities['chair'].entity_id]


def test_pronoun_anchor_precedes_entity_from_later_route_step() -> None:
    task = parse_question_full(
        'Go to the coffee table with the kettle on it and stop at the dining '
        'table near the big picture.'
    )
    coffee_table = next(
        entity for entity in task.entities if entity.class_name == 'coffee_table'
    )
    kettle = next(
        entity for entity in task.entities if entity.class_name == 'kettle'
    )
    support = next(
        relation
        for relation in task.relations
        if relation.subject_entity_id == kettle.entity_id
    )

    assert support.relation == 'on'
    assert support.anchor_entity_ids == [coffee_table.entity_id]


def test_full_parser_preserves_pair_cardinality_in_route_anchor() -> None:
    task = parse_question_full(
        'First go near the plant, then pass between the two tables and stop '
        'near the window.'
    )
    table = next(entity for entity in task.entities if entity.class_name == 'table')
    between_step = next(
        step for step in task.ordered_route_steps if step.action == 'pass_between'
    )

    assert table.cardinality == 2
    assert between_step.entity_ids == [table.entity_id]


def test_full_parser_keeps_avoidance_separate_from_positive_route() -> None:
    task = parse_question_full(
        'First go near the plant, avoid the sofa and stop near the window.'
    )
    classes_by_id = {entity.entity_id: entity.class_name for entity in task.entities}

    assert [step.action for step in task.ordered_route_steps] == [
        'go_near',
        'stop_near',
    ]
    assert task.forbidden_constraints[0].constraint_type == 'avoid_near'
    assert [
        classes_by_id[item]
        for item in task.forbidden_constraints[0].entity_ids
    ] == ['sofa']
    assert task.terminal_target is not None
    assert task.terminal_target.class_name == 'window'


def test_parse_question_uses_degraded_mode_for_unsupported_grammar() -> None:
    task = parse_question('Navigate toward the television.')

    assert task.task_type == INSTRUCTION_FOLLOWING
    assert task.parse_mode == 'degraded'
    assert 0.0 < task.parse_confidence < 1.0
    assert [entity.class_name for entity in task.entities] == ['tv']
    assert task.terminal_target is not None
    assert task.terminal_target.class_name == 'tv'
    assert task.ordered_route_steps[0].entity_ids == [
        task.terminal_target.entity_id
    ]


def test_degraded_parser_skips_ungroundable_relation() -> None:
    task = parse_question_degraded('Find near the window.')

    assert task.task_type == OBJECT_REFERENCE
    assert task.parse_mode == 'degraded'
    assert [entity.class_name for entity in task.entities] == ['window']
    assert task.relations == []


def test_degraded_parser_returns_valid_schema_without_known_entities() -> None:
    task = parse_question('Identify the mysterious artifact.')

    assert task.task_type == OBJECT_REFERENCE
    assert task.parse_mode == 'degraded'
    assert task.entities == []
    assert task.parse_confidence > 0.0


def test_full_parser_reports_unsupported_grammar() -> None:
    with pytest.raises(FullParseError, match='supported task type'):
        parse_question_full('Navigate toward the television.')


@pytest.mark.parametrize('question', ['', '  \n\t '])
def test_parser_rejects_empty_questions(question: str) -> None:
    with pytest.raises(ValueError, match='empty'):
        parse_question(question)
