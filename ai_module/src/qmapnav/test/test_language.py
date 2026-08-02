"""Tests for deterministic task classification and lexical extraction."""

import json
from pathlib import Path

import pytest

from qmapnav.language import classify_task_type
from qmapnav.language import extract_language_features
from qmapnav.language import INSTRUCTION_FOLLOWING
from qmapnav.language import NUMERICAL
from qmapnav.language import OBJECT_REFERENCE


QUESTION_PATH = Path(__file__).parent / 'fixtures' / 'released_questions.json'


def _released_questions() -> list[tuple[str, str, str]]:
    records = json.loads(QUESTION_PATH.read_text(encoding='utf-8'))
    questions = []
    for record in records:
        for task_type, task_questions in record['questions'].items():
            questions.extend(
                (record['scene'], task_type, question)
                for question in task_questions
            )
    return questions


RELEASED_QUESTIONS = _released_questions()
INSTRUCTION_QUESTIONS = [
    question
    for _, task_type, question in RELEASED_QUESTIONS
    if task_type == INSTRUCTION_FOLLOWING
]


def _normalized(mentions: tuple[object, ...]) -> list[str]:
    return [mention.normalized for mention in mentions]


@pytest.mark.parametrize(
    ('expected', 'question'),
    [(task_type, question) for _, task_type, question in RELEASED_QUESTIONS],
)
def test_classifies_every_released_question(
    expected: str,
    question: str,
) -> None:
    assert classify_task_type(question) == expected


def test_released_corpus_has_expected_task_distribution() -> None:
    classifications = [
        classify_task_type(question) for _, _, question in RELEASED_QUESTIONS
    ]

    assert classifications.count(NUMERICAL) == 15
    assert classifications.count(OBJECT_REFERENCE) == 30
    assert classifications.count(INSTRUCTION_FOLLOWING) == 30


@pytest.mark.parametrize(
    'question',
    [
        'The lantern between the vase and the stone decoration.',
        'The red pillow closest to the sushi.',
        'The blue chair that is closest to the cup of coffee.',
    ],
)
def test_classifies_object_reference_without_find_prefix(question: str) -> None:
    assert classify_task_type(question) == OBJECT_REFERENCE


@pytest.mark.parametrize('question', ['', '  \t\n  '])
def test_classification_rejects_empty_text(question: str) -> None:
    with pytest.raises(ValueError, match='empty'):
        classify_task_type(question)


def test_classification_rejects_unsupported_statement() -> None:
    with pytest.raises(ValueError, match='supported task type'):
        classify_task_type('Robot language without a task cue.')


@pytest.mark.parametrize(
    ('question', 'expected'),
    [
        ('What number of chairs are near the wall?', NUMERICAL),
        ('Locate the orange chair beside the sink.', OBJECT_REFERENCE),
        ('Identify the plant inside the room.', OBJECT_REFERENCE),
        ('Finish beside the television.', INSTRUCTION_FOLLOWING),
    ],
)
def test_classifies_additional_documented_task_cues(
    question: str,
    expected: str,
) -> None:
    assert classify_task_type(question) == expected


def test_extracts_multiword_objects_structures_and_relations() -> None:
    question = (
        'How many computer monitors are on the table closest to the '
        'map wall decal?'
    )

    extraction = extract_language_features(question)

    assert _normalized(extraction.object_nouns) == ['computer_monitor', 'table']
    assert _normalized(extraction.structural_entities) == ['map_wall_decal']
    assert _normalized(extraction.spatial_relations) == ['on', 'closest_to']


def test_longest_entity_match_prevents_nested_false_entities() -> None:
    extraction = extract_language_features(
        'Find the wall lamp that is between a door frame and a window.'
    )

    assert _normalized(extraction.object_nouns) == ['wall_lamp']
    assert _normalized(extraction.structural_entities) == [
        'door_frame',
        'window',
    ]


@pytest.mark.parametrize(
    ('question', 'expected'),
    [
        ('How many red pillows are on the sofa?', ['red']),
        ('The blue chair that is closest to the cup of coffee.', ['blue']),
        ('How many black pillows are on the sofa?', ['black']),
        ('Find the box closest to the whiteboard.', []),
    ],
)
def test_extracts_colours_without_substring_false_positives(
    question: str,
    expected: list[str],
) -> None:
    assert _normalized(extract_language_features(question).colours) == expected


def test_extracts_explicit_cardinality_and_not_indefinite_articles() -> None:
    extraction = extract_language_features(
        'Take the path between the two columns and stop at a table.'
    )

    assert [item.value for item in extraction.cardinalities] == [2]
    assert [item.source_text.lower() for item in extraction.cardinalities] == [
        'two'
    ]


def test_normalizes_relation_aliases_in_source_order() -> None:
    extraction = extract_language_features(
        'Find the picture under the lamp furthest from the window.'
    )

    assert _normalized(extraction.spatial_relations) == [
        'below',
        'farthest_from',
    ]


def test_extracts_additional_documented_relations() -> None:
    extraction = extract_language_features(
        'Find the chair beside the sink inside the room behind the table.'
    )

    assert _normalized(extraction.spatial_relations) == [
        'beside',
        'inside',
        'behind',
    ]


def test_extracts_ordering_terms_without_overlap_duplicates() -> None:
    extraction = extract_language_features(
        'First go to the chair, and then go to the table, and finally to '
        'the door.'
    )

    assert _normalized(extraction.ordering_terms) == ['first', 'then', 'finally']
    assert [item.source_text.lower() for item in extraction.ordering_terms] == [
        'first',
        'and then',
        'and finally',
    ]


def test_extracts_additional_documented_ordering_terms() -> None:
    extraction = extract_language_features(
        'Before the table, go to the chair; next go to the plant after that.'
    )

    assert _normalized(extraction.ordering_terms) == [
        'before',
        'next',
        'then',
    ]


@pytest.mark.parametrize(
    ('question', 'expected_type'),
    [
        (
            'Go to the cup and avoid the path near the cabinet.',
            'avoid_near',
        ),
        (
            'Go to the chair, avoiding the path between the TV and the table.',
            'avoid_between',
        ),
    ],
)
def test_extracts_complete_avoidance_clause(
    question: str,
    expected_type: str,
) -> None:
    phrase = extract_language_features(question).avoidance_phrases[0]

    assert phrase.constraint_type == expected_type
    assert phrase.source_text.lower().startswith(('avoid ', 'avoiding '))
    assert question[phrase.start:phrase.end] == phrase.source_text


@pytest.mark.parametrize(
    ('question', 'expected_type'),
    [
        ('Do not go near the sofa and finish beside the TV.', 'avoid_near'),
        (
            'Do not pass between the tables and stop near the window.',
            'avoid_between',
        ),
        ('Stay away from the sofa and go to the plant.', 'avoid_near'),
        (
            'Go to the plant without passing between the tables.',
            'avoid_between',
        ),
    ],
)
def test_extracts_additional_documented_avoidance_forms(
    question: str,
    expected_type: str,
) -> None:
    phrase = extract_language_features(question).avoidance_phrases[0]

    assert phrase.constraint_type == expected_type
    assert question[phrase.start:phrase.end] == phrase.source_text


def test_terminal_target_can_follow_mid_route_avoidance() -> None:
    extraction = extract_language_features(
        'First go near the plant, avoid the sofa and stop near the window.'
    )

    assert extraction.terminal_target is not None
    assert extraction.terminal_target.target.normalized == 'window'
    assert extraction.terminal_target.action == 'stop_near'


def test_extracts_finish_beside_terminal_target() -> None:
    extraction = extract_language_features(
        'Head past the tables and finish beside the television.'
    )

    assert extraction.terminal_target is not None
    assert extraction.terminal_target.target.normalized == 'tv'
    assert extraction.terminal_target.action == 'finish_beside'


EXPECTED_TERMINAL_TARGETS = [
    'small_table',
    'tray',
    'painting',
    'table',
    'dining_table',
    'trash_can',
    'potted_plant',
    'potted_plant',
    'chair',
    'picture',
    'lamp',
    'curtain',
    'flower',
    'flower',
    'vase',
    'cabinet',
    'crystal_ball_decoration',
    'soccer_ball',
    'pillow',
    'bowl',
    'table',
    'chair',
    'cup',
    'sphere_decoration',
    'water_cooler',
    'bench',
    'window',
    'door',
    'guitar',
    'window',
]


@pytest.mark.parametrize(
    ('question', 'expected_target'),
    zip(INSTRUCTION_QUESTIONS, EXPECTED_TERMINAL_TARGETS),
)
def test_extracts_every_released_instruction_terminal_target(
    question: str,
    expected_target: str,
) -> None:
    terminal = extract_language_features(question).terminal_target

    assert terminal is not None
    assert terminal.target.normalized == expected_target


@pytest.mark.parametrize(
    'question',
    [
        'How many pillows are on the bed?',
        'Find the flowers near the window.',
    ],
)
def test_non_instruction_has_no_terminal_target(question: str) -> None:
    assert extract_language_features(question).terminal_target is None


@pytest.mark.parametrize(
    ('question', 'expected_objects', 'expected_structures'),
    [
        (
            'Find the speaker on the TV cabinet.',
            ['speaker', 'tv_cabinet'],
            [],
        ),
        (
            'Stop at the trash can closest to the refridgerator.',
            ['trash_can', 'refrigerator'],
            [],
        ),
        (
            'Go near the wardrobe doors to the flowers on the display ledge.',
            ['flower'],
            ['wardrobe_door', 'display_ledge'],
        ),
    ],
)
def test_normalizes_plural_and_released_spelling_variants(
    question: str,
    expected_objects: list[str],
    expected_structures: list[str],
) -> None:
    extraction = extract_language_features(question)

    assert _normalized(extraction.object_nouns) == expected_objects
    assert _normalized(extraction.structural_entities) == expected_structures


@pytest.mark.parametrize(
    'question',
    [question for _, _, question in RELEASED_QUESTIONS],
)
def test_all_released_questions_extract_entities_with_valid_source_spans(
    question: str,
) -> None:
    extraction = extract_language_features(question)
    entities = extraction.object_nouns + extraction.structural_entities

    assert entities
    for mention in entities:
        assert question[mention.start:mention.end] == mention.source_text


def test_extraction_is_deterministic() -> None:
    question = (
        'First, go near the potted plant on the shelf, then take the path '
        'between the two tables, and stop at the bench closest to the map '
        'wall decal.'
    )

    assert extract_language_features(question) == extract_language_features(
        question
    )


@pytest.mark.parametrize('question', ['', '  \t\n  '])
def test_extraction_rejects_empty_text(question: str) -> None:
    with pytest.raises(ValueError, match='empty'):
        extract_language_features(question)
