"""Tests for the frozen cross-subsystem data contracts."""

from dataclasses import fields

import numpy as np
import pytest

from qmapnav.common import EntityReference
from qmapnav.common import EpisodeResult
from qmapnav.common import ObjectInstance
from qmapnav.common import RelationConstraint
from qmapnav.common import ResolvedConstraint
from qmapnav.common import ResolvedTask
from qmapnav.common import RouteConstraint
from qmapnav.common import RouteStep
from qmapnav.common import TaskSpecification


CORE_CONTRACT_FIELDS = (
    (
        TaskSpecification,
        (
            'task_type',
            'entities',
            'relations',
            'ordered_route_steps',
            'forbidden_constraints',
            'terminal_target',
            'parse_confidence',
            'parse_mode',
        ),
    ),
    (
        ObjectInstance,
        (
            'instance_id',
            'class_scores',
            'colour_scores',
            'centroid_xyz',
            'aabb_min_xyz',
            'aabb_max_xyz',
            'obb_dimensions',
            'obb_yaw',
            'orientation_confidence',
            'observation_count',
            'confidence',
        ),
    ),
    (
        ResolvedTask,
        (
            'task_type',
            'selected_target_id',
            'count',
            'ordered_constraints',
            'forbidden_constraints',
            'unresolved_constraints',
            'confidence',
        ),
    ),
    (
        EpisodeResult,
        (
            'success',
            'score_proxy',
            'execution_time',
            'parser_mode',
            'detected_objects',
            'completed_constraints',
            'failed_constraints',
            'failure_category',
        ),
    ),
)


def _make_object_instance(**overrides: object) -> ObjectInstance:
    values = {
        'instance_id': 7,
        'class_scores': {'chair': 0.9},
        'colour_scores': {'blue': 0.7},
        'centroid_xyz': np.array([1.0, 2.0, 0.5]),
        'aabb_min_xyz': np.array([0.5, 1.5, 0.0]),
        'aabb_max_xyz': np.array([1.5, 2.5, 1.0]),
        'obb_dimensions': np.array([1.0, 0.8, 1.0]),
        'obb_yaw': 0.25,
        'orientation_confidence': 0.8,
        'observation_count': 3,
        'confidence': 0.85,
    }
    values.update(overrides)
    return ObjectInstance(**values)


@pytest.mark.parametrize(('contract_type', 'expected_fields'), CORE_CONTRACT_FIELDS)
def test_core_contract_field_order_is_frozen(
    contract_type: type,
    expected_fields: tuple[str, ...],
) -> None:
    assert tuple(field.name for field in fields(contract_type)) == expected_fields


def test_task_specification_supports_chained_relations_and_routes() -> None:
    plant = EntityReference('target', 'potted_plant')
    holder = EntityReference('holder', 'pyramid_candle_holder')
    sofa = EntityReference('sofa', 'sofa')
    tables = EntityReference('tables', 'round_table', cardinality=2)

    task = TaskSpecification(
        task_type='instruction_following',
        entities=[plant, holder, sofa, tables],
        relations=[RelationConstraint('closest', 'target', ['holder'])],
        ordered_route_steps=[
            RouteStep(0, 'go_to', ['target']),
            RouteStep(1, 'pass_between', ['sofa', 'tables']),
        ],
        forbidden_constraints=[
            RouteConstraint('avoid_between', ['sofa', 'tables'])
        ],
        terminal_target=plant,
        parse_confidence=0.9,
        parse_mode='full',
    )

    assert task.terminal_target is plant
    assert task.ordered_route_steps[1].entity_ids == ['sofa', 'tables']


def test_task_specification_rejects_duplicate_entity_ids() -> None:
    entities = [
        EntityReference('chair', 'chair'),
        EntityReference('chair', 'chair', attributes={'colour': 'blue'}),
    ]

    with pytest.raises(ValueError, match='unique'):
        TaskSpecification(
            'object_reference',
            entities,
            [],
            [],
            [],
            entities[0],
            0.5,
            'degraded',
        )


def test_task_specification_rejects_unknown_entity_references() -> None:
    target = EntityReference('target', 'vase')

    with pytest.raises(ValueError, match='missing_anchor'):
        TaskSpecification(
            'object_reference',
            [target],
            [RelationConstraint('near', 'target', ['missing_anchor'])],
            [],
            [],
            target,
            0.8,
            'full',
        )


@pytest.mark.parametrize('cardinality', [0, -1])
def test_entity_reference_requires_positive_explicit_cardinality(
    cardinality: int,
) -> None:
    with pytest.raises(ValueError, match='cardinality'):
        EntityReference('tables', 'table', cardinality=cardinality)


def test_object_instance_copies_and_normalizes_vectors() -> None:
    centroid = np.array([1, 2, 3], dtype=np.int32)
    instance = _make_object_instance(centroid_xyz=centroid)

    centroid[0] = 99

    np.testing.assert_array_equal(instance.centroid_xyz, [1.0, 2.0, 3.0])
    assert instance.centroid_xyz.dtype == np.float64


@pytest.mark.parametrize(
    ('field_name', 'invalid_value', 'message'),
    [
        ('centroid_xyz', np.array([1.0, 2.0]), 'shape'),
        ('obb_dimensions', np.array([1.0, 0.0, 1.0]), 'strictly positive'),
        ('obb_yaw', np.pi + 0.01, 'obb_yaw'),
        ('confidence', 1.01, 'confidence'),
    ],
)
def test_object_instance_rejects_invalid_geometry_and_confidence(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _make_object_instance(**{field_name: invalid_value})


def test_object_instance_rejects_inverted_aabb() -> None:
    with pytest.raises(ValueError, match='aabb_min_xyz'):
        _make_object_instance(
            aabb_min_xyz=np.array([2.0, 1.5, 0.0]),
            aabb_max_xyz=np.array([1.5, 2.5, 1.0]),
        )


def test_resolved_task_distinguishes_zero_count_from_unresolved_count() -> None:
    counted = ResolvedTask('numerical', None, 0, [], [], [], 0.9)
    unresolved = ResolvedTask(
        'numerical',
        None,
        None,
        [],
        [],
        ['target class was not observed'],
        0.2,
    )

    assert counted.count == 0
    assert unresolved.count is None


def test_resolved_constraint_requires_grounded_object_ids() -> None:
    with pytest.raises(ValueError, match='at least one'):
        ResolvedConstraint('near', [], 0.5)


def test_episode_result_copies_constraint_lists() -> None:
    completed = ['go_near:chair_1']
    result = EpisodeResult(
        success=True,
        score_proxy=6.0,
        execution_time=42.5,
        parser_mode='full',
        detected_objects=4,
        completed_constraints=completed,
        failed_constraints=[],
        failure_category=None,
    )

    completed.append('stop_at:table_2')

    assert result.completed_constraints == ['go_near:chair_1']


@pytest.mark.parametrize('parse_mode', ['', 'fallback'])
def test_episode_result_rejects_unknown_parse_mode(parse_mode: str) -> None:
    with pytest.raises(ValueError, match='parser_mode'):
        EpisodeResult(True, 1.0, 2.0, parse_mode, 1, [], [], None)
