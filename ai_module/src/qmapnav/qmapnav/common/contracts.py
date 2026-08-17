"""Stable data contracts shared by Q-MapNav subsystems."""

from dataclasses import dataclass, field
from math import isfinite, pi

import numpy as np


TASK_TYPES = frozenset(
    {
        'instruction_following',
        'numerical',
        'object_reference',
    }
)
PARSE_MODES = frozenset({'degraded', 'full'})


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')


def _require_choice(name: str, value: str, choices: frozenset[str]) -> None:
    if value not in choices:
        expected = ', '.join(sorted(choices))
        raise ValueError(f'{name} must be one of: {expected}')


def _require_confidence(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be finite and in [0, 1]')


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _copy_string_list(name: str, values: list[str], *, allow_empty: bool) -> list[str]:
    copied = list(values)
    if not allow_empty and not copied:
        raise ValueError(f'{name} must contain at least one value')
    for value in copied:
        _require_non_empty(name, value)
    return copied


def _copy_vector3(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f'{name} must have shape (3,)')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    return array.copy()


def _copy_scores(name: str, scores: dict[str, float], *, allow_empty: bool) -> dict[str, float]:
    copied = dict(scores)
    if not allow_empty and not copied:
        raise ValueError(f'{name} must contain at least one score')
    for label, score in copied.items():
        _require_non_empty(f'{name} label', label)
        _require_confidence(f'{name}[{label!r}]', score)
    return copied


@dataclass
class EntityReference:
    """
    A normalized entity mention produced by the language subsystem.

    ``entity_id`` is unique within one task specification. ``class_name`` and
    attribute keys and values use normalized snake-case vocabulary. Attributes
    hold constraints such as ``colour=red`` or ``size=small``. ``cardinality``
    is positive when explicitly stated and ``None`` when unspecified.
    """

    entity_id: str
    class_name: str
    attributes: dict[str, str] = field(default_factory=dict)
    cardinality: int | None = None
    source_text: str = ''

    def __post_init__(self) -> None:
        _require_non_empty('entity_id', self.entity_id)
        _require_non_empty('class_name', self.class_name)

        self.attributes = dict(self.attributes)
        for name, value in self.attributes.items():
            _require_non_empty('attribute name', name)
            _require_non_empty(f'attribute {name!r}', value)

        if self.cardinality is not None:
            _require_positive_int('cardinality', self.cardinality)
        if not isinstance(self.source_text, str):
            raise ValueError('source_text must be a string')


@dataclass
class RelationConstraint:
    """A spatial relation between a subject entity and one or more anchors."""

    relation: str
    subject_entity_id: str
    anchor_entity_ids: list[str]

    def __post_init__(self) -> None:
        _require_non_empty('relation', self.relation)
        _require_non_empty('subject_entity_id', self.subject_entity_id)
        self.anchor_entity_ids = _copy_string_list(
            'anchor_entity_ids',
            self.anchor_entity_ids,
            allow_empty=False,
        )


@dataclass
class RouteStep:
    """One ordered semantic action in an instruction-following route."""

    step_index: int
    action: str
    entity_ids: list[str]

    def __post_init__(self) -> None:
        _require_non_negative_int('step_index', self.step_index)
        _require_non_empty('action', self.action)
        self.entity_ids = _copy_string_list(
            'entity_ids',
            self.entity_ids,
            allow_empty=False,
        )


@dataclass
class RouteConstraint:
    """A semantic region the executed route must avoid."""

    constraint_type: str
    entity_ids: list[str]

    def __post_init__(self) -> None:
        _require_non_empty('constraint_type', self.constraint_type)
        self.entity_ids = _copy_string_list(
            'entity_ids',
            self.entity_ids,
            allow_empty=False,
        )


@dataclass
class TaskSpecification:
    """Structured output shared by language and downstream task reasoning."""

    task_type: str
    entities: list[EntityReference]
    relations: list[RelationConstraint]
    ordered_route_steps: list[RouteStep]
    forbidden_constraints: list[RouteConstraint]
    terminal_target: EntityReference | None
    parse_confidence: float
    parse_mode: str

    def __post_init__(self) -> None:
        _require_choice('task_type', self.task_type, TASK_TYPES)
        _require_confidence('parse_confidence', self.parse_confidence)
        _require_choice('parse_mode', self.parse_mode, PARSE_MODES)

        self.entities = list(self.entities)
        self.relations = list(self.relations)
        self.ordered_route_steps = list(self.ordered_route_steps)
        self.forbidden_constraints = list(self.forbidden_constraints)

        if not all(isinstance(entity, EntityReference) for entity in self.entities):
            raise TypeError('entities must contain only EntityReference values')
        if not all(
            isinstance(relation, RelationConstraint) for relation in self.relations
        ):
            raise TypeError('relations must contain only RelationConstraint values')
        if not all(isinstance(step, RouteStep) for step in self.ordered_route_steps):
            raise TypeError('ordered_route_steps must contain only RouteStep values')
        if not all(
            isinstance(constraint, RouteConstraint)
            for constraint in self.forbidden_constraints
        ):
            raise TypeError(
                'forbidden_constraints must contain only RouteConstraint values'
            )

        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError('entity_id values must be unique within a task')

        known_ids = set(entity_ids)
        for relation in self.relations:
            self._require_known_entity(relation.subject_entity_id, known_ids)
            for entity_id in relation.anchor_entity_ids:
                self._require_known_entity(entity_id, known_ids)
        for step in self.ordered_route_steps:
            for entity_id in step.entity_ids:
                self._require_known_entity(entity_id, known_ids)
        for constraint in self.forbidden_constraints:
            for entity_id in constraint.entity_ids:
                self._require_known_entity(entity_id, known_ids)

        if self.terminal_target is not None:
            if not isinstance(self.terminal_target, EntityReference):
                raise TypeError('terminal_target must be an EntityReference or None')
            self._require_known_entity(self.terminal_target.entity_id, known_ids)

    @staticmethod
    def _require_known_entity(entity_id: str, known_ids: set[str]) -> None:
        if entity_id not in known_ids:
            raise ValueError(f'unknown entity_id referenced by task: {entity_id!r}')


@dataclass
class ObjectInstance:
    """
    A fused object hypothesis expressed in the ROS ``map`` frame.

    All XYZ values and box dimensions are metres. OBB dimensions are ordered as
    length, width, and height. ``obb_yaw`` is a rotation about positive Z in
    radians, normalized to ``[-pi, pi]``. Scores and confidence values are in
    ``[0, 1]`` but do not need to sum to one.
    """

    instance_id: int
    class_scores: dict[str, float]
    colour_scores: dict[str, float]
    centroid_xyz: np.ndarray
    aabb_min_xyz: np.ndarray
    aabb_max_xyz: np.ndarray
    obb_dimensions: np.ndarray
    obb_yaw: float
    orientation_confidence: float
    observation_count: int
    confidence: float

    def __post_init__(self) -> None:
        _require_non_negative_int('instance_id', self.instance_id)
        _require_positive_int('observation_count', self.observation_count)
        _require_confidence('orientation_confidence', self.orientation_confidence)
        _require_confidence('confidence', self.confidence)

        self.class_scores = _copy_scores(
            'class_scores',
            self.class_scores,
            allow_empty=False,
        )
        self.colour_scores = _copy_scores(
            'colour_scores',
            self.colour_scores,
            allow_empty=True,
        )
        self.centroid_xyz = _copy_vector3('centroid_xyz', self.centroid_xyz)
        self.aabb_min_xyz = _copy_vector3('aabb_min_xyz', self.aabb_min_xyz)
        self.aabb_max_xyz = _copy_vector3('aabb_max_xyz', self.aabb_max_xyz)
        self.obb_dimensions = _copy_vector3('obb_dimensions', self.obb_dimensions)

        if np.any(self.aabb_min_xyz > self.aabb_max_xyz):
            raise ValueError('aabb_min_xyz must not exceed aabb_max_xyz')
        if np.any(self.obb_dimensions <= 0.0):
            raise ValueError('obb_dimensions must be strictly positive')
        if not isfinite(self.obb_yaw) or not -pi <= self.obb_yaw <= pi:
            raise ValueError('obb_yaw must be finite and in [-pi, pi]')


@dataclass
class ResolvedConstraint:
    """A semantic constraint grounded to episode-local object IDs."""

    constraint_type: str
    object_ids: list[int]
    confidence: float

    def __post_init__(self) -> None:
        _require_non_empty('constraint_type', self.constraint_type)
        self.object_ids = list(self.object_ids)
        if not self.object_ids:
            raise ValueError('object_ids must contain at least one value')
        for object_id in self.object_ids:
            _require_non_negative_int('object_id', object_id)
        _require_confidence('confidence', self.confidence)


@dataclass
class ResolvedTask:
    """Semantic-resolution result consumed by counting or route planning."""

    task_type: str
    selected_target_id: int | None
    count: int | None
    ordered_constraints: list[ResolvedConstraint]
    forbidden_constraints: list[ResolvedConstraint]
    unresolved_constraints: list[str]
    confidence: float

    def __post_init__(self) -> None:
        _require_choice('task_type', self.task_type, TASK_TYPES)
        _require_confidence('confidence', self.confidence)

        if self.selected_target_id is not None:
            _require_non_negative_int('selected_target_id', self.selected_target_id)
        if self.count is not None:
            _require_non_negative_int('count', self.count)

        self.ordered_constraints = list(self.ordered_constraints)
        self.forbidden_constraints = list(self.forbidden_constraints)
        if not all(
            isinstance(constraint, ResolvedConstraint)
            for constraint in self.ordered_constraints
        ):
            raise TypeError(
                'ordered_constraints must contain only ResolvedConstraint values'
            )
        if not all(
            isinstance(constraint, ResolvedConstraint)
            for constraint in self.forbidden_constraints
        ):
            raise TypeError(
                'forbidden_constraints must contain only ResolvedConstraint values'
            )
        self.unresolved_constraints = _copy_string_list(
            'unresolved_constraints',
            self.unresolved_constraints,
            allow_empty=True,
        )


@dataclass
class EpisodeResult:
    """Stable evaluation summary produced when an episode terminates."""

    success: bool
    score_proxy: float
    execution_time: float
    parser_mode: str
    detected_objects: int
    completed_constraints: list[str]
    failed_constraints: list[str]
    failure_category: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError('success must be a boolean')
        if not isfinite(self.score_proxy) or self.score_proxy < 0.0:
            raise ValueError('score_proxy must be finite and non-negative')
        if not isfinite(self.execution_time) or self.execution_time < 0.0:
            raise ValueError('execution_time must be finite and non-negative')
        _require_choice('parser_mode', self.parser_mode, PARSE_MODES)
        _require_non_negative_int('detected_objects', self.detected_objects)

        self.completed_constraints = _copy_string_list(
            'completed_constraints',
            self.completed_constraints,
            allow_empty=True,
        )
        self.failed_constraints = _copy_string_list(
            'failed_constraints',
            self.failed_constraints,
            allow_empty=True,
        )
        if self.failure_category is not None:
            _require_non_empty('failure_category', self.failure_category)
