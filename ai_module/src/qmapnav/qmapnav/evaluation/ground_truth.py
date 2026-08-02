"""Validated normalized records for released development ground truth."""

from dataclasses import dataclass, fields, is_dataclass
import json
from math import isfinite, pi
from pathlib import Path
from typing import Any


ANSWER_PROVENANCE = frozenset(
    {
        'machine_readable',
        'unavailable',
        'visualization_only',
    }
)
GROUND_TRUTH_TASK_TYPES = frozenset(
    {
        'instruction_following',
        'numerical',
        'object_reference',
    }
)
RELATION_ARITY = {
    'above': 1,
    'below': 1,
    'between': 2,
    'closest_to': 1,
    'farthest_from': 1,
    'hanging_on': 1,
    'inside': 1,
    'near': 1,
    'on': 1,
}


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')


def _require_choice(name: str, value: str, choices: frozenset[str]) -> None:
    if value not in choices:
        expected = ', '.join(sorted(choices))
        raise ValueError(f'{name} must be one of: {expected}')


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    if not isfinite(float(value)):
        raise ValueError(f'{name} must be finite')


def _require_vector3(
    name: str,
    value: tuple[float, float, float],
    *,
    positive: bool = False,
) -> tuple[float, float, float]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f'{name} must contain exactly three values')
    for item in values:
        _require_finite(name, item)
        if positive and float(item) <= 0.0:
            raise ValueError(f'{name} values must be positive')
    return tuple(float(item) for item in values)


@dataclass(frozen=True)
class ColourAttribute:
    """One dominant object colour and its released confidence metadata."""

    label: str
    rgb: tuple[int, int, int]
    proportion: float
    average_lab_distance: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty('colour label', self.label)
        rgb = tuple(self.rgb)
        if len(rgb) != 3:
            raise ValueError('rgb must contain exactly three channels')
        for channel in rgb:
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise ValueError('rgb channels must be integers')
            if not 0 <= channel <= 255:
                raise ValueError('rgb channels must be in [0, 255]')
        object.__setattr__(self, 'rgb', rgb)

        _require_finite('colour proportion', self.proportion)
        if not 0.0 <= self.proportion <= 1.0:
            raise ValueError('colour proportion must be in [0, 1]')
        object.__setattr__(self, 'proportion', float(self.proportion))

        if self.average_lab_distance is not None:
            _require_finite('average_lab_distance', self.average_lab_distance)
            if self.average_lab_distance < 0.0:
                raise ValueError('average_lab_distance must be non-negative')
            object.__setattr__(
                self,
                'average_lab_distance',
                float(self.average_lab_distance),
            )


@dataclass(frozen=True)
class OracleObject:
    """One normalized perfect object annotation in the Unity map frame."""

    object_id: str
    class_name: str
    centre_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    yaw: float
    colours: tuple[ColourAttribute, ...] = ()
    region_id: str | None = None
    raw_class_name: str = ''
    nyu_label: str | None = None
    nyu40_label: str | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty('object_id', self.object_id)
        _require_non_empty('class_name', self.class_name)
        object.__setattr__(
            self,
            'centre_xyz',
            _require_vector3('centre_xyz', self.centre_xyz),
        )
        object.__setattr__(
            self,
            'dimensions_xyz',
            _require_vector3(
                'dimensions_xyz',
                self.dimensions_xyz,
                positive=True,
            ),
        )
        _require_finite('yaw', self.yaw)
        if not -pi <= self.yaw <= pi:
            raise ValueError('yaw must be in [-pi, pi]')
        object.__setattr__(self, 'yaw', float(self.yaw))

        colours = tuple(self.colours)
        if not all(isinstance(item, ColourAttribute) for item in colours):
            raise TypeError('colours must contain only ColourAttribute values')
        labels = [item.label for item in colours]
        if len(labels) != len(set(labels)):
            raise ValueError('dominant colour labels must be unique per object')
        object.__setattr__(self, 'colours', colours)

        if self.region_id is not None:
            _require_non_empty('region_id', self.region_id)
        if self.raw_class_name:
            _require_non_empty('raw_class_name', self.raw_class_name)
        for name, value in (
            ('nyu_label', self.nyu_label),
            ('nyu40_label', self.nyu40_label),
        ):
            if value is not None:
                _require_non_empty(name, value)

        sources = tuple(self.sources)
        for source in sources:
            _require_non_empty('source', source)
        if len(sources) != len(set(sources)):
            raise ValueError('sources must not contain duplicates')
        object.__setattr__(self, 'sources', sources)


@dataclass(frozen=True)
class OracleRegion:
    """One normalized room or region annotation."""

    region_id: str
    label: str
    centre_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    yaw: float

    def __post_init__(self) -> None:
        _require_non_empty('region_id', self.region_id)
        _require_non_empty('region label', self.label)
        object.__setattr__(
            self,
            'centre_xyz',
            _require_vector3('centre_xyz', self.centre_xyz),
        )
        object.__setattr__(
            self,
            'dimensions_xyz',
            _require_vector3(
                'dimensions_xyz',
                self.dimensions_xyz,
                positive=True,
            ),
        )
        _require_finite('yaw', self.yaw)
        if not -pi <= self.yaw <= pi:
            raise ValueError('yaw must be in [-pi, pi]')
        object.__setattr__(self, 'yaw', float(self.yaw))


@dataclass(frozen=True)
class OracleRelation:
    """One normalized binary or ternary ground-truth relation edge."""

    relation: str
    subject_id: str
    object_ids: tuple[str, ...]
    region_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty('relation', self.relation)
        if self.relation not in RELATION_ARITY:
            raise ValueError(f'unsupported normalized relation: {self.relation!r}')
        _require_non_empty('subject_id', self.subject_id)

        object_ids = tuple(self.object_ids)
        expected_arity = RELATION_ARITY[self.relation]
        if len(object_ids) != expected_arity:
            raise ValueError(
                f'{self.relation} requires {expected_arity} object ID value(s)'
            )
        for object_id in object_ids:
            _require_non_empty('relation object_id', object_id)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError('relation object_ids must be distinct')
        object.__setattr__(self, 'object_ids', object_ids)

        if self.region_id is not None:
            _require_non_empty('region_id', self.region_id)


@dataclass(frozen=True)
class OracleTrajectory:
    """An ordered reference trajectory loaded from an ASCII PLY file."""

    points_xyz: tuple[tuple[float, float, float], ...]
    source_path: Path

    def __post_init__(self) -> None:
        points = tuple(
            _require_vector3('trajectory point', point)
            for point in self.points_xyz
        )
        if not points:
            raise ValueError('trajectory must contain at least one point')
        object.__setattr__(self, 'points_xyz', points)
        object.__setattr__(self, 'source_path', Path(self.source_path))


@dataclass(frozen=True)
class QuestionRecord:
    """One released challenge question with stable development-data links."""

    question_id: str
    scene_id: str
    question_text: str
    task_type: str
    question_number: int
    task_index: int
    trajectory_path: Path | None = None
    answer_document_path: Path | None = None
    answer_visualization_index: int | None = None
    expected_count: int | None = None
    expected_object_id: str | None = None
    answer_provenance: str = 'unavailable'

    def __post_init__(self) -> None:
        _require_non_empty('question_id', self.question_id)
        _require_non_empty('scene_id', self.scene_id)
        _require_non_empty('question_text', self.question_text)
        _require_choice('task_type', self.task_type, GROUND_TRUTH_TASK_TYPES)
        for name, value in (
            ('question_number', self.question_number),
            ('task_index', self.task_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        if self.trajectory_path is not None:
            object.__setattr__(self, 'trajectory_path', Path(self.trajectory_path))
            if self.task_type != 'instruction_following':
                raise ValueError('only instruction questions may have trajectories')
        if self.answer_document_path is not None:
            object.__setattr__(
                self,
                'answer_document_path',
                Path(self.answer_document_path),
            )
        if self.answer_visualization_index is not None:
            value = self.answer_visualization_index
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    'answer_visualization_index must be a positive integer'
                )
            if self.answer_document_path is None:
                raise ValueError(
                    'answer visualization index requires an answer document'
                )

        if self.expected_count is not None:
            if self.task_type != 'numerical':
                raise ValueError('expected_count is only valid for numerical tasks')
            if (
                isinstance(self.expected_count, bool)
                or not isinstance(self.expected_count, int)
                or self.expected_count < 0
            ):
                raise ValueError('expected_count must be a non-negative integer')
        if self.expected_object_id is not None:
            if self.task_type != 'object_reference':
                raise ValueError(
                    'expected_object_id is only valid for object-reference tasks'
                )
            _require_non_empty('expected_object_id', self.expected_object_id)

        _require_choice(
            'answer_provenance',
            self.answer_provenance,
            ANSWER_PROVENANCE,
        )
        has_machine_answer = (
            self.expected_count is not None or self.expected_object_id is not None
        )
        if has_machine_answer != (self.answer_provenance == 'machine_readable'):
            raise ValueError(
                'machine-readable provenance must match a structured answer'
            )
        if (
            self.answer_provenance == 'visualization_only'
            and self.answer_document_path is None
        ):
            raise ValueError('visualization-only provenance requires a document')


@dataclass(frozen=True)
class OracleScene:
    """One complete normalized development scene."""

    scene_id: str
    objects: tuple[OracleObject, ...]
    relations: tuple[OracleRelation, ...]
    regions: tuple[OracleRegion, ...]
    questions: tuple[QuestionRecord, ...]
    trajectories: tuple[tuple[str, OracleTrajectory], ...] = ()
    scene_point_cloud_path: Path | None = None
    traversable_area_path: Path | None = None
    source_archive_path: Path | None = None

    def __post_init__(self) -> None:
        _require_non_empty('scene_id', self.scene_id)
        objects = tuple(self.objects)
        relations = tuple(self.relations)
        regions = tuple(self.regions)
        questions = tuple(self.questions)
        trajectories = tuple(self.trajectories)

        if not objects:
            raise ValueError('scene must contain at least one object')
        if not all(isinstance(item, OracleObject) for item in objects):
            raise TypeError('objects must contain only OracleObject values')
        if not all(isinstance(item, OracleRelation) for item in relations):
            raise TypeError('relations must contain only OracleRelation values')
        if not all(isinstance(item, OracleRegion) for item in regions):
            raise TypeError('regions must contain only OracleRegion values')
        if not all(isinstance(item, QuestionRecord) for item in questions):
            raise TypeError('questions must contain only QuestionRecord values')

        object_ids = [item.object_id for item in objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError('object IDs must be unique within a scene')
        region_ids = [item.region_id for item in regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError('region IDs must be unique within a scene')
        question_ids = [item.question_id for item in questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError('question IDs must be unique within a scene')

        known_objects = set(object_ids)
        known_regions = set(region_ids)
        for item in objects:
            if item.region_id is not None and item.region_id not in known_regions:
                raise ValueError(
                    f'object {item.object_id!r} has unknown region {item.region_id!r}'
                )
        for relation in relations:
            if relation.subject_id not in known_objects:
                raise ValueError(
                    f'relation has unknown subject {relation.subject_id!r}'
                )
            for object_id in relation.object_ids:
                if object_id not in known_objects:
                    raise ValueError(
                        f'relation has unknown object {object_id!r}'
                    )
            if (
                relation.region_id is not None
                and relation.region_id not in known_regions
            ):
                raise ValueError(
                    f'relation has unknown region {relation.region_id!r}'
                )
        for question in questions:
            if question.scene_id != self.scene_id:
                raise ValueError('question scene_id does not match OracleScene')

        seen_trajectory_questions: set[str] = set()
        for question_id, trajectory in trajectories:
            _require_non_empty('trajectory question_id', question_id)
            if question_id not in set(question_ids):
                raise ValueError(f'trajectory has unknown question ID {question_id!r}')
            if question_id in seen_trajectory_questions:
                raise ValueError('trajectory question IDs must be unique')
            if not isinstance(trajectory, OracleTrajectory):
                raise TypeError('trajectory values must be OracleTrajectory values')
            seen_trajectory_questions.add(question_id)

        object.__setattr__(self, 'objects', objects)
        object.__setattr__(self, 'relations', relations)
        object.__setattr__(self, 'regions', regions)
        object.__setattr__(self, 'questions', questions)
        object.__setattr__(self, 'trajectories', trajectories)
        for name in (
            'scene_point_cloud_path',
            'traversable_area_path',
            'source_archive_path',
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

    def object_by_id(self, object_id: str) -> OracleObject:
        """Return one object or raise ``KeyError`` for an unknown scene ID."""
        for item in self.objects:
            if item.object_id == object_id:
                return item
        raise KeyError(object_id)

    def trajectory_for(self, question_id: str) -> OracleTrajectory:
        """Return a trajectory linked to a question or raise ``KeyError``."""
        for candidate_id, trajectory in self.trajectories:
            if candidate_id == question_id:
                return trajectory
        raise KeyError(question_id)


def ground_truth_to_data(value: Any) -> Any:
    """Convert normalized records to deterministic JSON-compatible values."""
    if is_dataclass(value):
        return {
            item.name: ground_truth_to_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            key: ground_truth_to_data(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [ground_truth_to_data(item) for item in value]
    return value


def ground_truth_to_json(value: Any, *, indent: int | None = 2) -> str:
    """Serialize normalized records with stable key and collection ordering."""
    return json.dumps(
        ground_truth_to_data(value),
        indent=indent,
        sort_keys=True,
    )
