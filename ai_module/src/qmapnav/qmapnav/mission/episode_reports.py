"""Validated object-reference episode and benchmark records."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

from qmapnav.common import TaskSpecification


OBJECT_REFERENCE_SCHEMA_VERSION = '1.0'
PIPELINE_MODES = frozenset({'oracle_replay', 'perceived', 'synthetic'})
EPISODE_STATUSES = frozenset({
    'completed',
    'completed_with_fallback',
    'launch_failure',
    'protocol_failure',
    'runtime_failure',
    'timeout',
})
PRIMARY_FAILURE_CATEGORIES = frozenset({
    'bad_lifting',
    'bad_relation',
    'duplicate_instance',
    'incorrect_colour',
    'incorrect_obb',
    'missed_anchor',
    'missed_target',
    'parsing',
    'protocol_failure',
})
QUICK_OBJECT_REFERENCE_IDS = (
    'japanese_room_object_reference_01',
    'japanese_room_object_reference_02',
    'livingroom_1_object_reference_02',
    'livingroom_2_object_reference_01',
    'hotel_room_1_object_reference_01',
    'office_1_object_reference_02',
)


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _text(name, value)


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def _finite_non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return float(value)


def _optional_finite(
    name: str,
    value: float | None,
    *,
    non_negative: bool = False,
) -> float | None:
    if value is None:
        return None
    if not isfinite(value) or (non_negative and value < 0.0):
        qualifier = ' finite and non-negative' if non_negative else ' finite'
        raise ValueError(f'{name} must be{qualifier}')
    return float(value)


def _strings(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    copied = tuple(_text(name, item) for item in values)
    if len(copied) != len(set(copied)):
        raise ValueError(f'{name} must not contain duplicates')
    return copied


def _json_safe(value: Any) -> Any:
    """Return a recursively stable JSON-compatible value."""
    if hasattr(value, 'tolist'):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'value is not JSON compatible: {type(value).__name__}')


def task_specification_data(task: TaskSpecification) -> dict[str, Any]:
    """Serialize the frozen task contract without changing it."""
    if not isinstance(task, TaskSpecification):
        raise TypeError('task must be TaskSpecification')
    return _json_safe(asdict(task))


def _task_tags(task: TaskSpecification) -> tuple[str, ...]:
    tags = set()
    target = task.entities[0]
    if 'colour' in target.attributes:
        tags.add('colour')
    relation_names = {item.relation for item in task.relations}
    tags.update(relation_names & {
        'above', 'below', 'between', 'closest_to', 'farthest_from',
        'near', 'on',
    })
    if {'window', 'door', 'door_frame', 'wall'} & {
        item.class_name for item in task.entities[1:]
    }:
        tags.add('structural_anchor')
    if len(task.entities) > 2:
        tags.add('nested_anchor_relation')
    if target.class_name in {
        'beer_bottle', 'book', 'bowl', 'clock', 'flowers', 'paper_cup',
        'pillow', 'speaker', 'vase',
    }:
        tags.add('small_object')
    return tuple(sorted(tags))


@dataclass(frozen=True)
class ObjectReferenceCase:
    """One released object-reference benchmark case."""

    case_id: str
    scene_id: str
    question_id: str
    question: str
    expected_target_class: str
    expected_anchor_classes: tuple[str, ...]
    expected_target_instance_id: str | None = None
    ground_truth_box: dict[str, Any] | None = None
    ground_truth_centre_xyz: tuple[float, float, float] | None = None
    answer_provenance: str | None = None
    initial_robot_pose: tuple[float, float, float] | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            'case_id', 'scene_id', 'question_id', 'question',
            'expected_target_class',
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(
            self,
            'expected_anchor_classes',
            _strings('expected_anchor_classes', self.expected_anchor_classes),
        )
        object.__setattr__(
            self,
            'expected_target_instance_id',
            _optional_text(
                'expected_target_instance_id',
                self.expected_target_instance_id,
            ),
        )
        object.__setattr__(
            self,
            'answer_provenance',
            _optional_text('answer_provenance', self.answer_provenance),
        )
        if self.ground_truth_centre_xyz is not None:
            centre = tuple(float(item) for item in self.ground_truth_centre_xyz)
            if len(centre) != 3 or not all(isfinite(item) for item in centre):
                raise ValueError('ground_truth_centre_xyz must be finite XYZ')
            object.__setattr__(self, 'ground_truth_centre_xyz', centre)
        if self.initial_robot_pose is not None:
            pose = tuple(float(item) for item in self.initial_robot_pose)
            if len(pose) != 3 or not all(isfinite(item) for item in pose):
                raise ValueError('initial_robot_pose must be finite XY-yaw')
            object.__setattr__(self, 'initial_robot_pose', pose)
        if self.ground_truth_box is not None:
            object.__setattr__(
                self, 'ground_truth_box', _json_safe(self.ground_truth_box)
            )
        object.__setattr__(self, 'tags', _strings('tags', self.tags))

    @classmethod
    def from_question(
        cls,
        question: Any,
        task: TaskSpecification,
    ) -> 'ObjectReferenceCase':
        """Build manifest metadata from released text and its parse."""
        if question.task_type != 'object_reference':
            raise ValueError('question must be object_reference')
        if task.task_type != 'object_reference' or not task.entities:
            raise ValueError('task must contain an object-reference target')
        target = task.entities[0]
        anchors = tuple(dict.fromkeys(
            item.class_name for item in task.entities[1:]
        ))
        return cls(
            case_id=question.question_id,
            scene_id=question.scene_id,
            question_id=question.question_id,
            question=question.question_text,
            expected_target_class=target.class_name,
            expected_anchor_classes=anchors,
            expected_target_instance_id=question.expected_object_id,
            answer_provenance=question.answer_provenance,
            tags=_task_tags(task),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable manifest JSON."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class StageEvidence:
    """Earliest-failure evidence used by the deterministic classifier."""

    parser_correct: bool | None = None
    target_observed: bool | None = None
    target_detected: bool | None = None
    anchors_available: bool | None = None
    target_lifted: bool | None = None
    identity_correct: bool | None = None
    colour_correct: bool | None = None
    relation_correct: bool | None = None
    target_selected_correctly: bool | None = None
    obb_acceptable: bool | None = None
    protocol_valid: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            'parser_correct', 'target_observed', 'target_detected',
            'anchors_available', 'target_lifted', 'identity_correct',
            'colour_correct', 'relation_correct',
            'target_selected_correctly', 'obb_acceptable', 'protocol_valid',
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f'{name} must be bool or None')
        object.__setattr__(self, 'detail', _json_safe(self.detail))


@dataclass(frozen=True)
class ObjectReferenceEpisodeResult:
    """One terminal Day 10 response with full pipeline diagnostics."""

    run_id: str
    case_id: str
    scene_id: str
    question: str
    pipeline_mode: str
    episode_status: str
    parser_mode: str | None
    task_specification: dict[str, Any]
    requested_classes: tuple[str, ...]
    stage_evidence: StageEvidence
    target_detections: int = 0
    anchor_detections: dict[str, int] = field(default_factory=dict)
    lifting_results: tuple[dict[str, Any], ...] = ()
    object_candidates_3d: int = 0
    persistent_instances: int = 0
    fusion_events: tuple[dict[str, Any], ...] = ()
    ranked_target_ids: tuple[str, ...] = ()
    ranked_target_scores: tuple[float, ...] = ()
    ranked_score_components: tuple[dict[str, Any], ...] = ()
    confidence_margin: float | None = None
    unresolved_constraints: tuple[str, ...] = ()
    selected_target_id: str | None = None
    predicted_box: dict[str, Any] | None = None
    marker_validation_errors: tuple[str, ...] = ()
    marker_published: bool = False
    marker_publish_count: int = 0
    marker_publish_time_sec: float | None = None
    matching_waypoint_published: bool = False
    targeted_viewpoint_used: bool = False
    targeted_viewpoint_reason: str | None = None
    targeted_viewpoint_pose: tuple[float, float, float] | None = None
    target_selection_correct: bool | None = None
    centre_error_m: float | None = None
    aabb_iou: float | None = None
    obb_iou: float | None = None
    yaw_error_rad: float | None = None
    marker_success: bool | None = None
    success: bool | None = None
    primary_failure_category: str | None = None
    failure_subtype: str | None = None
    failure_detail: str | None = None
    manual_review: str | None = None
    episode_duration_sec: float = 0.0
    trace_path: str = ''
    evidence_directory: str = ''
    final_response_logged: bool = True
    proxy_score: float = 0.0
    schema_version: str = OBJECT_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ('run_id', 'case_id', 'scene_id', 'question'):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        if self.pipeline_mode not in PIPELINE_MODES:
            raise ValueError('unsupported pipeline_mode')
        if self.episode_status not in EPISODE_STATUSES:
            raise ValueError('unsupported episode_status')
        if self.parser_mode not in {None, 'degraded', 'full'}:
            raise ValueError('unsupported parser_mode')
        if not isinstance(self.stage_evidence, StageEvidence):
            raise TypeError('stage_evidence must be StageEvidence')
        for name in (
            'target_detections', 'object_candidates_3d',
            'persistent_instances', 'marker_publish_count',
        ):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        for name, value in self.anchor_detections.items():
            _text('anchor class', name)
            _count(f'anchor_detections[{name!r}]', value)
        object.__setattr__(
            self, 'anchor_detections', dict(sorted(self.anchor_detections.items()))
        )
        object.__setattr__(
            self, 'requested_classes', _strings(
                'requested_classes', self.requested_classes
            )
        )
        object.__setattr__(
            self, 'ranked_target_ids', _strings(
                'ranked_target_ids', self.ranked_target_ids
            )
        )
        if len(self.ranked_target_ids) != len(self.ranked_target_scores):
            raise ValueError('ranked IDs and scores must have equal length')
        if not all(isfinite(item) for item in self.ranked_target_scores):
            raise ValueError('ranked_target_scores must be finite')
        for name in (
            'confidence_margin', 'centre_error_m', 'aabb_iou', 'obb_iou',
            'yaw_error_rad', 'marker_publish_time_sec',
        ):
            object.__setattr__(
                self,
                name,
                _optional_finite(
                    name,
                    getattr(self, name),
                    non_negative=name != 'yaw_error_rad',
                ),
            )
        object.__setattr__(
            self,
            'episode_duration_sec',
            _finite_non_negative(
                'episode_duration_sec', self.episode_duration_sec
            ),
        )
        object.__setattr__(
            self, 'proxy_score', _finite_non_negative(
                'proxy_score', self.proxy_score
            )
        )
        if self.primary_failure_category is not None and (
            self.primary_failure_category not in PRIMARY_FAILURE_CATEGORIES
        ):
            raise ValueError('unsupported primary_failure_category')
        if self.marker_published != (self.marker_publish_count > 0):
            raise ValueError('marker_published must match publish count')
        if self.marker_publish_count > 1:
            raise ValueError('official marker may be published at most once')
        if self.targeted_viewpoint_used and self.targeted_viewpoint_pose is None:
            raise ValueError('used viewpoint requires a pose')
        if not isinstance(self.final_response_logged, bool):
            raise TypeError('final_response_logged must be bool')
        object.__setattr__(
            self, 'task_specification', _json_safe(self.task_specification)
        )
        for name in (
            'lifting_results', 'fusion_events', 'ranked_score_components',
        ):
            object.__setattr__(
                self, name, tuple(_json_safe(item) for item in getattr(self, name))
            )
        if self.predicted_box is not None:
            object.__setattr__(
                self, 'predicted_box', _json_safe(self.predicted_box)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-safe episode evidence."""
        return _json_safe(asdict(self))


def build_object_reference_manifest(
    questions: tuple[Any, ...],
    parser,
) -> tuple[ObjectReferenceCase, ...]:
    """Build the exact released Day 10 case list in source order."""
    selected = tuple(
        item for item in questions if item.task_type == 'object_reference'
    )
    if len(selected) != 30:
        raise ValueError(
            f'expected 30 released object-reference questions, got {len(selected)}'
        )
    cases = tuple(
        ObjectReferenceCase.from_question(item, parser(item.question_text))
        for item in selected
    )
    scene_counts: dict[str, int] = {}
    for case in cases:
        scene_counts[case.scene_id] = scene_counts.get(case.scene_id, 0) + 1
    if len(scene_counts) != 15 or set(scene_counts.values()) != {2}:
        raise ValueError('expected 15 scenes with two object-reference cases each')
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError('manifest case IDs must be unique')
    return cases


def manifest_digest(cases: tuple[ObjectReferenceCase, ...]) -> str:
    """Hash stable case metadata for run provenance."""
    payload = json.dumps(
        [item.to_dict() for item in cases],
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class FailureClassification:
    """One primary category with a more specific diagnostic subtype."""

    category: str | None
    subtype: str | None
    detail: str | None


@dataclass(frozen=True)
class FixCandidate:
    """Evidence-supported proposed correction ranked by expected score."""

    failure_source: str
    affected_tasks: int
    proposed_fix: str
    fix_confidence: float
    effort: float
    integration_risk: float
    points_per_task: float = 2.0

    def __post_init__(self) -> None:
        if not self.failure_source.strip() or not self.proposed_fix.strip():
            raise ValueError('failure source and proposed fix must be non-empty')
        if (
            isinstance(self.affected_tasks, bool)
            or not isinstance(self.affected_tasks, int)
            or self.affected_tasks < 0
        ):
            raise ValueError('affected_tasks must be a non-negative integer')
        if not isfinite(self.fix_confidence) or not (
            0.0 <= self.fix_confidence <= 1.0
        ):
            raise ValueError('fix_confidence must lie in [0, 1]')
        for name in ('effort', 'integration_risk', 'points_per_task'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')

    @property
    def expected_recovered_score(self) -> float:
        """Return expected recovered challenge points."""
        return (
            self.affected_tasks
            * self.points_per_task
            * self.fix_confidence
        )

    @property
    def priority(self) -> float:
        """Return score gain normalized by effort and integration risk."""
        return self.expected_recovered_score / (
            self.effort * self.integration_risk
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable report row."""
        return {
            'failure_source': self.failure_source,
            'affected_tasks': self.affected_tasks,
            'proposed_fix': self.proposed_fix,
            'fix_confidence': self.fix_confidence,
            'effort': self.effort,
            'integration_risk': self.integration_risk,
            'points_per_task': self.points_per_task,
            'expected_recovered_score': self.expected_recovered_score,
            'priority': self.priority,
        }


def classify_primary_failure(
    evidence: StageEvidence,
) -> FailureClassification:
    """Apply the frozen earliest-meaningful-failure decision tree."""
    if not isinstance(evidence, StageEvidence):
        raise TypeError('evidence must be StageEvidence')
    detail = dict(evidence.detail)
    if evidence.parser_correct is False:
        return FailureClassification(
            'parsing', _detail(detail, 'parser_subtype'),
            _detail(detail, 'parser_detail'),
        )
    if (
        evidence.target_observed is False
        or evidence.target_detected is False
    ):
        subtype = _detail(detail, 'target_subtype')
        if subtype is None:
            subtype = (
                'not_observed'
                if evidence.target_observed is False else 'detector_miss'
            )
        return FailureClassification(
            'missed_target', subtype, _detail(detail, 'target_detail')
        )
    if evidence.anchors_available is False:
        return FailureClassification(
            'missed_anchor',
            _detail(detail, 'anchor_subtype') or 'anchor_unavailable',
            _detail(detail, 'anchor_detail'),
        )
    if evidence.target_lifted is False:
        return FailureClassification(
            'bad_lifting',
            _detail(detail, 'lifting_subtype') or 'unusable_3d_cluster',
            _detail(detail, 'lifting_detail'),
        )
    if evidence.identity_correct is False:
        return FailureClassification(
            'duplicate_instance',
            _detail(detail, 'identity_subtype') or 'duplicate_split',
            _detail(detail, 'identity_detail'),
        )
    if evidence.colour_correct is False:
        return FailureClassification(
            'incorrect_colour',
            _detail(detail, 'colour_subtype') or 'wrong_colour_ranking',
            _detail(detail, 'colour_detail'),
        )
    if (
        evidence.relation_correct is False
        or evidence.target_selected_correctly is False
    ):
        return FailureClassification(
            'bad_relation',
            _detail(detail, 'relation_subtype') or 'wrong_candidate_rank',
            _detail(detail, 'relation_detail'),
        )
    if evidence.obb_acceptable is False:
        return FailureClassification(
            'incorrect_obb',
            _detail(detail, 'obb_subtype') or 'unacceptable_geometry',
            _detail(detail, 'obb_detail'),
        )
    if evidence.protocol_valid is False:
        return FailureClassification(
            'protocol_failure',
            _detail(detail, 'protocol_subtype') or 'invalid_delivery',
            _detail(detail, 'protocol_detail'),
        )
    return FailureClassification(None, None, None)


def rank_fix_candidates(
    candidates: Any,
) -> tuple[FixCandidate, ...]:
    """Rank proposed corrections deterministically by expected gain."""
    values = tuple(candidates)
    return tuple(sorted(
        values,
        key=lambda item: (
            -item.priority,
            -item.expected_recovered_score,
            item.effort,
            item.failure_source,
        ),
    ))


def _detail(values: dict[str, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    'EPISODE_STATUSES',
    'FailureClassification',
    'FixCandidate',
    'OBJECT_REFERENCE_SCHEMA_VERSION',
    'ObjectReferenceCase',
    'ObjectReferenceEpisodeResult',
    'PIPELINE_MODES',
    'PRIMARY_FAILURE_CATEGORIES',
    'QUICK_OBJECT_REFERENCE_IDS',
    'StageEvidence',
    'build_object_reference_manifest',
    'classify_primary_failure',
    'manifest_digest',
    'rank_fix_candidates',
    'task_specification_data',
]
