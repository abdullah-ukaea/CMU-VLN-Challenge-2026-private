"""Deterministic task resolution against perfect development-scene objects."""

from dataclasses import dataclass
from itertools import combinations, product
from math import cos, hypot, sin

from qmapnav.common import EntityReference
from qmapnav.common import RelationConstraint
from qmapnav.common import TaskSpecification
from qmapnav.evaluation.dataset_loader import normalize_class_name
from qmapnav.evaluation.ground_truth import OracleObject
from qmapnav.evaluation.ground_truth import OracleRelation
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.reasoning.semantic_geometry import object_footprint


_HARD_RELATION_ALIASES = {
    'above': 'above',
    'below': 'below',
    'beside': 'near',
    'between': 'between',
    'inside': 'inside',
    'near': 'near',
    'on': 'on',
}
_RANKING_RELATIONS = frozenset(
    {'closest_to', 'farthest_from', 'far_from'}
)
_QUERY_CLASS_EXPANSIONS = {
    'flower': frozenset({'flower', 'flowers'}),
    'lamp': frozenset({'desk_light', 'lamp'}),
    'small_table': frozenset({'small_table', 'table'}),
    'stone_decoration': frozenset(
        {'stone_decoration', 'zen_stone_decoration'}
    ),
    'tea_table': frozenset({'table', 'tea_table'}),
}


class OracleReasoningError(ValueError):
    """Indicate that a task cannot be resolved against oracle scene data."""


@dataclass(frozen=True)
class CandidateDecision:
    """Explain why one class-matching target candidate was kept or rejected."""

    object_id: str
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError('object_id must be non-empty')
        reasons = tuple(self.reasons)
        if not reasons or any(not reason.strip() for reason in reasons):
            raise ValueError('candidate decisions require non-empty reasons')
        object.__setattr__(self, 'reasons', reasons)


@dataclass(frozen=True)
class EntityResolution:
    """Candidate sets and rejection evidence for every symbolic entity."""

    candidates: tuple[tuple[str, tuple[str, ...]], ...]
    rejection_reasons: tuple[tuple[str, str, tuple[str, ...]], ...]
    warnings: tuple[str, ...] = ()

    def candidates_for(self, entity_id: str) -> tuple[str, ...]:
        """Return remaining object IDs for one parser entity."""
        for candidate_entity_id, object_ids in self.candidates:
            if candidate_entity_id == entity_id:
                return object_ids
        raise KeyError(entity_id)

    def reasons_for(self, entity_id: str, object_id: str) -> tuple[str, ...]:
        """Return accumulated rejection reasons for one entity/object pair."""
        for candidate_entity_id, candidate_id, reasons in self.rejection_reasons:
            if candidate_entity_id == entity_id and candidate_id == object_id:
                return reasons
        return ()


@dataclass(frozen=True)
class NumericalResult:
    """Oracle answer and complete target-candidate trace for a count task."""

    count: int
    matching_object_ids: tuple[str, ...]
    candidate_decisions: tuple[CandidateDecision, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectReferenceResult:
    """Oracle selection result with deterministic ambiguity diagnostics."""

    selected_object_id: str | None
    selected_object: OracleObject | None
    candidate_scores: tuple[tuple[str, float], ...]
    candidate_decisions: tuple[CandidateDecision, ...]
    confidence_margin: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.selected_object_id is None) != (self.selected_object is None):
            raise ValueError('selected object ID and object must both be present')
        if self.selected_object is not None:
            if self.selected_object.object_id != self.selected_object_id:
                raise ValueError('selected object ID does not match selected object')
        if not 0.0 <= self.confidence_margin <= 1.0:
            raise ValueError('confidence_margin must be in [0, 1]')


def _id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _target_entity(task: TaskSpecification) -> EntityReference:
    if not task.entities:
        raise OracleReasoningError('task has no target entity')
    return task.entities[0]


def _object_by_id(scene: OracleScene) -> dict[str, OracleObject]:
    return {obj.object_id: obj for obj in scene.objects}


def _class_candidates(
    entity: EntityReference,
    scene: OracleScene,
) -> tuple[OracleObject, ...]:
    query_class = normalize_class_name(entity.class_name)
    accepted_classes = _QUERY_CLASS_EXPANSIONS.get(
        query_class,
        frozenset({query_class}),
    )
    matches = tuple(
        obj
        for obj in scene.objects
        if (
            normalize_class_name(obj.class_name) in accepted_classes
            or (
                query_class == 'cabinet'
                and normalize_class_name(obj.class_name).endswith('_cabinet')
            )
        )
    )
    if not matches and query_class == 'window':
        return tuple(obj for obj in scene.objects if obj.class_name == 'wall')
    return matches


def _colour_matches(requested_colour: str, obj: OracleObject) -> bool:
    labels = {colour.label for colour in obj.colours}
    if requested_colour in labels:
        return True
    if requested_colour == 'red' and 'maroon' in labels:
        return True
    for colour in obj.colours:
        red, green, blue = colour.rgb
        if requested_colour == 'black' and max(red, green, blue) <= 85:
            return True
        if (
            requested_colour == 'blue'
            and blue >= red + 20
            and blue >= green
        ):
            return True
    return False


def _base_candidates(
    task: TaskSpecification,
    scene: OracleScene,
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], list[str]],
    list[str],
]:
    candidates: dict[str, set[str]] = {}
    reasons: dict[tuple[str, str], list[str]] = {}
    warnings: list[str] = []
    for entity in task.entities:
        class_matches = _class_candidates(entity, scene)
        candidate_ids = {obj.object_id for obj in class_matches}
        requested_colour = entity.attributes.get('colour')
        if requested_colour is not None:
            for obj in class_matches:
                if not _colour_matches(requested_colour, obj):
                    candidate_ids.discard(obj.object_id)
                    reasons.setdefault(
                        (entity.entity_id, obj.object_id), []
                    ).append(f'colour is not {requested_colour}')

        unsupported = sorted(set(entity.attributes) - {'colour'})
        if unsupported:
            warnings.append(
                f'{entity.entity_id}: oracle metadata cannot enforce attributes '
                f'{unsupported}'
            )
        if not class_matches:
            warnings.append(
                f'{entity.entity_id}: no scene object has class '
                f'{normalize_class_name(entity.class_name)!r}'
            )
        candidates[entity.entity_id] = candidate_ids
    return candidates, reasons, warnings


def _support_radius(obj: OracleObject, direction: tuple[float, float]) -> float:
    axis_x = (cos(obj.yaw), sin(obj.yaw))
    axis_y = (-sin(obj.yaw), cos(obj.yaw))
    length, width, _ = obj.dimensions_xyz
    return (
        abs(direction[0] * axis_x[0] + direction[1] * axis_x[1])
        * length
        / 2.0
        + abs(direction[0] * axis_y[0] + direction[1] * axis_y[1])
        * width
        / 2.0
    )


def _horizontal_gap(object_a: OracleObject, object_b: OracleObject) -> float:
    delta_x = object_b.centre_xyz[0] - object_a.centre_xyz[0]
    delta_y = object_b.centre_xyz[1] - object_a.centre_xyz[1]
    centre_distance = hypot(delta_x, delta_y)
    if centre_distance <= 1e-9:
        return 0.0
    direction = (delta_x / centre_distance, delta_y / centre_distance)
    return max(
        0.0,
        centre_distance
        - _support_radius(object_a, direction)
        - _support_radius(object_b, direction),
    )


def _direct_relation_exists(
    relation: str,
    subject_id: str,
    anchor_ids: tuple[str, ...],
    scene_relations: tuple[OracleRelation, ...],
) -> bool:
    for item in scene_relations:
        if item.relation != relation or item.subject_id != subject_id:
            continue
        if item.object_ids == anchor_ids:
            return True
        if relation == 'between' and set(item.object_ids) == set(anchor_ids):
            return True
    return False


def _geometric_relation_holds(
    relation: str,
    subject: OracleObject,
    anchors: tuple[OracleObject, ...],
) -> bool:
    if relation == 'near':
        return _horizontal_gap(subject, anchors[0]) <= 1.5
    if relation == 'inside':
        anchor = anchors[0]
        anchor_bottom = anchor.centre_xyz[2] - anchor.dimensions_xyz[2] / 2.0
        anchor_top = anchor.centre_xyz[2] + anchor.dimensions_xyz[2] / 2.0
        return (
            object_footprint(anchor).contains(subject.centre_xyz[:2])
            and anchor_bottom - 0.1 <= subject.centre_xyz[2] <= anchor_top + 0.1
        )
    if relation == 'on':
        anchor = anchors[0]
        anchor_bottom = anchor.centre_xyz[2] - anchor.dimensions_xyz[2] / 2.0
        anchor_top = anchor.centre_xyz[2] + anchor.dimensions_xyz[2] / 2.0
        return (
            object_footprint(anchor, inflation=0.15).contains(
                subject.centre_xyz[:2]
            )
            and anchor_bottom - 0.1
            <= subject.centre_xyz[2]
            <= anchor_top + 0.5
        )
    if relation in {'above', 'below'}:
        anchor = anchors[0]
        vertical_delta = subject.centre_xyz[2] - anchor.centre_xyz[2]
        if relation == 'below':
            vertical_delta = -vertical_delta
        horizontal_limit = max(
            1.0,
            max(anchor.dimensions_xyz[:2]) / 2.0 + 0.75,
        )
        return vertical_delta > 0.05 and _horizontal_gap(
            subject,
            anchor,
        ) <= horizontal_limit
    if relation == 'between':
        left, right = anchors
        segment_x = right.centre_xyz[0] - left.centre_xyz[0]
        segment_y = right.centre_xyz[1] - left.centre_xyz[1]
        squared_length = segment_x ** 2 + segment_y ** 2
        if squared_length <= 1e-9:
            return False
        subject_x = subject.centre_xyz[0] - left.centre_xyz[0]
        subject_y = subject.centre_xyz[1] - left.centre_xyz[1]
        projection = (
            subject_x * segment_x + subject_y * segment_y
        ) / squared_length
        if not 0.05 <= projection <= 0.95:
            return False
        projected_x = left.centre_xyz[0] + projection * segment_x
        projected_y = left.centre_xyz[1] + projection * segment_y
        perpendicular_distance = hypot(
            subject.centre_xyz[0] - projected_x,
            subject.centre_xyz[1] - projected_y,
        )
        return perpendicular_distance <= max(0.75, squared_length ** 0.5 * 0.25)
    return False


def _candidate_anchor_tuples(
    relation: str,
    anchor_sets: list[set[str]],
) -> tuple[tuple[str, ...], ...]:
    if relation == 'between' and len(anchor_sets) == 1:
        return tuple(combinations(sorted(anchor_sets[0], key=_id_sort_key), 2))
    return tuple(product(*(sorted(items, key=_id_sort_key) for items in anchor_sets)))


def _hard_relation_tuples(
    relation: RelationConstraint,
    scene: OracleScene,
    candidates: dict[str, set[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    normalized = _HARD_RELATION_ALIASES.get(relation.relation)
    if normalized is None:
        return None
    objects = _object_by_id(scene)
    subject_candidates = candidates[relation.subject_entity_id]
    anchor_sets = [candidates[item] for item in relation.anchor_entity_ids]
    matches = []
    for subject_id in sorted(subject_candidates, key=_id_sort_key):
        for anchor_ids in _candidate_anchor_tuples(normalized, anchor_sets):
            if subject_id in anchor_ids:
                continue
            if _direct_relation_exists(
                normalized,
                subject_id,
                anchor_ids,
                scene.relations,
            ) or _geometric_relation_holds(
                normalized,
                objects[subject_id],
                tuple(objects[object_id] for object_id in anchor_ids),
            ):
                matches.append((subject_id, anchor_ids))
    return tuple(matches)


def _reject_removed(
    entity_id: str,
    previous: set[str],
    remaining: set[str],
    reason: str,
    reasons: dict[tuple[str, str], list[str]],
) -> None:
    for object_id in previous - remaining:
        object_reasons = reasons.setdefault((entity_id, object_id), [])
        if reason not in object_reasons:
            object_reasons.append(reason)


def _apply_hard_relation(
    relation: RelationConstraint,
    scene: OracleScene,
    candidates: dict[str, set[str]],
    reasons: dict[tuple[str, str], list[str]],
    *,
    strict: bool,
    warnings: list[str],
) -> bool:
    matches = _hard_relation_tuples(relation, scene, candidates)
    if matches is None:
        return False
    description = (
        f'failed {relation.relation}({relation.subject_entity_id}, '
        f'{", ".join(relation.anchor_entity_ids)})'
    )
    if not matches and not strict:
        warnings.append(f'preserved candidates because oracle {description}')
        return False
    changed = False

    subject_previous = set(candidates[relation.subject_entity_id])
    subject_allowed = {subject_id for subject_id, _ in matches}
    candidates[relation.subject_entity_id].intersection_update(subject_allowed)
    _reject_removed(
        relation.subject_entity_id,
        subject_previous,
        candidates[relation.subject_entity_id],
        description,
        reasons,
    )
    changed |= subject_previous != candidates[relation.subject_entity_id]

    if relation.relation == 'between' and len(relation.anchor_entity_ids) == 1:
        entity_id = relation.anchor_entity_ids[0]
        previous = set(candidates[entity_id])
        allowed = {item for _, object_ids in matches for item in object_ids}
        candidates[entity_id].intersection_update(allowed)
        _reject_removed(
            entity_id,
            previous,
            candidates[entity_id],
            description,
            reasons,
        )
        return changed or previous != candidates[entity_id]

    for anchor_index, entity_id in enumerate(relation.anchor_entity_ids):
        previous = set(candidates[entity_id])
        allowed = set()
        for _, object_ids in matches:
            if anchor_index < len(object_ids):
                allowed.add(object_ids[anchor_index])
            if relation.relation == 'between' and len(object_ids) == 2:
                allowed.add(object_ids[1 - anchor_index])
        candidates[entity_id].intersection_update(allowed)
        _reject_removed(
            entity_id,
            previous,
            candidates[entity_id],
            description,
            reasons,
        )
        changed |= previous != candidates[entity_id]
    return changed


def _distance_between(object_a: OracleObject, object_b: OracleObject) -> float:
    return hypot(
        object_a.centre_xyz[0] - object_b.centre_xyz[0],
        object_a.centre_xyz[1] - object_b.centre_xyz[1],
    )


def _apply_ranking_relation(
    relation: RelationConstraint,
    objects: dict[str, OracleObject],
    candidates: dict[str, set[str]],
    reasons: dict[tuple[str, str], list[str]],
) -> bool:
    if relation.relation not in _RANKING_RELATIONS:
        return False
    anchor_ids = {
        item
        for entity_id in relation.anchor_entity_ids
        for item in candidates[entity_id]
    }
    subjects = candidates[relation.subject_entity_id]
    if not anchor_ids or not subjects:
        return False
    distances = {
        subject_id: min(
            _distance_between(objects[subject_id], objects[anchor_id])
            for anchor_id in anchor_ids
            if anchor_id != subject_id
        )
        for subject_id in subjects
        if any(anchor_id != subject_id for anchor_id in anchor_ids)
    }
    if not distances:
        return False
    if relation.relation == 'closest_to':
        best_value = min(distances.values())
    else:
        best_value = max(distances.values())
    tolerance = 1e-9
    selected = {
        object_id
        for object_id, distance in distances.items()
        if abs(distance - best_value) <= tolerance
    }
    previous = set(subjects)
    subjects.intersection_update(selected)
    description = (
        f'not {relation.relation} anchor(s) '
        f'{", ".join(relation.anchor_entity_ids)}'
    )
    _reject_removed(
        relation.subject_entity_id,
        previous,
        subjects,
        description,
        reasons,
    )
    return previous != subjects


def resolve_task_entities(
    task: TaskSpecification,
    scene: OracleScene,
    *,
    strict_relations: bool = True,
) -> EntityResolution:
    """
    Resolve all entity variables with deterministic graph constraints.

    ``strict_relations=False`` retains class/attribute candidates when released
    relation metadata cannot ground a route clause. This is reserved for the
    instruction planner's explicit partial-route fallback; answer solvers are
    always strict.
    """
    candidates, reasons, warnings = _base_candidates(task, scene)
    objects = _object_by_id(scene)
    unsupported_relations = sorted(
        {
            relation.relation
            for relation in task.relations
            if relation.relation not in _HARD_RELATION_ALIASES
            and relation.relation not in _RANKING_RELATIONS
        }
    )
    if unsupported_relations:
        warnings.append(
            f'oracle cannot enforce relations {unsupported_relations}'
        )

    maximum_passes = max(1, len(task.entities) + len(task.relations))
    for _ in range(maximum_passes):
        changed = False
        for relation in task.relations:
            changed |= _apply_hard_relation(
                relation,
                scene,
                candidates,
                reasons,
                strict=strict_relations,
                warnings=warnings,
            )
        for relation in task.relations:
            changed |= _apply_ranking_relation(
                relation,
                objects,
                candidates,
                reasons,
            )
        if not changed:
            break

    ordered_candidates = tuple(
        (
            entity.entity_id,
            tuple(sorted(candidates[entity.entity_id], key=_id_sort_key)),
        )
        for entity in task.entities
    )
    ordered_reasons = tuple(
        (entity_id, object_id, tuple(object_reasons))
        for (entity_id, object_id), object_reasons in sorted(
            reasons.items(),
            key=lambda item: (item[0][0], _id_sort_key(item[0][1])),
        )
    )
    return EntityResolution(
        candidates=ordered_candidates,
        rejection_reasons=ordered_reasons,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _candidate_decisions(
    entity: EntityReference,
    scene: OracleScene,
    resolution: EntityResolution,
) -> tuple[CandidateDecision, ...]:
    accepted = set(resolution.candidates_for(entity.entity_id))
    decisions = []
    class_matches = sorted(
        _class_candidates(entity, scene),
        key=lambda item: _id_sort_key(item.object_id),
    )
    for obj in class_matches:
        reasons = resolution.reasons_for(entity.entity_id, obj.object_id)
        if obj.object_id in accepted:
            reasons = ('accepted',)
        elif not reasons:
            reasons = ('rejected by combined constraints',)
        decisions.append(
            CandidateDecision(
                object_id=obj.object_id,
                accepted=obj.object_id in accepted,
                reasons=reasons,
            )
        )
    return tuple(decisions)


def solve_numerical(
    task: TaskSpecification,
    scene: OracleScene,
) -> NumericalResult:
    """Count unique perfect objects satisfying a numerical task graph."""
    if task.task_type != 'numerical':
        raise OracleReasoningError('numerical solver requires a numerical task')
    target = _target_entity(task)
    resolution = resolve_task_entities(task, scene)
    matching_ids = resolution.candidates_for(target.entity_id)
    warnings = list(resolution.warnings)
    if not matching_ids:
        warnings.append('no target candidate satisfies the task constraints')
    return NumericalResult(
        count=len(matching_ids),
        matching_object_ids=matching_ids,
        candidate_decisions=_candidate_decisions(target, scene, resolution),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def solve_object_reference(
    task: TaskSpecification,
    scene: OracleScene,
) -> ObjectReferenceResult:
    """Select the unique perfect object satisfying an object-reference task."""
    if task.task_type != 'object_reference':
        raise OracleReasoningError(
            'object-reference solver requires an object-reference task'
        )
    target = _target_entity(task)
    resolution = resolve_task_entities(task, scene)
    matching_ids = resolution.candidates_for(target.entity_id)
    warnings = list(resolution.warnings)
    selected_id = matching_ids[0] if matching_ids else None
    if not matching_ids:
        warnings.append('no target candidate satisfies the task constraints')
    elif len(matching_ids) > 1:
        warnings.append(
            f'expected a unique target but found {len(matching_ids)} candidates; '
            'selected the lowest stable object ID'
        )
    objects = _object_by_id(scene)
    decisions = _candidate_decisions(target, scene, resolution)
    scores = tuple(
        (decision.object_id, 1.0 if decision.accepted else 0.0)
        for decision in decisions
    )
    return ObjectReferenceResult(
        selected_object_id=selected_id,
        selected_object=objects[selected_id] if selected_id is not None else None,
        candidate_scores=scores,
        candidate_decisions=decisions,
        confidence_margin=1.0 if len(matching_ids) == 1 else 0.0,
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    'CandidateDecision',
    'EntityResolution',
    'NumericalResult',
    'ObjectReferenceResult',
    'OracleReasoningError',
    'resolve_task_entities',
    'solve_numerical',
    'solve_object_reference',
]
