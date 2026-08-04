"""Complete Day 10 object-reference resolution over persistent maps."""

from dataclasses import dataclass
from dataclasses import replace
from itertools import product
from types import MappingProxyType
from typing import Mapping

from qmapnav.common import RelationConstraint
from qmapnav.common import TaskSpecification
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.reasoning.ambiguity import AmbiguityConfig
from qmapnav.reasoning.candidate_generation import CandidateGenerationConfig
from qmapnav.reasoning.candidate_generation import CandidateGenerationResult
from qmapnav.reasoning.candidate_generation import generate_candidates_from_maps
from qmapnav.reasoning.hypothesis_scoring import CompleteHypothesis
from qmapnav.reasoning.hypothesis_scoring import HypothesisScoringConfig
from qmapnav.reasoning.hypothesis_scoring import score_complete_hypothesis
from qmapnav.reasoning.reference_resolver import candidate_constraints
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation
from qmapnav.reasoning.spatial_relations import evaluate_between
from qmapnav.reasoning.spatial_relations import evaluate_near
from qmapnav.reasoning.spatial_relations import rank_distances
from qmapnav.reasoning.spatial_relations import SpatialRelationConfig
from qmapnav.reasoning.support_relations import on_evidence
from qmapnav.reasoning.support_relations import SupportRelationConfig
from qmapnav.reasoning.vertical_relations import above_evidence
from qmapnav.reasoning.vertical_relations import below_evidence
from qmapnav.reasoning.vertical_relations import VerticalRelationConfig


@dataclass(frozen=True)
class RankedObjectReferenceHypothesis:
    """Best complete anchor assignment retained for one target instance."""

    target_id: str
    role_ids: Mapping[str, str]
    score: float
    confidence: float
    satisfied_constraints: tuple[str, ...]
    violated_constraints: tuple[str, ...]
    unresolved_constraints: tuple[str, ...]
    evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'role_ids', MappingProxyType(dict(self.role_ids))
        )
        object.__setattr__(
            self, 'evidence', MappingProxyType(dict(self.evidence))
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready score decomposition."""
        return {
            'target_id': self.target_id,
            'role_ids': dict(sorted(self.role_ids.items())),
            'score': self.score,
            'confidence': self.confidence,
            'satisfied_constraints': list(self.satisfied_constraints),
            'violated_constraints': list(self.violated_constraints),
            'unresolved_constraints': list(self.unresolved_constraints),
            'evidence': dict(sorted(self.evidence.items())),
        }


@dataclass(frozen=True)
class PerceivedObjectReferenceResolution:
    """Ranked persistent targets plus explicit bounded-fallback state."""

    target_reference_id: str
    candidate_generation: Mapping[str, CandidateGenerationResult]
    ranked_hypotheses: tuple[RankedObjectReferenceHypothesis, ...]
    selected_target_id: str | None
    confidence_margin: float
    normalized_margin: float
    resolution_status: str
    unresolved_constraints: tuple[str, ...]
    used_fallback: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'candidate_generation',
            MappingProxyType(dict(self.candidate_generation)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return complete candidate and ranking evidence."""
        return {
            'target_reference_id': self.target_reference_id,
            'candidate_generation': {
                key: value.to_dict()
                for key, value in sorted(self.candidate_generation.items())
            },
            'ranked_hypotheses': [
                item.to_dict() for item in self.ranked_hypotheses
            ],
            'selected_target_id': self.selected_target_id,
            'confidence_margin': self.confidence_margin,
            'normalized_margin': self.normalized_margin,
            'resolution_status': self.resolution_status,
            'unresolved_constraints': list(self.unresolved_constraints),
            'used_fallback': self.used_fallback,
        }


def resolve_object_reference_from_maps(
    task: TaskSpecification,
    object_map: ObjectMap,
    structural_map: StructuralMap,
    *,
    candidate_config: CandidateGenerationConfig | None = None,
    spatial_config: SpatialRelationConfig | None = None,
    vertical_config: VerticalRelationConfig | None = None,
    support_config: SupportRelationConfig | None = None,
    scoring_config: HypothesisScoringConfig | None = None,
    ambiguity_config: AmbiguityConfig | None = None,
) -> PerceivedObjectReferenceResolution:
    """Evaluate all target-anchor products and commit a bounded fallback."""
    if not isinstance(task, TaskSpecification):
        raise TypeError('task must be TaskSpecification')
    if task.task_type != 'object_reference' or not task.entities:
        raise ValueError('task must contain an object-reference target')
    target_reference = task.entities[0]
    candidate_policy = candidate_config or CandidateGenerationConfig()
    spatial_policy = spatial_config or SpatialRelationConfig()
    vertical_policy = vertical_config or VerticalRelationConfig()
    support_policy = support_config or SupportRelationConfig()
    scoring_policy = scoring_config or HypothesisScoringConfig()
    ambiguity_policy = ambiguity_config or AmbiguityConfig()
    generated = {
        entity.entity_id: generate_candidates_from_maps(
            entity, object_map, structural_map, candidate_policy
        )
        for entity in task.entities
    }
    target_pool = tuple(
        item for item in generated[target_reference.entity_id].retained
        if item.source_type == 'object'
    )
    if not target_pool:
        return _empty_resolution(target_reference.entity_id, generated)
    pools = []
    for entity in task.entities:
        pool = (
            target_pool
            if entity.entity_id == target_reference.entity_id
            else generated[entity.entity_id].retained
        )
        if not pool:
            return _missing_anchor_resolution(
                target_reference.entity_id, generated, entity.entity_id,
                target_pool,
            )
        pools.append(pool)
    ranking_evidence = _ranking_evidence(
        task,
        generated,
        target_reference,
        spatial_policy,
    )
    complete = []
    for values in product(*pools):
        assignment = {
            entity.entity_id: candidate
            for entity, candidate in zip(task.entities, values)
        }
        if _relation_participants_reuse_candidate(task, assignment):
            continue
        evaluations = _evaluate_assignment(
            task,
            assignment,
            candidate_policy,
            spatial_policy,
            vertical_policy,
            support_policy,
            ranking_evidence,
        )
        role_ids = tuple(
            (entity.entity_id, assignment[entity.entity_id].candidate_id)
            for entity in task.entities
        )
        scoring_role_ids = tuple(
            (role_id, f'{role_id}:{candidate_id}')
            for role_id, candidate_id in role_ids
        )
        scored = score_complete_hypothesis(
            CompleteHypothesis(scoring_role_ids, evaluations),
            config=scoring_policy,
        )
        complete.append((assignment, scored))
    best_by_target = {}
    for assignment, scored in complete:
        target_id = assignment[target_reference.entity_id].candidate_id
        hypothesis = RankedObjectReferenceHypothesis(
            target_id,
            {
                entity.entity_id: assignment[entity.entity_id].candidate_id
                for entity in task.entities
            },
            scored.score,
            scored.confidence,
            scored.satisfied_constraints,
            scored.violated_constraints,
            scored.unresolved_constraints,
            scored.evidence,
        )
        previous = best_by_target.get(target_id)
        if previous is None or _hypothesis_key(hypothesis) < (
            _hypothesis_key(previous)
        ):
            best_by_target[target_id] = hypothesis
    ranked = tuple(sorted(best_by_target.values(), key=_hypothesis_key))
    return _final_resolution(
        target_reference,
        task,
        generated,
        ranked,
        ambiguity_policy,
    )


def _relation_participants_reuse_candidate(task, assignment):
    """Reject self-relations while allowing repeated mentions of one anchor."""
    for relation in task.relations:
        participant_ids = (
            relation.subject_entity_id,
            *relation.anchor_entity_ids,
        )
        candidate_ids = tuple(
            assignment[item].candidate_id for item in participant_ids
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            return True
    return False


def _ranking_evidence(task, generated, target, spatial_policy):
    output = {}
    for index, relation in enumerate(task.relations):
        if relation.relation not in {'closest_to', 'farthest_from'}:
            continue
        subject_pool = generated[relation.subject_entity_id].retained
        anchor_pool = generated[relation.anchor_entity_ids[0]].retained
        subjects = [item.geometry for item in subject_pool if item.geometry]
        anchors = [item.geometry for item in anchor_pool if item.geometry]
        operator = (
            'closest' if relation.relation == 'closest_to' else 'farthest'
        )
        ranking = rank_distances(subjects, anchors, operator, spatial_policy)
        output[index] = {
            (item.target_id, item.anchor_id): item for item in ranking.ranked
        }, ranking
    return output


def _evaluate_assignment(
    task,
    assignment,
    candidate_policy,
    spatial_policy,
    vertical_policy,
    support_policy,
    ranking_evidence,
):
    evaluations = []
    for entity in task.entities:
        for evaluation in candidate_constraints(
            assignment[entity.entity_id], entity, candidate_policy
        ):
            evaluations.append(_prefix_evaluation(
                f'{entity.entity_id}.{evaluation.constraint_name}', evaluation
            ))
    for index, relation in enumerate(task.relations):
        evaluation = _evaluate_relation(
            index,
            relation,
            assignment,
            spatial_policy,
            vertical_policy,
            support_policy,
            ranking_evidence,
        )
        evaluations.append(_prefix_evaluation(
            f'{relation.subject_entity_id}.{relation.relation}.{index}',
            evaluation,
        ))
    return tuple(evaluations)


def _evaluate_relation(
    index,
    relation: RelationConstraint,
    assignment,
    spatial_policy,
    vertical_policy,
    support_policy,
    ranking_evidence,
):
    subject = assignment[relation.subject_entity_id]
    anchors = [assignment[item] for item in relation.anchor_entity_ids]
    geometries = [subject.geometry] + [item.geometry for item in anchors]
    if any(item is None for item in geometries):
        return ConstraintEvaluation(
            relation.relation, 0.0, False, None, 0.0,
            {'geometry_available': 0.0},
        )
    if relation.relation == 'near':
        return evaluate_near(geometries[0], geometries[1], spatial_policy)
    if relation.relation == 'between' and len(geometries) == 3:
        return evaluate_between(
            geometries[0], geometries[1], geometries[2], spatial_policy
        )
    if relation.relation == 'above':
        return _relation_evidence_to_constraint(
            above_evidence(geometries[0], geometries[1], vertical_policy)
        )
    if relation.relation == 'below':
        return _relation_evidence_to_constraint(
            below_evidence(geometries[0], geometries[1], vertical_policy)
        )
    if relation.relation == 'on':
        return _relation_evidence_to_constraint(
            on_evidence(geometries[0], geometries[1], support_policy)
        )
    if relation.relation in {'closest_to', 'farthest_from'}:
        by_pair, ranking = ranking_evidence.get(index, ({}, None))
        item = by_pair.get((subject.candidate_id, anchors[0].candidate_id))
        if item is None or ranking is None:
            return ConstraintEvaluation(
                relation.relation, 0.0, False, None, 0.0,
                {'ranking_available': 0.0},
            )
        best = ranking.ranked[0]
        satisfied = (
            item.target_id == best.target_id
            and item.anchor_id == best.anchor_id
        )
        return ConstraintEvaluation(
            relation.relation,
            item.score,
            False,
            satisfied,
            item.confidence,
            {
                'distance_m': item.distance_m,
                'ranking_margin': ranking.raw_margin,
                'ranking_normalized_margin': ranking.normalized_margin,
            },
        )
    return ConstraintEvaluation(
        relation.relation, 0.0, False, None, 0.0,
        {'unsupported_relation': 1.0},
    )


def _prefix_evaluation(name, evaluation):
    return replace(evaluation, constraint_name=name)


def _relation_evidence_to_constraint(evidence):
    return ConstraintEvaluation(
        evidence.relation,
        evidence.confidence,
        False,
        evidence.accepted,
        evidence.geometry_confidence,
        {
            'vertical_gap_m': evidence.vertical_gap_m,
            'subject_support_overlap': evidence.subject_support_overlap,
            'horizontal_distance_m': evidence.horizontal_distance_m,
            'geometry_confidence': evidence.geometry_confidence,
        },
    )


def _hypothesis_key(item):
    return (-item.score, -item.confidence, item.target_id,
            tuple(sorted(item.role_ids.items())))


def _empty_resolution(target_id, generated):
    return PerceivedObjectReferenceResolution(
        target_id, generated, (), None, 0.0, 0.0, 'no_candidates',
        ('target_candidate_missing',), True,
    )


def _missing_anchor_resolution(target_id, generated, missing_id, target_pool):
    ranked = tuple(
        RankedObjectReferenceHypothesis(
            item.candidate_id,
            {target_id: item.candidate_id},
            item.class_probability,
            item.geometry_confidence,
            ('target.class',),
            (),
            (f'{missing_id}.candidate_missing',),
            {
                'target.class': item.class_probability,
                'target.geometry': item.geometry_confidence,
            },
        )
        for item in sorted(
            target_pool,
            key=lambda value: (
                -value.class_probability,
                -value.geometry_confidence,
                value.candidate_id,
            ),
        )
    )
    return PerceivedObjectReferenceResolution(
        target_id,
        generated,
        ranked,
        ranked[0].target_id if ranked else None,
        0.0,
        0.0,
        'low_confidence',
        (f'{missing_id}.candidate_missing',),
        True,
    )


def _final_resolution(target, task, generated, ranked, policy):
    if not ranked:
        return _empty_resolution(target.entity_id, generated)
    top = ranked[0]
    if len(ranked) == 1:
        raw_margin = max(0.0, top.score)
        normalized = 1.0
    else:
        raw_margin = top.score - ranked[1].score
        normalized = raw_margin / (abs(top.score) + 1.0e-9)
    unresolved = tuple(sorted({
        name for item in ranked for name in item.unresolved_constraints
    }))
    underconstrained = (
        len(ranked) > 1
        and not target.attributes
        and not task.relations
    )
    if underconstrained:
        status = 'underconstrained'
    elif top.score < policy.resolved_minimum_score or (
        top.confidence < policy.resolved_minimum_score
    ):
        status = 'low_confidence'
    elif len(ranked) > 1 and (
        raw_margin <= policy.ambiguous_margin
        or normalized < policy.resolved_minimum_margin
        or raw_margin < policy.resolved_minimum_margin
    ):
        status = 'ambiguous'
    else:
        status = 'resolved'
    return PerceivedObjectReferenceResolution(
        target.entity_id,
        generated,
        ranked,
        top.target_id,
        raw_margin,
        normalized,
        status,
        unresolved,
        status != 'resolved',
    )


__all__ = [
    'PerceivedObjectReferenceResolution',
    'RankedObjectReferenceHypothesis',
    'resolve_object_reference_from_maps',
]
