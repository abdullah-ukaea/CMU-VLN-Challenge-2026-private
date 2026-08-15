"""Count persistent target IDs after complete class/colour/relation filtering."""

from dataclasses import dataclass
from itertools import islice
from itertools import product
from math import isfinite

import cv2

from qmapnav.common import RelationConstraint
from qmapnav.common import TaskSpecification
from qmapnav.counting.anchor_ambiguity import AnchorCountHypothesis
from qmapnav.counting.anchor_ambiguity import assess_anchor_counts
from qmapnav.counting.numerical_result import CountDiagnostic
from qmapnav.counting.numerical_result import NumericalResult
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.reasoning.candidate_generation import CandidateGenerationConfig
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
from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_relations import on_evidence
from qmapnav.reasoning.support_relations import SupportRelationConfig
from qmapnav.reasoning.vertical_relations import above_evidence
from qmapnav.reasoning.vertical_relations import below_evidence
from qmapnav.reasoning.vertical_relations import VerticalRelationConfig


@dataclass(frozen=True)
class NumericalSolverConfig:
    """Thresholds and a hard bound for persistent numerical reasoning."""

    definite_class_probability: float = 0.65
    probable_class_probability: float = 0.15
    reject_colour_probability: float = 0.08
    probable_colour_probability: float = 0.15
    definite_colour_probability: float = 0.55
    probable_relation_confidence: float = 0.25
    definite_relation_confidence: float = 0.60
    max_complete_hypotheses: int = 50_000

    def __post_init__(self) -> None:
        values = (
            self.definite_class_probability,
            self.probable_class_probability,
            self.reject_colour_probability,
            self.probable_colour_probability,
            self.definite_colour_probability,
            self.probable_relation_confidence,
            self.definite_relation_confidence,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError('numerical thresholds must lie in [0, 1]')
        if self.probable_class_probability > self.definite_class_probability:
            raise ValueError('probable class threshold must not exceed definite')
        if not (
            self.reject_colour_probability
            <= self.probable_colour_probability
            <= self.definite_colour_probability
        ):
            raise ValueError('colour thresholds are out of order')
        if (
            self.probable_relation_confidence
            > self.definite_relation_confidence
        ):
            raise ValueError('relation thresholds are out of order')
        if (
            isinstance(self.max_complete_hypotheses, bool)
            or self.max_complete_hypotheses <= 0
        ):
            raise ValueError('max_complete_hypotheses must be positive')


@dataclass(frozen=True)
class _EvaluatedAssignment:
    """Internal complete role assignment retained for set aggregation."""

    target_id: int
    role_ids: tuple[tuple[str, str], ...]
    classification: str
    score: float
    confidence: float
    relation_score: float
    reasons: tuple[str, ...]
    evidence: dict[str, float]


def resolve_numerical_from_maps(
    task: TaskSpecification,
    object_map: ObjectMap,
    structural_map: StructuralMap,
    *,
    config: NumericalSolverConfig | None = None,
    candidate_config: CandidateGenerationConfig | None = None,
    spatial_config: SpatialRelationConfig | None = None,
    vertical_config: VerticalRelationConfig | None = None,
    support_config: SupportRelationConfig | None = None,
    scoring_config: HypothesisScoringConfig | None = None,
) -> NumericalResult:
    """Resolve one numerical task exclusively from persistent episode maps."""
    if not isinstance(task, TaskSpecification):
        raise TypeError('task must be TaskSpecification')
    if task.task_type != 'numerical' or not task.entities:
        raise ValueError('task must contain a numerical target')
    if not isinstance(object_map, ObjectMap):
        raise TypeError('object_map must be ObjectMap')
    if not isinstance(structural_map, StructuralMap):
        raise TypeError('structural_map must be StructuralMap')
    policy = config or NumericalSolverConfig()
    candidate_policy = candidate_config or CandidateGenerationConfig(
        minimum_class_probability=policy.probable_class_probability,
    )
    spatial_policy = spatial_config or SpatialRelationConfig()
    vertical_policy = vertical_config or VerticalRelationConfig()
    support_policy = support_config or SupportRelationConfig()
    scoring_policy = scoring_config or HypothesisScoringConfig()
    generated = {
        entity.entity_id: generate_candidates_from_maps(
            entity, object_map, structural_map, candidate_policy
        )
        for entity in task.entities
    }
    target = task.entities[0]
    target_candidates = tuple(
        item for item in generated[target.entity_id].candidates
        if item.source_type == 'object'
    )
    pools = []
    missing_roles = []
    for entity in task.entities:
        pool = generated[entity.entity_id].retained
        if entity.entity_id == target.entity_id:
            pool = tuple(item for item in pool if item.source_type == 'object')
        pools.append(tuple(pool))
        if not pool:
            missing_roles.append(entity.entity_id)
    ranking_evidence = _ranking_evidence(
        task, generated, spatial_policy
    )
    assignments = []
    limit_reached = False
    if not missing_roles:
        iterator = product(*pools)
        for values in islice(iterator, policy.max_complete_hypotheses + 1):
            if len(assignments) >= policy.max_complete_hypotheses:
                limit_reached = True
                break
            assignment = {
                entity.entity_id: candidate
                for entity, candidate in zip(task.entities, values)
            }
            if _relation_participants_reuse_candidate(task, assignment):
                continue
            assignments.append(_evaluate_assignment(
                task,
                assignment,
                target.entity_id,
                policy,
                candidate_policy,
                spatial_policy,
                vertical_policy,
                support_policy,
                scoring_policy,
                ranking_evidence,
            ))
    diagnostics = _target_diagnostics(
        target,
        target_candidates,
        assignments,
        missing_roles,
        policy,
    )
    partitions = {
        name: tuple(sorted(
            item.instance_id for item in diagnostics
            if item.classification == name
        ))
        for name in ('definite', 'probable', 'rejected', 'unresolved')
    }
    ambiguity = _anchor_assessment(
        task,
        target.entity_id,
        generated,
        assignments,
    )
    qualifying = partitions['definite'] + partitions['probable']
    confidence = _count_confidence(
        diagnostics,
        ambiguity.count_consistent,
        limit_reached,
    )
    reason = 'awaiting_temporal_and_viewpoint_stability'
    if missing_roles:
        reason = 'missing_candidate_roles:' + ','.join(sorted(missing_roles))
    if limit_reached:
        reason = 'bounded_hypothesis_limit_reached'
    return NumericalResult(
        target.class_name,
        partitions['definite'],
        partitions['probable'],
        partitions['rejected'],
        partitions['unresolved'],
        len(set(qualifying)),
        confidence,
        False,
        reason,
        diagnostics,
        ambiguity,
        limit_reached,
    )


def _ranking_evidence(task, generated, spatial_policy):
    output = {}
    for index, relation in enumerate(task.relations):
        if relation.relation not in {'closest_to', 'farthest_from'}:
            continue
        subjects = [
            item.geometry
            for item in generated[relation.subject_entity_id].retained
            if item.geometry is not None
        ]
        anchors = [
            item.geometry
            for item in generated[relation.anchor_entity_ids[0]].retained
            if item.geometry is not None
        ]
        operator = 'closest' if relation.relation == 'closest_to' else 'farthest'
        ranking = rank_distances(subjects, anchors, operator, spatial_policy)
        output[index] = (
            {(item.target_id, item.anchor_id): item for item in ranking.ranked},
            ranking,
        )
    return output


def _relation_participants_reuse_candidate(task, assignment):
    for relation in task.relations:
        roles = (relation.subject_entity_id, *relation.anchor_entity_ids)
        identifiers = tuple(assignment[role].candidate_id for role in roles)
        if len(identifiers) != len(set(identifiers)):
            return True
    return False


def _evaluate_assignment(
    task,
    assignment,
    target_reference_id,
    policy,
    candidate_policy,
    spatial_policy,
    vertical_policy,
    support_policy,
    scoring_policy,
    ranking_evidence,
):
    evaluations = []
    for entity in task.entities:
        candidate = assignment[entity.entity_id]
        evaluations.extend(
            _prefix(f'{entity.entity_id}.{item.constraint_name}', item)
            for item in candidate_constraints(
                candidate, entity, candidate_policy
            )
        )
    relation_evaluations = []
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
        evaluation = _prefix(
            f'{relation.subject_entity_id}.{relation.relation}.{index}',
            evaluation,
        )
        relation_evaluations.append(evaluation)
        evaluations.append(evaluation)
    roles = tuple(
        (entity.entity_id, assignment[entity.entity_id].candidate_id)
        for entity in task.entities
    )
    scored = score_complete_hypothesis(
        CompleteHypothesis(tuple(
            (role, f'{role}:{candidate_id}')
            for role, candidate_id in roles
        ), tuple(evaluations)),
        config=scoring_policy,
    )
    target = assignment[target_reference_id]
    classification, reasons = _classify_assignment(
        target,
        task.entities[0],
        evaluations,
        relation_evaluations,
        policy,
    )
    relation_score = (
        sum(item.score for item in relation_evaluations)
        / len(relation_evaluations)
        if relation_evaluations else 1.0
    )
    evidence = dict(scored.evidence)
    evidence['relation_score'] = relation_score
    return _EvaluatedAssignment(
        int(target.candidate_id),
        roles,
        classification,
        scored.score,
        scored.confidence,
        relation_score,
        reasons,
        evidence,
    )


def _classify_assignment(target, reference, evaluations, relations, policy):
    requested_colour = reference.attributes.get('colour')
    reasons = []
    if target.class_probability < policy.probable_class_probability:
        return 'rejected', ('class_probability_rejected',)
    definite = target.class_probability >= policy.definite_class_probability
    reasons.append(
        'class_definite' if definite else 'class_probable'
    )
    if requested_colour is not None:
        colour = target.colour_probability
        if colour is None:
            return 'unresolved', tuple(reasons + ['colour_unavailable'])
        if colour < policy.reject_colour_probability:
            return 'rejected', tuple(reasons + ['colour_rejected'])
        if colour < policy.probable_colour_probability:
            return 'unresolved', tuple(reasons + ['colour_weak'])
        colour_definite = colour >= policy.definite_colour_probability
        definite = definite and colour_definite
        reasons.append('colour_definite' if colour_definite else 'colour_probable')
    if any(item.satisfied is False for item in relations):
        return 'rejected', tuple(reasons + ['relation_rejected'])
    if any(item.satisfied is None for item in relations):
        return 'unresolved', tuple(reasons + ['relation_unresolved'])
    if any(
        item.satisfied is None
        for item in evaluations
        if item.constraint_name.endswith('.geometry')
    ):
        return 'unresolved', tuple(reasons + ['geometry_unresolved'])
    if relations:
        minimum = min(item.confidence for item in relations)
        if minimum < policy.probable_relation_confidence:
            return 'unresolved', tuple(reasons + ['relation_confidence_weak'])
        relation_definite = minimum >= policy.definite_relation_confidence
        definite = definite and relation_definite
        reasons.append(
            'relations_definite' if relation_definite else 'relations_probable'
        )
    return (
        ('definite', tuple(reasons + ['qualifying_definite']))
        if definite
        else ('probable', tuple(reasons + ['qualifying_probable']))
    )


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
        return _counting_vertical_evidence(
            'above', geometries[0], geometries[1], vertical_policy
        )
    if relation.relation == 'below':
        return _counting_vertical_evidence(
            'below', geometries[0], geometries[1], vertical_policy
        )
    if relation.relation == 'on':
        return _counting_on_evidence(
            geometries[0], geometries[1], support_policy
        )
    if relation.relation in {'closest_to', 'farthest_from'}:
        by_pair, ranking = ranking_evidence.get(index, ({}, None))
        item = by_pair.get((subject.candidate_id, anchors[0].candidate_id))
        if item is None or ranking is None or not ranking.ranked:
            return ConstraintEvaluation(
                relation.relation, 0.0, False, None, 0.0,
                {'ranking_available': 0.0},
            )
        best = ranking.ranked[0]
        return ConstraintEvaluation(
            relation.relation,
            item.score,
            False,
            (
                item.target_id == best.target_id
                and item.anchor_id == best.anchor_id
            ),
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


def _physical_evidence(evidence, *, uncertain_status=None):
    satisfied = evidence.accepted
    if uncertain_status is not None and evidence.status == uncertain_status:
        satisfied = None
    return ConstraintEvaluation(
        evidence.relation,
        evidence.confidence,
        False,
        satisfied,
        evidence.geometry_confidence,
        {
            'vertical_gap_m': evidence.vertical_gap_m,
            'subject_support_overlap': evidence.subject_support_overlap,
            'horizontal_distance_m': evidence.horizontal_distance_m,
            'geometry_confidence': evidence.geometry_confidence,
        },
    )


def _counting_on_evidence(subject, anchor, policy):
    evidence = on_evidence(subject, anchor, policy)
    if evidence.accepted:
        return _physical_evidence(evidence)
    if evidence.status == 'uncertain':
        return ConstraintEvaluation(
            'on',
            evidence.confidence,
            False,
            True,
            evidence.geometry_confidence,
            {
                'vertical_gap_m': evidence.vertical_gap_m,
                'subject_support_overlap': evidence.subject_support_overlap,
                'horizontal_distance_m': evidence.horizontal_distance_m,
                'geometry_confidence': evidence.geometry_confidence,
                'probable_support_evidence': 1.0,
            },
        )
    signed_distance = float(cv2.pointPolygonTest(
        anchor.footprint_xy.astype('float32'),
        tuple(map(float, subject.centre_xyz[:2])),
        True,
    ))
    centre_supported = signed_distance >= -0.15
    vertical_volume = (
        anchor.bottom_z - 0.10
        <= subject.centre_xyz[2]
        <= anchor.top_z + 0.50
    )
    if not (centre_supported and vertical_volume):
        return _physical_evidence(evidence)
    confidence = min(subject.confidence, anchor.confidence) * 0.55
    return ConstraintEvaluation(
        'on',
        confidence,
        False,
        True,
        confidence,
        {
            'vertical_gap_m': evidence.vertical_gap_m,
            'subject_support_overlap': evidence.subject_support_overlap,
            'horizontal_distance_m': evidence.horizontal_distance_m,
            'geometry_confidence': evidence.geometry_confidence,
            'centre_footprint_signed_distance_m': signed_distance,
            'embedded_volume_fallback': 1.0,
        },
    )


def _counting_vertical_evidence(relation, subject, anchor, policy):
    evidence = (
        above_evidence(subject, anchor, policy)
        if relation == 'above'
        else below_evidence(subject, anchor, policy)
    )
    if evidence.accepted:
        return _physical_evidence(evidence)
    vertical_delta = subject.centre_xyz[2] - anchor.centre_xyz[2]
    if relation == 'below':
        vertical_delta = -vertical_delta
    horizontal_limit = max(
        1.0,
        float(max(anchor.dimensions_xyz[:2])) / 2.0 + 0.75,
    )
    horizontal_gap = footprint_metrics(subject, anchor).edge_distance_m
    if vertical_delta <= 0.05 or horizontal_gap > horizontal_limit:
        return _physical_evidence(evidence)
    confidence = min(subject.confidence, anchor.confidence) * 0.55
    return ConstraintEvaluation(
        relation,
        confidence,
        False,
        True,
        confidence,
        {
            'vertical_gap_m': evidence.vertical_gap_m,
            'subject_support_overlap': evidence.subject_support_overlap,
            'horizontal_distance_m': horizontal_gap,
            'geometry_confidence': evidence.geometry_confidence,
            'volume_box_vertical_fallback': 1.0,
        },
    )


def _prefix(name, evaluation):
    return ConstraintEvaluation(
        name,
        evaluation.score,
        evaluation.is_hard,
        evaluation.satisfied,
        evaluation.confidence,
        evaluation.evidence,
    )


def _target_diagnostics(
    target,
    candidates,
    assignments,
    missing_roles,
    policy,
):
    by_target = {}
    for assignment in assignments:
        by_target.setdefault(assignment.target_id, []).append(assignment)
    output = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        identifier = int(candidate.candidate_id)
        options = by_target.get(identifier, [])
        if not candidate.retained:
            classification = 'rejected'
            reasons = ('class_probability_rejected',)
            score = candidate.class_probability
            confidence = candidate.class_probability
            relation_score = 0.0
            roles = {target.entity_id: candidate.candidate_id}
            evidence = {'class_probability': candidate.class_probability}
        elif not options:
            classification = 'unresolved' if missing_roles else 'rejected'
            reasons = (
                ('missing_candidate_roles:' + ','.join(sorted(missing_roles)),)
                if missing_roles else ('no_complete_relation_hypothesis',)
            )
            score = candidate.class_probability
            confidence = min(candidate.geometry_confidence, candidate.class_probability)
            relation_score = 0.0
            roles = {target.entity_id: candidate.candidate_id}
            evidence = {'class_probability': candidate.class_probability}
        else:
            best = min(options, key=_assignment_key)
            classification = best.classification
            reasons = best.reasons
            score = best.score
            confidence = best.confidence
            relation_score = best.relation_score
            roles = dict(best.role_ids)
            evidence = best.evidence
        output.append(CountDiagnostic(
            identifier,
            classification,
            candidate.class_probability,
            candidate.colour_probability,
            relation_score,
            score,
            max(0.0, min(1.0, confidence)),
            reasons,
            roles,
            evidence,
        ))
    return tuple(output)


def _assignment_key(item):
    rank = {'definite': 0, 'probable': 1, 'unresolved': 2, 'rejected': 3}
    return (
        rank[item.classification],
        -item.score,
        -item.confidence,
        item.role_ids,
    )


def _primary_anchor_roles(task, target_reference_id):
    roles = []
    for relation in task.relations:
        if relation.subject_entity_id == target_reference_id:
            roles.extend(relation.anchor_entity_ids)
    return tuple(dict.fromkeys(roles))


def _anchor_assessment(task, target_reference_id, generated, assignments):
    roles = _primary_anchor_roles(task, target_reference_id)
    signatures = {}
    if roles:
        pools = [generated[role].retained for role in roles]
        if all(pools):
            for values in product(*pools):
                signature = tuple(sorted(
                    (role, value.candidate_id)
                    for role, value in zip(roles, values)
                ))
                scores = [value.class_probability for value in values]
                signatures[signature] = {
                    'ids': set(),
                    'score': sum(scores) / len(scores),
                    'confidence': min(
                        value.geometry_confidence for value in values
                    ),
                }
    else:
        signatures[()] = {'ids': set(), 'score': 1.0, 'confidence': 1.0}
    for item in assignments:
        role_mapping = dict(item.role_ids)
        signature = tuple(sorted(
            (role, role_mapping[role]) for role in roles
        ))
        state = signatures.setdefault(signature, {
            'ids': set(), 'score': item.score, 'confidence': item.confidence,
        })
        state['score'] = max(state['score'], item.score)
        state['confidence'] = max(state['confidence'], item.confidence)
        if item.classification in {'definite', 'probable'}:
            state['ids'].add(item.target_id)
    hypotheses = tuple(
        AnchorCountHypothesis(
            signature,
            tuple(sorted(state['ids'])),
            state['score'],
            state['confidence'],
        )
        for signature, state in sorted(signatures.items())
    )
    return assess_anchor_counts(hypotheses)


def _count_confidence(diagnostics, anchor_consistent, limit_reached):
    qualifying = [
        item.confidence for item in diagnostics
        if item.classification in {'definite', 'probable'}
    ]
    unresolved = sum(
        item.classification == 'unresolved' for item in diagnostics
    )
    if qualifying:
        confidence = sum(qualifying) / len(qualifying)
    elif unresolved:
        confidence = 0.35
    else:
        confidence = 0.60
    if not anchor_consistent:
        confidence *= 0.65
    if limit_reached:
        confidence *= 0.50
    confidence *= max(0.50, 1.0 - 0.10 * unresolved)
    return max(0.0, min(1.0, confidence))


__all__ = ['NumericalSolverConfig', 'resolve_numerical_from_maps']
