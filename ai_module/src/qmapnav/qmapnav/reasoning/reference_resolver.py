"""Adapt candidate evidence to explicit ranked reference resolutions."""

from typing import Mapping, Sequence

from qmapnav.common import EntityReference
from qmapnav.reasoning.ambiguity import AmbiguityAssessment
from qmapnav.reasoning.ambiguity import AmbiguityConfig
from qmapnav.reasoning.ambiguity import assess_ambiguity
from qmapnav.reasoning.candidate_generation import CandidateGenerationConfig
from qmapnav.reasoning.candidate_generation import CandidateGenerationResult
from qmapnav.reasoning.candidate_generation import EntityCandidate
from qmapnav.reasoning.hypothesis_scoring import CompleteHypothesis
from qmapnav.reasoning.hypothesis_scoring import HypothesisScoringConfig
from qmapnav.reasoning.hypothesis_scoring import rank_complete_hypotheses
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation
from qmapnav.reasoning.resolution_contracts import SetResolution
from qmapnav.reasoning.spatial_relations import DistanceRanking


def candidate_constraints(
    candidate: EntityCandidate,
    reference: EntityReference,
    config: CandidateGenerationConfig | None = None,
) -> tuple[ConstraintEvaluation, ...]:
    """Translate class, colour, and geometry evidence into constraints."""
    policy = config or CandidateGenerationConfig()
    class_satisfied = (
        candidate.class_probability >= policy.minimum_class_probability
    )
    output = [ConstraintEvaluation(
        'class',
        candidate.class_probability,
        True,
        class_satisfied,
        candidate.class_probability,
        {'class_probability': candidate.class_probability},
    )]
    requested_colour = reference.attributes.get('colour')
    if requested_colour is not None:
        probability = candidate.colour_probability
        output.append(ConstraintEvaluation(
            'colour',
            0.0 if probability is None else probability,
            False,
            None if probability is None else (
                probability >= policy.minimum_colour_probability
            ),
            0.0 if probability is None else probability,
            {
                'colour_probability': (
                    0.0 if probability is None else probability
                ),
                'colour_available': float(probability is not None),
            },
        ))
    geometry_satisfied = (
        candidate.geometry_confidence >= policy.minimum_geometry_confidence
    )
    output.append(ConstraintEvaluation(
        'geometry',
        candidate.geometry_confidence,
        False,
        True if geometry_satisfied else None,
        candidate.geometry_confidence,
        {'geometry_confidence': candidate.geometry_confidence},
    ))
    return tuple(output)


def resolve_single_reference(
    reference: EntityReference,
    generated: CandidateGenerationResult,
    extra_evaluations: (
        Mapping[str, Sequence[ConstraintEvaluation]] | None
    ) = None,
    *,
    candidate_config: CandidateGenerationConfig | None = None,
    scoring_config: HypothesisScoringConfig | None = None,
    ambiguity_config: AmbiguityConfig | None = None,
) -> AmbiguityAssessment:
    """Rank every retained candidate using its complete constraint set."""
    if generated.reference_id != reference.entity_id:
        raise ValueError('candidate result belongs to a different reference')
    extras = dict(extra_evaluations or {})
    complete = []
    for candidate in generated.retained:
        evaluations = candidate_constraints(
            candidate, reference, candidate_config
        ) + tuple(extras.get(candidate.candidate_id, ()))
        complete.append(CompleteHypothesis(
            (('target', candidate.candidate_id),), evaluations
        ))
    ranked = rank_complete_hypotheses(complete, config=scoring_config)
    underconstrained = (
        len(ranked) > 1
        and not reference.attributes
        and not any(extras.values())
    )
    return assess_ambiguity(
        reference.entity_id,
        ranked,
        ambiguity_config,
        underconstrained=underconstrained,
    )


def resolve_distance_ranking(
    reference_id: str,
    ranking: DistanceRanking,
    ambiguity_config: AmbiguityConfig | None = None,
) -> AmbiguityAssessment:
    """Adapt exhaustive target-anchor distances to ranked hypotheses."""
    complete = []
    for rank, item in enumerate(ranking.ranked):
        evaluation = ConstraintEvaluation(
            ranking.operator,
            item.score,
            False,
            True if rank == 0 else False,
            item.confidence,
            {
                'distance_m': item.distance_m,
                'rank': float(rank + 1),
                'distance_raw_margin': ranking.raw_margin,
                'distance_normalized_margin': ranking.normalized_margin,
            },
        )
        complete.append(CompleteHypothesis((
            ('target', item.target_id),
            ('anchor', item.anchor_id),
        ), (evaluation,)))
    ranked = rank_complete_hypotheses(complete)
    return assess_ambiguity(reference_id, ranked, ambiguity_config)


def set_resolution(
    generated: CandidateGenerationResult,
    *,
    definite_threshold: float = 0.65,
    probable_threshold: float = 0.15,
) -> SetResolution:
    """Partition persistent IDs for later numerical count stability policy."""
    if not 0.0 <= probable_threshold <= definite_threshold <= 1.0:
        raise ValueError(
            'set thresholds must satisfy 0 <= probable <= definite'
        )
    definite = []
    probable = []
    rejected = []
    unresolved = []
    for candidate in generated.candidates:
        if not candidate.retained:
            rejected.append(candidate.candidate_id)
        elif (
            candidate.geometry is None
            or candidate.geometry_confidence < 0.20
        ):
            unresolved.append(candidate.candidate_id)
        elif candidate.class_probability >= definite_threshold:
            definite.append(candidate.candidate_id)
        elif candidate.class_probability >= probable_threshold:
            probable.append(candidate.candidate_id)
        else:
            rejected.append(candidate.candidate_id)
    return SetResolution(
        tuple(definite), tuple(probable), tuple(rejected), tuple(unresolved)
    )


__all__ = [
    'candidate_constraints',
    'resolve_distance_ranking',
    'resolve_single_reference',
    'set_resolution',
]
