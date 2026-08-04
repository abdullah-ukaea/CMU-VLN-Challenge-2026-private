"""Joint complete-hypothesis scoring without early nearest selection."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Callable, Mapping, Sequence

from qmapnav.reasoning.candidate_generation import EntityCandidate
from qmapnav.reasoning.resolution_contracts import CandidateHypothesis
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation


@dataclass(frozen=True)
class HypothesisScoringConfig:
    """Weights and penalties for complete constraint combinations."""

    hard_violation_penalty: float = 2.0
    soft_violation_penalty: float = 0.75
    unresolved_confidence_penalty: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            'hard_violation_penalty',
            'soft_violation_penalty',
            'unresolved_confidence_penalty',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')


@dataclass(frozen=True)
class CompleteHypothesis:
    """Named role assignment and its fully evaluated constraints."""

    role_ids: tuple[tuple[str, str], ...]
    evaluations: tuple[ConstraintEvaluation, ...]

    def __post_init__(self) -> None:
        role_ids = tuple(self.role_ids)
        role_names = [item[0] for item in role_ids]
        candidate_ids = [item[1] for item in role_ids]
        if len(role_names) != len(set(role_names)):
            raise ValueError('complete-hypothesis roles must be unique')
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('one candidate cannot fill multiple roles')
        if not all(
            isinstance(item, ConstraintEvaluation) for item in self.evaluations
        ):
            raise TypeError('evaluations must contain ConstraintEvaluation')
        object.__setattr__(self, 'role_ids', role_ids)
        object.__setattr__(self, 'evaluations', tuple(self.evaluations))


EvaluationFunction = Callable[
    [Mapping[str, EntityCandidate]], Sequence[ConstraintEvaluation]
]


def enumerate_complete_hypotheses(
    role_candidates: Mapping[str, Sequence[EntityCandidate]],
    evaluator: EvaluationFunction,
) -> tuple[CompleteHypothesis, ...]:
    """Enumerate a role product and evaluate every valid tuple."""
    if not role_candidates:
        return ()
    roles = tuple(sorted(role_candidates))
    pools = tuple(
        tuple(sorted(
            (item for item in role_candidates[role] if item.retained),
            key=lambda item: item.candidate_id,
        ))
        for role in roles
    )
    output = []
    for values in product(*pools):
        identifiers = tuple(item.candidate_id for item in values)
        if len(identifiers) != len(set(identifiers)):
            continue
        assignment = dict(zip(roles, values))
        evaluations = tuple(evaluator(assignment))
        output.append(CompleteHypothesis(
            tuple(zip(roles, identifiers)), evaluations
        ))
    return tuple(output)


def score_complete_hypothesis(
    hypothesis: CompleteHypothesis,
    weights: Mapping[str, float] | None = None,
    config: HypothesisScoringConfig | None = None,
) -> CandidateHypothesis:
    """Score all hard, soft, violated, and unresolved evidence together."""
    if not isinstance(hypothesis, CompleteHypothesis):
        raise TypeError('hypothesis must be CompleteHypothesis')
    policy = config or HypothesisScoringConfig()
    configured_weights = dict(weights or {})
    numerator = 0.0
    denominator = 0.0
    satisfied = []
    violated = []
    unresolved = []
    confidence_sum = 0.0
    evidence = {}
    hard_violation = False
    for evaluation in hypothesis.evaluations:
        weight = float(configured_weights.get(evaluation.constraint_name, 1.0))
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError('constraint weights must be finite and positive')
        denominator += weight
        confidence_sum += weight * evaluation.confidence
        evidence[evaluation.constraint_name] = evaluation.score
        evidence[f'{evaluation.constraint_name}_confidence'] = (
            evaluation.confidence
        )
        if evaluation.satisfied is True:
            satisfied.append(evaluation.constraint_name)
            numerator += weight * evaluation.score
        elif evaluation.satisfied is False:
            violated.append(evaluation.constraint_name)
            penalty = (
                policy.hard_violation_penalty
                if evaluation.is_hard else policy.soft_violation_penalty
            )
            numerator -= weight * penalty * max(1.0 - evaluation.score, 0.25)
            hard_violation = hard_violation or evaluation.is_hard
        else:
            unresolved.append(evaluation.constraint_name)
    denominator = max(denominator, 1.0)
    score = numerator / denominator
    if hard_violation:
        score = min(score, 0.0)
    evidence['hard_violation_count'] = float(sum(
        evaluation.is_hard and evaluation.satisfied is False
        for evaluation in hypothesis.evaluations
    ))
    evidence['unresolved_count'] = float(len(unresolved))
    coverage = 1.0 - len(unresolved) / max(len(hypothesis.evaluations), 1)
    confidence = confidence_sum / denominator
    confidence *= max(
        0.0,
        coverage - policy.unresolved_confidence_penalty * len(unresolved),
    )
    return CandidateHypothesis(
        tuple(item[1] for item in hypothesis.role_ids),
        score,
        min(1.0, max(0.0, confidence)),
        tuple(satisfied),
        tuple(violated),
        tuple(unresolved),
        evidence,
    )


def rank_complete_hypotheses(
    hypotheses: Sequence[CompleteHypothesis],
    weights: Mapping[str, float] | None = None,
    config: HypothesisScoringConfig | None = None,
) -> tuple[CandidateHypothesis, ...]:
    """Return every complete hypothesis with deterministic score ordering."""
    scored = [
        score_complete_hypothesis(item, weights, config)
        for item in hypotheses
    ]
    return tuple(sorted(
        scored,
        key=lambda item: (-item.score, -item.confidence, item.candidate_ids),
    ))


__all__ = [
    'CompleteHypothesis',
    'HypothesisScoringConfig',
    'enumerate_complete_hypotheses',
    'rank_complete_hypotheses',
    'score_complete_hypothesis',
]
