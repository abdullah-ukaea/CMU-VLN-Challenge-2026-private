"""Explicit confidence-margin policy for ranked reference hypotheses."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from qmapnav.reasoning.resolution_contracts import CandidateHypothesis
from qmapnav.reasoning.resolution_contracts import ReferenceResolution


@dataclass(frozen=True)
class AmbiguityConfig:
    """Absolute-score and top-two separation requirements."""

    resolved_minimum_score: float = 0.65
    resolved_minimum_margin: float = 0.12
    ambiguous_margin: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            'resolved_minimum_score',
            'resolved_minimum_margin',
            'ambiguous_margin',
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        if self.ambiguous_margin > self.resolved_minimum_margin:
            raise ValueError(
                'ambiguous margin must not exceed resolved margin'
            )


@dataclass(frozen=True)
class AmbiguityAssessment:
    """Reference resolution plus raw and normalized margins."""

    resolution: ReferenceResolution
    raw_margin: float
    normalized_margin: float

    def to_dict(self) -> dict[str, object]:
        """Return trace-ready ambiguity evidence."""
        output = self.resolution.to_dict()
        output['raw_margin'] = self.raw_margin
        output['normalized_margin'] = self.normalized_margin
        return output


def assess_ambiguity(
    reference_id: str,
    hypotheses: Sequence[CandidateHypothesis],
    config: AmbiguityConfig | None = None,
    *,
    underconstrained: bool = False,
) -> AmbiguityAssessment:
    """Require both an adequate winner and adequate score separation."""
    policy = config or AmbiguityConfig()
    ranked = tuple(sorted(
        hypotheses,
        key=lambda item: (-item.score, -item.confidence, item.candidate_ids),
    ))
    unresolved = tuple(sorted({
        name for item in ranked for name in item.unresolved_constraints
    }))
    if not ranked:
        return _assessment(
            reference_id, ranked, None, 0.0, 0.0, 'no_candidates', unresolved
        )
    top = ranked[0]
    if len(ranked) == 1:
        raw_margin = max(0.0, top.score)
        normalized = 1.0
    else:
        raw_margin = top.score - ranked[1].score
        normalized = raw_margin / (abs(top.score) + 1.0e-9)
    all_conflicting = all(
        item.evidence.get('hard_violation_count', 0.0) > 0.0
        for item in ranked
    )
    if all_conflicting:
        status = 'conflicting_constraints'
    elif underconstrained:
        status = 'underconstrained'
    elif top.score < policy.resolved_minimum_score or (
        top.confidence < policy.resolved_minimum_score
    ):
        status = 'low_confidence'
    elif len(ranked) > 1 and (
        raw_margin <= policy.ambiguous_margin
        or normalized < policy.resolved_minimum_margin
    ):
        status = 'ambiguous'
    elif len(ranked) > 1 and raw_margin < policy.resolved_minimum_margin:
        status = 'ambiguous'
    else:
        status = 'resolved'
    selected = top.candidate_ids if status == 'resolved' else None
    return _assessment(
        reference_id,
        ranked,
        selected,
        raw_margin,
        normalized,
        status,
        unresolved,
    )


def _assessment(
    reference_id,
    ranked,
    selected,
    raw_margin,
    normalized,
    status,
    unresolved,
):
    resolution = ReferenceResolution(
        reference_id,
        ranked,
        selected,
        raw_margin,
        status,
        unresolved,
    )
    return AmbiguityAssessment(resolution, raw_margin, normalized)


__all__ = ['AmbiguityAssessment', 'AmbiguityConfig', 'assess_ambiguity']
