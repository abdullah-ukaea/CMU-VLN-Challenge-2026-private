"""Earliest-cause failure classification and Day 10 fix ranking."""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from qmapnav.evaluation.object_reference_contracts import StageEvidence


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
    candidates: Iterable[FixCandidate],
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
    'FailureClassification',
    'FixCandidate',
    'classify_primary_failure',
    'rank_fix_candidates',
]
