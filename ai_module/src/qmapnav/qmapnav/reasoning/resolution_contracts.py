"""Validated, ROS-independent contracts for spatial reference resolution."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping


RESOLUTION_STATUSES = frozenset({
    'resolved',
    'ambiguous',
    'underconstrained',
    'no_candidates',
    'conflicting_constraints',
    'low_confidence',
})


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _unit(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be finite and lie in [0, 1]')
    return float(value)


def _finite(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _ids(name: str, values: tuple[str, ...], *, empty: bool = False):
    copied = tuple(values)
    if not empty and not copied:
        raise ValueError(f'{name} must not be empty')
    if any(
        not isinstance(value, str) or not value.strip() for value in copied
    ):
        raise ValueError(f'{name} must contain non-empty strings')
    if len(copied) != len(set(copied)):
        raise ValueError(f'{name} must contain unique values')
    return copied


def _names(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    return _ids(name, values, empty=True)


def _evidence(values: Mapping[str, float]) -> Mapping[str, float]:
    copied = dict(values)
    for name, value in copied.items():
        _text('evidence name', name)
        copied[name] = _finite(f'evidence[{name!r}]', value)
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True)
class ConstraintEvaluation:
    """One hard or soft constraint with an explicit unknown state."""

    constraint_name: str
    score: float
    is_hard: bool
    satisfied: bool | None
    confidence: float
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'constraint_name', _text(
            'constraint_name', self.constraint_name
        ))
        object.__setattr__(self, 'score', _finite('score', self.score))
        if not isinstance(self.is_hard, bool):
            raise TypeError('is_hard must be bool')
        if self.satisfied is not None and not isinstance(self.satisfied, bool):
            raise TypeError('satisfied must be bool or None')
        object.__setattr__(
            self, 'confidence', _unit('confidence', self.confidence)
        )
        object.__setattr__(self, 'evidence', _evidence(self.evidence))

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe evidence."""
        return {
            'constraint_name': self.constraint_name,
            'score': self.score,
            'is_hard': self.is_hard,
            'satisfied': self.satisfied,
            'confidence': self.confidence,
            'evidence': dict(self.evidence),
        }


@dataclass(frozen=True)
class CandidateHypothesis:
    """One ranked single-object, pair, set, or complete-role hypothesis."""

    candidate_ids: tuple[str, ...]
    score: float
    confidence: float
    satisfied_constraints: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    unresolved_constraints: tuple[str, ...] = ()
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'candidate_ids', _ids('candidate_ids', self.candidate_ids)
        )
        object.__setattr__(self, 'score', _finite('score', self.score))
        object.__setattr__(
            self, 'confidence', _unit('confidence', self.confidence)
        )
        for name in (
            'satisfied_constraints',
            'violated_constraints',
            'unresolved_constraints',
        ):
            object.__setattr__(self, name, _names(name, getattr(self, name)))
        object.__setattr__(self, 'evidence', _evidence(self.evidence))

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe hypothesis evidence."""
        return {
            'candidate_ids': list(self.candidate_ids),
            'score': self.score,
            'confidence': self.confidence,
            'satisfied_constraints': list(self.satisfied_constraints),
            'violated_constraints': list(self.violated_constraints),
            'unresolved_constraints': list(self.unresolved_constraints),
            'evidence': dict(self.evidence),
        }


@dataclass(frozen=True)
class ReferenceResolution:
    """A ranked result that may intentionally leave selection unresolved."""

    reference_id: str
    ranked_hypotheses: tuple[CandidateHypothesis, ...]
    selected_candidate_ids: tuple[str, ...] | None
    confidence_margin: float
    resolution_status: str
    unresolved_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'reference_id', _text('reference_id', self.reference_id)
        )
        hypotheses = tuple(self.ranked_hypotheses)
        if not all(
            isinstance(item, CandidateHypothesis) for item in hypotheses
        ):
            raise TypeError('ranked_hypotheses must contain hypotheses')
        object.__setattr__(self, 'ranked_hypotheses', hypotheses)
        selected = self.selected_candidate_ids
        if selected is not None:
            selected = _ids('selected_candidate_ids', selected)
        object.__setattr__(self, 'selected_candidate_ids', selected)
        object.__setattr__(
            self,
            'confidence_margin',
            _finite('confidence_margin', self.confidence_margin),
        )
        if self.resolution_status not in RESOLUTION_STATUSES:
            raise ValueError('unsupported resolution_status')
        object.__setattr__(
            self,
            'unresolved_constraints',
            _names('unresolved_constraints', self.unresolved_constraints),
        )
        if self.resolution_status != 'resolved' and selected is not None:
            raise ValueError('only a resolved result may force a selection')
        if selected is not None and not hypotheses:
            raise ValueError('a selection requires a ranked hypothesis')

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe reference-resolution evidence."""
        return {
            'reference_id': self.reference_id,
            'ranked_hypotheses': [item.to_dict() for item in
                                  self.ranked_hypotheses],
            'selected_candidate_ids': (
                None if self.selected_candidate_ids is None
                else list(self.selected_candidate_ids)
            ),
            'confidence_margin': self.confidence_margin,
            'resolution_status': self.resolution_status,
            'unresolved_constraints': list(self.unresolved_constraints),
        }


@dataclass(frozen=True)
class PairHypothesis:
    """Canonical unordered same-class anchor-pair hypothesis."""

    first_id: str
    second_id: str
    class_name: str
    centre_distance_m: float
    footprint_gap_m: float
    traversable: bool
    corridor_width_m: float | None
    score: float
    confidence: float
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        first, second = sorted((
            _text('first_id', self.first_id),
            _text('second_id', self.second_id),
        ))
        if first == second:
            raise ValueError('pair members must be distinct')
        object.__setattr__(self, 'first_id', first)
        object.__setattr__(self, 'second_id', second)
        object.__setattr__(self, 'class_name', _text(
            'class_name', self.class_name
        ))
        for name in ('centre_distance_m', 'footprint_gap_m'):
            value = _finite(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative')
            object.__setattr__(self, name, value)
        if not isinstance(self.traversable, bool):
            raise TypeError('traversable must be bool')
        if self.corridor_width_m is not None:
            width = _finite('corridor_width_m', self.corridor_width_m)
            if width < 0.0:
                raise ValueError('corridor_width_m must be non-negative')
            object.__setattr__(self, 'corridor_width_m', width)
        object.__setattr__(self, 'score', _finite('score', self.score))
        object.__setattr__(
            self, 'confidence', _unit('confidence', self.confidence)
        )
        object.__setattr__(self, 'evidence', _evidence(self.evidence))

    @property
    def candidate_ids(self) -> tuple[str, str]:
        """Return the canonical pair identity."""
        return self.first_id, self.second_id

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe pair evidence."""
        return {
            'first_id': self.first_id,
            'second_id': self.second_id,
            'class_name': self.class_name,
            'centre_distance_m': self.centre_distance_m,
            'footprint_gap_m': self.footprint_gap_m,
            'traversable': self.traversable,
            'corridor_width_m': self.corridor_width_m,
            'score': self.score,
            'confidence': self.confidence,
            'evidence': dict(self.evidence),
        }


@dataclass(frozen=True)
class SetResolution:
    """Semantic persistent-ID partition for a later numerical answer."""

    definite_ids: tuple[str, ...] = ()
    probable_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    unresolved_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = {}
        for name in (
            'definite_ids', 'probable_ids', 'rejected_ids', 'unresolved_ids'
        ):
            groups[name] = _ids(name, getattr(self, name), empty=True)
            object.__setattr__(self, name, groups[name])
        flattened = [item for values in groups.values() for item in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError('set-resolution partitions must be disjoint')

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe set partitions."""
        return {
            'definite_ids': list(self.definite_ids),
            'probable_ids': list(self.probable_ids),
            'rejected_ids': list(self.rejected_ids),
            'unresolved_ids': list(self.unresolved_ids),
        }


__all__ = [
    'RESOLUTION_STATUSES',
    'CandidateHypothesis',
    'ConstraintEvaluation',
    'PairHypothesis',
    'ReferenceResolution',
    'SetResolution',
]
