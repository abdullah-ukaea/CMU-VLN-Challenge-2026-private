"""Compare qualifying persistent-ID sets across anchor hypotheses."""

from dataclasses import dataclass
from math import isfinite


def _ids(values: tuple[int, ...]) -> tuple[int, ...]:
    copied = tuple(sorted(values))
    if any(isinstance(value, bool) or value < 0 for value in copied):
        raise ValueError('instance IDs must be non-negative integers')
    if len(copied) != len(set(copied)):
        raise ValueError('instance IDs must be unique')
    return copied


@dataclass(frozen=True)
class AnchorCountHypothesis:
    """One complete non-target role assignment and its qualifying IDs."""

    role_ids: tuple[tuple[str, str], ...]
    qualifying_instance_ids: tuple[int, ...]
    score: float
    confidence: float

    def __post_init__(self) -> None:
        roles = tuple(sorted(self.role_ids))
        if len({name for name, _ in roles}) != len(roles):
            raise ValueError('anchor role names must be unique')
        if any(not name.strip() or not value.strip() for name, value in roles):
            raise ValueError('anchor roles and IDs must be non-empty')
        if not isfinite(self.score):
            raise ValueError('anchor hypothesis score must be finite')
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('anchor confidence must lie in [0, 1]')
        object.__setattr__(self, 'role_ids', roles)
        object.__setattr__(
            self,
            'qualifying_instance_ids',
            _ids(self.qualifying_instance_ids),
        )

    @property
    def count(self) -> int:
        """Return the number of unique qualifying persistent IDs."""
        return len(self.qualifying_instance_ids)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe hypothesis evidence."""
        return {
            'role_ids': dict(self.role_ids),
            'qualifying_instance_ids': list(self.qualifying_instance_ids),
            'count': self.count,
            'score': self.score,
            'confidence': self.confidence,
        }


@dataclass(frozen=True)
class AnchorAmbiguityAssessment:
    """Count and ID-set agreement across plausible anchor assignments."""

    hypotheses: tuple[AnchorCountHypothesis, ...]
    count_consistent: bool
    id_set_consistent: bool
    minimum_count: int
    maximum_count: int

    def __post_init__(self) -> None:
        hypotheses = tuple(self.hypotheses)
        if not all(isinstance(item, AnchorCountHypothesis) for item in hypotheses):
            raise TypeError('hypotheses must contain AnchorCountHypothesis values')
        object.__setattr__(self, 'hypotheses', hypotheses)
        for name in ('count_consistent', 'id_set_consistent'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be boolean')
        if self.minimum_count < 0 or self.maximum_count < self.minimum_count:
            raise ValueError('anchor count range is invalid')

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe ambiguity evidence."""
        return {
            'hypotheses': [item.to_dict() for item in self.hypotheses],
            'count_consistent': self.count_consistent,
            'id_set_consistent': self.id_set_consistent,
            'minimum_count': self.minimum_count,
            'maximum_count': self.maximum_count,
        }


def assess_anchor_counts(
    hypotheses: tuple[AnchorCountHypothesis, ...],
) -> AnchorAmbiguityAssessment:
    """Compare all retained anchor hypotheses without selecting the first."""
    ordered = tuple(sorted(
        hypotheses,
        key=lambda item: (-item.score, -item.confidence, item.role_ids),
    ))
    if not ordered:
        return AnchorAmbiguityAssessment((), True, True, 0, 0)
    counts = {item.count for item in ordered}
    identifier_sets = {
        item.qualifying_instance_ids for item in ordered
    }
    return AnchorAmbiguityAssessment(
        ordered,
        len(counts) == 1,
        len(identifier_sets) == 1,
        min(counts),
        max(counts),
    )


__all__ = [
    'AnchorAmbiguityAssessment',
    'AnchorCountHypothesis',
    'assess_anchor_counts',
]
