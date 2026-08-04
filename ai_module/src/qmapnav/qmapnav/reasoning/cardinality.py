"""Deterministic cardinality and candidate-role enumeration."""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

from qmapnav.reasoning.candidate_generation import EntityCandidate


@dataclass(frozen=True)
class CandidateSet:
    """One exact-cardinality persistent candidate set."""

    candidate_ids: tuple[str, ...]
    cardinality: int
    ordered_roles: bool = False

    def __post_init__(self) -> None:
        values = tuple(self.candidate_ids)
        if self.cardinality <= 0:
            raise ValueError('cardinality must be positive')
        if len(values) != self.cardinality:
            raise ValueError('candidate IDs must satisfy exact cardinality')
        if len(values) != len(set(values)):
            raise ValueError('one instance cannot fill multiple set roles')
        if not self.ordered_roles:
            values = tuple(sorted(values))
        object.__setattr__(self, 'candidate_ids', values)


def enumerate_candidate_sets(
    candidates: Sequence[EntityCandidate],
    cardinality: int | None,
) -> tuple[CandidateSet, ...]:
    """Enumerate exact sets, or singleton hypotheses when count is unknown."""
    retained = sorted(
        (item for item in candidates if item.retained),
        key=lambda item: item.candidate_id,
    )
    count = 1 if cardinality is None else cardinality
    if isinstance(count, bool) or count <= 0:
        raise ValueError('cardinality must be positive when specified')
    return tuple(
        CandidateSet(tuple(item.candidate_id for item in group), count)
        for group in combinations(retained, count)
    )


def enumerate_unordered_pairs(
    candidates: Sequence[EntityCandidate],
) -> tuple[CandidateSet, ...]:
    """Enumerate every same-pool unordered pair exactly once."""
    return enumerate_candidate_sets(candidates, 2)


def enumerate_mixed_roles(
    first_role: Sequence[EntityCandidate],
    second_role: Sequence[EntityCandidate],
) -> tuple[CandidateSet, ...]:
    """Enumerate semantically ordered mixed-role combinations."""
    first = sorted(
        (item for item in first_role if item.retained),
        key=lambda item: item.candidate_id,
    )
    second = sorted(
        (item for item in second_role if item.retained),
        key=lambda item: item.candidate_id,
    )
    return tuple(
        CandidateSet((left.candidate_id, right.candidate_id), 2, True)
        for left, right in product(first, second)
        if left.candidate_id != right.candidate_id
    )


def plural_set(candidates: Sequence[EntityCandidate]) -> CandidateSet | None:
    """Return all retained persistent IDs for an unspecified plural mention."""
    identifiers = tuple(sorted(
        item.candidate_id for item in candidates if item.retained
    ))
    if not identifiers:
        return None
    return CandidateSet(identifiers, len(identifiers))


__all__ = [
    'CandidateSet',
    'enumerate_candidate_sets',
    'enumerate_mixed_roles',
    'enumerate_unordered_pairs',
    'plural_set',
]
