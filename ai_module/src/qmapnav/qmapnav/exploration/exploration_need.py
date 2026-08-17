"""Explicit representation of the missing evidence that justifies motion."""

from dataclasses import dataclass
from math import isfinite


NEED_TYPES = frozenset(
    {
        'ambiguous_target',
        'missing_anchor',
        'missing_target',
        'small_object_search',
        'support_surface_search',
        'unexplored_region',
        'weak_colour',
        'weak_geometry',
    }
)

#: Needs that describe an entity the robot has never observed at all.
ABSENT_ENTITY_NEEDS = frozenset(
    {
        'missing_anchor',
        'missing_target',
        'small_object_search',
        'support_surface_search',
    }
)


def _require_unit(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return float(value)


def _require_non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return float(value)


def _clean_ids(name: str, values) -> tuple[str, ...]:
    cleaned = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in cleaned):
        raise ValueError(f'{name} must contain only non-empty strings')
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f'{name} must not contain duplicates')
    return cleaned


@dataclass(frozen=True)
class ExplorationNeed:
    """
    One specific unresolved part of the current question.

    Exploration is only ever justified by an instance of this contract, so a
    viewpoint can always be traced back to the evidence it was meant to
    recover rather than to a generic "explore more" policy.
    """

    need_type: str
    target_reference_id: str | None = None
    candidate_instance_ids: tuple[str, ...] = ()
    missing_classes: tuple[str, ...] = ()
    missing_anchor_classes: tuple[str, ...] = ()
    unresolved_constraints: tuple[str, ...] = ()
    ambiguity_score: float = 0.0
    urgency: float = 0.0
    expected_task_value: float = 0.0
    reason: str = ''

    def __post_init__(self) -> None:
        if self.need_type not in NEED_TYPES:
            expected = ', '.join(sorted(NEED_TYPES))
            raise ValueError(f'need_type must be one of: {expected}')
        if self.target_reference_id is not None and (
            not isinstance(self.target_reference_id, str)
            or not self.target_reference_id.strip()
        ):
            raise ValueError('target_reference_id must be non-empty or None')
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('reason must be a non-empty string')
        for name in (
            'candidate_instance_ids',
            'missing_classes',
            'missing_anchor_classes',
            'unresolved_constraints',
        ):
            object.__setattr__(
                self, name, _clean_ids(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            'ambiguity_score',
            _require_unit('ambiguity_score', self.ambiguity_score),
        )
        object.__setattr__(
            self, 'urgency', _require_unit('urgency', self.urgency)
        )
        object.__setattr__(
            self,
            'expected_task_value',
            _require_non_negative(
                'expected_task_value', self.expected_task_value
            ),
        )

    @property
    def seeks_absent_entity(self) -> bool:
        """Return whether this need is for an entity never yet observed."""
        return self.need_type in ABSENT_ENTITY_NEEDS

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'need_type': self.need_type,
            'target_reference_id': self.target_reference_id,
            'candidate_instance_ids': list(self.candidate_instance_ids),
            'missing_classes': list(self.missing_classes),
            'missing_anchor_classes': list(self.missing_anchor_classes),
            'unresolved_constraints': list(self.unresolved_constraints),
            'ambiguity_score': self.ambiguity_score,
            'urgency': self.urgency,
            'expected_task_value': self.expected_task_value,
            'reason': self.reason,
        }


__all__ = [
    'ABSENT_ENTITY_NEEDS',
    'NEED_TYPES',
    'ExplorationNeed',
]
