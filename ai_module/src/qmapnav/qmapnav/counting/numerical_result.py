"""Validated contracts for persistent-map numerical answers."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from qmapnav.counting.anchor_ambiguity import AnchorAmbiguityAssessment


COUNT_CLASSIFICATIONS = frozenset({
    'definite', 'probable', 'rejected', 'unresolved',
})


def _unit(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must lie in [0, 1]')
    return float(value)


def _ids(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    copied = tuple(sorted(values))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in copied
    ):
        raise ValueError(f'{name} must contain non-negative integers')
    if len(copied) != len(set(copied)):
        raise ValueError(f'{name} must contain unique values')
    return copied


@dataclass(frozen=True)
class CountDiagnostic:
    """Auditable class, colour, and relation decision for one target ID."""

    instance_id: int
    classification: str
    class_probability: float
    colour_probability: float | None
    relation_score: float
    overall_score: float
    confidence: float
    reasons: tuple[str, ...]
    role_ids: Mapping[str, str]
    evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.instance_id, bool)
            or not isinstance(self.instance_id, int)
            or self.instance_id < 0
        ):
            raise ValueError('instance_id must be a non-negative integer')
        if self.classification not in COUNT_CLASSIFICATIONS:
            raise ValueError('unsupported count classification')
        object.__setattr__(
            self,
            'class_probability',
            _unit('class_probability', self.class_probability),
        )
        if self.colour_probability is not None:
            object.__setattr__(
                self,
                'colour_probability',
                _unit('colour_probability', self.colour_probability),
            )
        for name in ('relation_score', 'overall_score'):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f'{name} must be finite')
        object.__setattr__(self, 'confidence', _unit('confidence', self.confidence))
        reasons = tuple(self.reasons)
        if not reasons or any(not value.strip() for value in reasons):
            raise ValueError('diagnostic reasons must be non-empty')
        object.__setattr__(self, 'reasons', reasons)
        roles = dict(self.role_ids)
        if any(not key.strip() or not value.strip() for key, value in roles.items()):
            raise ValueError('role IDs must be non-empty')
        object.__setattr__(
            self, 'role_ids', MappingProxyType(dict(sorted(roles.items())))
        )
        evidence = dict(self.evidence)
        if any(not isfinite(value) for value in evidence.values()):
            raise ValueError('diagnostic evidence must be finite')
        object.__setattr__(
            self, 'evidence', MappingProxyType(dict(sorted(evidence.items())))
        )

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe candidate evidence."""
        return {
            'instance_id': self.instance_id,
            'classification': self.classification,
            'class_probability': self.class_probability,
            'colour_probability': self.colour_probability,
            'relation_score': self.relation_score,
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'reasons': list(self.reasons),
            'role_ids': dict(self.role_ids),
            'evidence': dict(self.evidence),
        }


@dataclass(frozen=True)
class NumericalResult:
    """One persistent-ID count with explicit uncertainty partitions."""

    target_class: str
    definite_instance_ids: tuple[int, ...]
    probable_instance_ids: tuple[int, ...]
    rejected_instance_ids: tuple[int, ...]
    unresolved_instance_ids: tuple[int, ...]
    count: int
    count_confidence: float
    stable: bool
    stability_reason: str
    diagnostics: tuple[CountDiagnostic, ...]
    anchor_ambiguity: AnchorAmbiguityAssessment
    hypothesis_limit_reached: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.target_class, str) or not self.target_class.strip():
            raise ValueError('target_class must be a non-empty string')
        names = (
            'definite_instance_ids',
            'probable_instance_ids',
            'rejected_instance_ids',
            'unresolved_instance_ids',
        )
        partitions = []
        for name in names:
            values = _ids(name, getattr(self, name))
            object.__setattr__(self, name, values)
            partitions.append(set(values))
        if any(
            partitions[left] & partitions[right]
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        ):
            raise ValueError('numerical result partitions must be disjoint')
        expected = len(partitions[0] | partitions[1])
        if self.count != expected:
            raise ValueError('count must equal definite plus probable unique IDs')
        object.__setattr__(
            self,
            'count_confidence',
            _unit('count_confidence', self.count_confidence),
        )
        if not isinstance(self.stable, bool):
            raise TypeError('stable must be boolean')
        if not self.stability_reason.strip():
            raise ValueError('stability_reason must be non-empty')
        diagnostics = tuple(self.diagnostics)
        if not all(isinstance(item, CountDiagnostic) for item in diagnostics):
            raise TypeError('diagnostics must contain CountDiagnostic values')
        if len({item.instance_id for item in diagnostics}) != len(diagnostics):
            raise ValueError('diagnostics must contain unique instance IDs')
        object.__setattr__(self, 'diagnostics', diagnostics)
        if not isinstance(self.anchor_ambiguity, AnchorAmbiguityAssessment):
            raise TypeError('anchor_ambiguity has the wrong type')
        if not isinstance(self.hypothesis_limit_reached, bool):
            raise TypeError('hypothesis_limit_reached must be boolean')

    @property
    def qualifying_instance_ids(self) -> tuple[int, ...]:
        """Return definite and probable IDs in stable order."""
        return tuple(sorted(
            self.definite_instance_ids + self.probable_instance_ids
        ))

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe numerical evidence."""
        return {
            'target_class': self.target_class,
            'definite_instance_ids': list(self.definite_instance_ids),
            'probable_instance_ids': list(self.probable_instance_ids),
            'rejected_instance_ids': list(self.rejected_instance_ids),
            'unresolved_instance_ids': list(self.unresolved_instance_ids),
            'qualifying_instance_ids': list(self.qualifying_instance_ids),
            'count': self.count,
            'count_confidence': self.count_confidence,
            'stable': self.stable,
            'stability_reason': self.stability_reason,
            'anchor_ambiguity': self.anchor_ambiguity.to_dict(),
            'hypothesis_limit_reached': self.hypothesis_limit_reached,
            'diagnostics': [item.to_dict() for item in self.diagnostics],
        }


__all__ = ['COUNT_CLASSIFICATIONS', 'CountDiagnostic', 'NumericalResult']
