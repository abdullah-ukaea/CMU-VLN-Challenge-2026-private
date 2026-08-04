"""Conservative candidate generation from persistent episode maps."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from qmapnav.common import EntityReference, ObjectInstance
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_map import ObjectMap, PersistentObjectRecord
from qmapnav.mapping.structural_map import StructuralAnchor, StructuralMap
from qmapnav.reasoning.support_geometry import support_geometry
from qmapnav.reasoning.support_geometry import SupportGeometry


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Semantic and evidence floors applied before relation scoring."""

    minimum_class_probability: float = 0.15
    minimum_colour_probability: float = 0.10
    minimum_geometry_confidence: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            'minimum_class_probability',
            'minimum_colour_probability',
            'minimum_geometry_confidence',
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')


@dataclass(frozen=True)
class EntityCandidate:
    """A persistent map entity retained for complete-hypothesis scoring."""

    candidate_id: str
    semantic_class: str
    source_type: str
    class_probability: float
    colour_probability: float | None
    geometry_confidence: float
    geometry: SupportGeometry | None
    retained: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ('candidate_id', 'semantic_class', 'source_type'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')
        if self.source_type not in {'object', 'structural'}:
            raise ValueError('unsupported source_type')
        for name in ('class_probability', 'geometry_confidence'):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        if self.colour_probability is not None and not (
            isfinite(self.colour_probability)
            and 0.0 <= self.colour_probability <= 1.0
        ):
            raise ValueError('colour_probability must lie in [0, 1]')
        reasons = tuple(self.reasons)
        if not reasons or any(not item.strip() for item in reasons):
            raise ValueError('candidate reasons must be non-empty')
        object.__setattr__(self, 'reasons', reasons)
        evidence = dict(self.evidence)
        if any(not isfinite(value) for value in evidence.values()):
            raise ValueError('candidate evidence must be finite')
        object.__setattr__(
            self, 'evidence', MappingProxyType(dict(sorted(evidence.items())))
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace record for retained and rejected entities."""
        return {
            'candidate_id': self.candidate_id,
            'semantic_class': self.semantic_class,
            'source_type': self.source_type,
            'class_probability': self.class_probability,
            'colour_probability': self.colour_probability,
            'geometry_confidence': self.geometry_confidence,
            'geometry_available': self.geometry is not None,
            'retained': self.retained,
            'reasons': list(self.reasons),
            'evidence': dict(self.evidence),
        }


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Complete audit of a reference's persistent candidate pool."""

    reference_id: str
    candidates: tuple[EntityCandidate, ...]
    required_cardinality: int | None
    cardinality_satisfied: bool

    @property
    def retained(self) -> tuple[EntityCandidate, ...]:
        """Return retained candidates in deterministic identifier order."""
        return tuple(item for item in self.candidates if item.retained)

    @property
    def rejected(self) -> tuple[EntityCandidate, ...]:
        """Return rejected candidates with their audit reasons."""
        return tuple(item for item in self.candidates if not item.retained)

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-safe generation evidence."""
        return {
            'reference_id': self.reference_id,
            'required_cardinality': self.required_cardinality,
            'cardinality_satisfied': self.cardinality_satisfied,
            'candidates': [item.to_dict() for item in self.candidates],
        }


def records_from_object_map(
    object_map: ObjectMap,
) -> tuple[PersistentObjectRecord, ...]:
    """Snapshot persistent records without exposing raw detections."""
    if not isinstance(object_map, ObjectMap):
        raise TypeError('object_map must be ObjectMap')
    return tuple(
        object_map.record(instance.instance_id)
        for instance in object_map.active_instances()
    )


def anchors_from_structural_map(
    structural_map: StructuralMap,
) -> tuple[StructuralAnchor, ...]:
    """Snapshot walls and architectural anchors in stable ID order."""
    if not isinstance(structural_map, StructuralMap):
        raise TypeError('structural_map must be StructuralMap')
    anchors = structural_map.walls() + structural_map.anchors()
    return tuple(sorted(anchors, key=lambda item: item.anchor_id))


def generate_candidates_from_maps(
    reference: EntityReference,
    object_map: ObjectMap,
    structural_map: StructuralMap,
    config: CandidateGenerationConfig | None = None,
    *,
    hard_colour_threshold: bool = False,
) -> CandidateGenerationResult:
    """Generate candidates directly from the two episode-level maps."""
    return generate_candidates(
        reference,
        records_from_object_map(object_map),
        anchors_from_structural_map(structural_map),
        config,
        hard_colour_threshold=hard_colour_threshold,
    )


def generate_candidates(
    reference: EntityReference,
    objects: Sequence[ObjectInstance | PersistentObjectRecord],
    anchors: Sequence[StructuralAnchor] = (),
    config: CandidateGenerationConfig | None = None,
    *,
    hard_colour_threshold: bool = False,
) -> CandidateGenerationResult:
    """Audit all persistent entities and retain conservative class matches."""
    if not isinstance(reference, EntityReference):
        raise TypeError('reference must be EntityReference')
    policy = config or CandidateGenerationConfig()
    target_class = canonicalize_class_name(reference.class_name)
    requested_colour = reference.attributes.get('colour')
    generated = [
        _object_candidate(
            item, target_class, requested_colour, policy,
            hard_colour_threshold,
        )
        for item in objects
    ]
    generated.extend(
        _structural_candidate(item, target_class, requested_colour, policy)
        for item in anchors
    )
    generated.sort(key=lambda item: (item.source_type, item.candidate_id))
    retained_count = sum(item.retained for item in generated)
    required = reference.cardinality
    cardinality_satisfied = required is None or retained_count >= required
    return CandidateGenerationResult(
        reference.entity_id,
        tuple(generated),
        required,
        cardinality_satisfied,
    )


def _object_candidate(
    entity,
    target_class,
    requested_colour,
    config,
    hard_colour_threshold,
):
    if isinstance(entity, PersistentObjectRecord):
        instance = entity.instance
        confidence = entity.geometry_confidence
    elif isinstance(entity, ObjectInstance):
        instance = entity
        confidence = instance.confidence
    else:
        raise TypeError('objects must contain persistent object snapshots')
    identifier = str(instance.instance_id)
    class_probability = float(instance.class_scores.get(target_class, 0.0))
    colour_probability = (
        None if requested_colour is None
        else instance.colour_scores.get(requested_colour)
    )
    retained = class_probability >= config.minimum_class_probability
    reasons = [
        'class_probability_retained'
        if retained else 'class_probability_below_minimum'
    ]
    if requested_colour is not None:
        if colour_probability is None:
            reasons.append('colour_unavailable_soft_penalty')
        elif colour_probability < config.minimum_colour_probability:
            reasons.append('colour_probability_below_minimum')
            if hard_colour_threshold:
                retained = False
                reasons.append('hard_colour_threshold_rejected')
        else:
            reasons.append('colour_probability_retained')
    if confidence < config.minimum_geometry_confidence:
        reasons.append('weak_geometry_retained_for_later_constraints')
    geometry = _geometry_or_none(entity)
    if geometry is None:
        reasons.append('geometry_unavailable')
    return EntityCandidate(
        identifier,
        target_class,
        'object',
        class_probability,
        colour_probability,
        confidence,
        geometry,
        retained,
        tuple(reasons),
        {
            'class_probability': class_probability,
            'colour_probability': colour_probability or 0.0,
            'geometry_confidence': confidence,
        },
    )


def _structural_candidate(entity, target_class, requested_colour, config):
    if not isinstance(entity, StructuralAnchor):
        raise TypeError('anchors must contain StructuralAnchor values')
    semantic = canonicalize_class_name(entity.semantic_class)
    class_probability = entity.confidence if semantic == target_class else 0.0
    retained = class_probability >= config.minimum_class_probability
    reasons = [
        'structural_class_retained'
        if retained else 'structural_class_incompatible'
    ]
    if requested_colour is not None:
        reasons.append('structural_colour_unavailable_soft_penalty')
    if entity.confidence < config.minimum_geometry_confidence:
        reasons.append('weak_geometry_retained_for_later_constraints')
    geometry = _geometry_or_none(entity)
    if geometry is None:
        reasons.append('geometry_unavailable')
    return EntityCandidate(
        entity.anchor_id,
        semantic,
        'structural',
        class_probability,
        None,
        entity.confidence,
        geometry,
        retained,
        tuple(reasons),
        {
            'class_probability': class_probability,
            'colour_probability': 0.0,
            'geometry_confidence': entity.confidence,
        },
    )


def _geometry_or_none(entity):
    try:
        return support_geometry(entity)
    except ValueError:
        return None


__all__ = [
    'CandidateGenerationConfig',
    'CandidateGenerationResult',
    'EntityCandidate',
    'anchors_from_structural_map',
    'generate_candidates',
    'generate_candidates_from_maps',
    'records_from_object_map',
]
