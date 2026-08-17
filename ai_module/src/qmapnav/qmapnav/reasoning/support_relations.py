"""Candidate generation and conservative physical-support inference."""

from dataclasses import dataclass

from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.vertical_relations import RelationEvidence


STRONG_SUPPORT_CLASSES = frozenset({
    'cabinet', 'counter', 'desk', 'shelf', 'table',
})
POSSIBLE_SUPPORT_CLASSES = frozenset({
    'bed', 'bench', 'chair', 'sofa', 'stool', 'window_ledge',
})


@dataclass(frozen=True)
class SupportRelationConfig:
    """Measured support-contact, search, and acceptance thresholds."""

    maximum_support_gap_m: float = 0.15
    penetration_tolerance_m: float = 0.08
    minimum_subject_support_overlap: float = 0.50
    support_search_radius_m: float = 2.0
    minimum_geometry_confidence: float = 0.25
    accept_on_confidence: float = 0.70
    uncertain_on_confidence: float = 0.40
    include_floor_supports: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.maximum_support_gap_m,
            self.penetration_tolerance_m,
            self.support_search_radius_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError('support distances must be positive')
        bounded = (
            self.minimum_subject_support_overlap,
            self.minimum_geometry_confidence,
            self.accept_on_confidence,
            self.uncertain_on_confidence,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError('support confidence thresholds must lie in [0, 1]')
        if self.uncertain_on_confidence > self.accept_on_confidence:
            raise ValueError('uncertain threshold must not exceed acceptance')
        if not isinstance(self.include_floor_supports, bool):
            raise TypeError('include_floor_supports must be boolean')


def generate_support_candidates(
    subject: SupportGeometry,
    entities: list[SupportGeometry] | tuple[SupportGeometry, ...],
    config: SupportRelationConfig | None = None,
) -> list[SupportGeometry]:
    """Return nearby supports below the subject, ordered deterministically."""
    policy = config or SupportRelationConfig()
    candidates = []
    for entity in entities:
        if entity.entity_id == subject.entity_id:
            continue
        if (
            entity.semantic_class == 'floor'
            and not policy.include_floor_supports
        ):
            continue
        metrics = footprint_metrics(subject, entity)
        gap = subject.bottom_z - entity.top_z
        if metrics.edge_distance_m > policy.support_search_radius_m:
            continue
        if gap < -policy.penetration_tolerance_m:
            continue
        if entity.top_z > subject.centre_xyz[2]:
            continue
        candidates.append(entity)
    return sorted(candidates, key=lambda item: (
        footprint_metrics(subject, item).edge_distance_m,
        abs(subject.bottom_z - item.top_z),
        item.entity_id,
    ))


def on_evidence(
    subject: SupportGeometry,
    support: SupportGeometry,
    config: SupportRelationConfig | None = None,
) -> RelationEvidence:
    """Score physical support using contact and subject-footprint coverage."""
    if subject.entity_id == support.entity_id:
        raise ValueError('self-relations are not permitted')
    policy = config or SupportRelationConfig()
    metrics = footprint_metrics(subject, support)
    gap = subject.bottom_z - support.top_z
    if -policy.penetration_tolerance_m <= gap <= 0.0:
        vertical = 1.0 + gap / policy.penetration_tolerance_m
    elif 0.0 < gap <= policy.maximum_support_gap_m:
        vertical = 1.0 - gap / policy.maximum_support_gap_m
    else:
        vertical = 0.0
    footprint = min(
        1.0,
        metrics.subject_overlap / policy.minimum_subject_support_overlap,
    )
    if support.semantic_class in STRONG_SUPPORT_CLASSES:
        semantic = 1.0
    elif support.semantic_class in POSSIBLE_SUPPORT_CLASSES:
        semantic = 0.85
    elif support.source_type == 'structural':
        semantic = 0.75
    else:
        semantic = 0.65
    geometry = min(subject.confidence, support.confidence)
    if geometry < policy.minimum_geometry_confidence:
        geometry *= 0.5
    if subject.quality in {'sparse', 'partially_observed', 'uncertain'}:
        geometry *= 0.60
    if support.quality in {'sparse', 'partially_observed', 'uncertain'}:
        geometry *= 0.75
    confidence = min(1.0, vertical * footprint * semantic * geometry)
    if confidence >= policy.accept_on_confidence:
        status = 'accepted'
        accepted = True
    elif confidence >= policy.uncertain_on_confidence:
        status = 'uncertain'
        accepted = False
    elif vertical <= 0.0:
        status = 'no_contact'
        accepted = False
    elif metrics.subject_overlap < policy.minimum_subject_support_overlap:
        status = 'insufficient_overlap'
        accepted = False
    else:
        status = 'weak_geometry'
        accepted = False
    return RelationEvidence(
        'on', subject.entity_id, support.entity_id, confidence, accepted,
        status, gap, metrics.subject_overlap, metrics.centre_distance_m,
        geometry,
    )


def ranked_support_hypotheses(
    subject: SupportGeometry,
    entities: list[SupportGeometry] | tuple[SupportGeometry, ...],
    config: SupportRelationConfig | None = None,
) -> tuple[RelationEvidence, ...]:
    """Retain all plausible support hypotheses in confidence order."""
    policy = config or SupportRelationConfig()
    evidence = [
        on_evidence(subject, item, policy)
        for item in generate_support_candidates(subject, entities, policy)
    ]
    retained = [
        item for item in evidence
        if item.confidence >= policy.uncertain_on_confidence
    ]
    return tuple(sorted(
        retained, key=lambda item: (-item.confidence, item.anchor_id)
    ))


__all__ = [
    'POSSIBLE_SUPPORT_CLASSES', 'STRONG_SUPPORT_CLASSES',
    'SupportRelationConfig', 'generate_support_candidates', 'on_evidence',
    'ranked_support_hypotheses',
]
