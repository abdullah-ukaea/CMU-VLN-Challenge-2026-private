"""Map-frame `above` and inverse `below` evidence."""

from dataclasses import dataclass
from math import exp, isfinite

from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_geometry import SupportGeometry


@dataclass(frozen=True)
class VerticalRelationConfig:
    """Vertical tolerance and horizontal-relevance policy."""

    vertical_tolerance_m: float = 0.08
    maximum_horizontal_gap_m: float = 0.50
    minimum_footprint_overlap: float = 0.02

    def __post_init__(self) -> None:
        if self.vertical_tolerance_m < 0.0:
            raise ValueError('vertical tolerance must be non-negative')
        if self.maximum_horizontal_gap_m <= 0.0:
            raise ValueError('maximum horizontal gap must be positive')
        if not 0.0 <= self.minimum_footprint_overlap <= 1.0:
            raise ValueError('minimum footprint overlap must lie in [0, 1]')


@dataclass(frozen=True)
class RelationEvidence:
    """Auditable geometric evidence for one ordered relation."""

    relation: str
    subject_id: str
    anchor_id: str
    confidence: float
    accepted: bool
    status: str
    vertical_gap_m: float
    subject_support_overlap: float
    horizontal_distance_m: float
    geometry_confidence: float
    source: str = 'geometry'

    def __post_init__(self) -> None:
        values = (
            self.confidence, self.vertical_gap_m,
            self.subject_support_overlap, self.horizontal_distance_m,
            self.geometry_confidence,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError('relation evidence must be finite')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('relation confidence must lie in [0, 1]')


def above_evidence(
    subject: SupportGeometry,
    anchor: SupportGeometry,
    config: VerticalRelationConfig | None = None,
) -> RelationEvidence:
    """Evaluate whether subject is above a horizontally relevant anchor."""
    if subject.entity_id == anchor.entity_id:
        raise ValueError('self-relations are not permitted')
    policy = config or VerticalRelationConfig()
    metrics = footprint_metrics(subject, anchor)
    gap = subject.bottom_z - anchor.top_z
    ordered = subject.centre_xyz[2] > anchor.centre_xyz[2]
    relevant = (
        metrics.subject_overlap >= policy.minimum_footprint_overlap
        or metrics.edge_distance_m <= policy.maximum_horizontal_gap_m
    )
    accepted = ordered and gap >= -policy.vertical_tolerance_m and relevant
    vertical_score = (
        0.0 if not ordered or gap < -policy.vertical_tolerance_m
        else 1.0 - exp(-(gap + policy.vertical_tolerance_m) / 0.15)
    )
    horizontal_score = max(
        metrics.subject_overlap,
        max(0.0, 1.0 - metrics.edge_distance_m /
            policy.maximum_horizontal_gap_m) * 0.65,
    )
    geometry = min(subject.confidence, anchor.confidence)
    confidence = min(1.0, vertical_score * horizontal_score * geometry)
    status = 'accepted' if accepted else (
        'horizontally_irrelevant' if not relevant else 'not_above'
    )
    return RelationEvidence(
        'above', subject.entity_id, anchor.entity_id, confidence, accepted,
        status, gap, metrics.subject_overlap, metrics.centre_distance_m,
        geometry,
    )


def below_evidence(
    subject: SupportGeometry,
    anchor: SupportGeometry,
    config: VerticalRelationConfig | None = None,
) -> RelationEvidence:
    """Evaluate below(A, B) as the exact inverse of above(B, A)."""
    inverse = above_evidence(anchor, subject, config)
    return RelationEvidence(
        'below', subject.entity_id, anchor.entity_id, inverse.confidence,
        inverse.accepted, inverse.status, inverse.vertical_gap_m,
        inverse.subject_support_overlap, inverse.horizontal_distance_m,
        inverse.geometry_confidence,
    )


__all__ = ['RelationEvidence', 'VerticalRelationConfig', 'above_evidence',
           'below_evidence']
