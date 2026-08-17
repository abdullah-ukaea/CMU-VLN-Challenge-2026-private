"""Consistent derived spatial-relation graph over persistent map entities."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.support_relations import ranked_support_hypotheses
from qmapnav.reasoning.support_relations import SupportRelationConfig
from qmapnav.reasoning.vertical_relations import above_evidence
from qmapnav.reasoning.vertical_relations import RelationEvidence
from qmapnav.reasoning.vertical_relations import VerticalRelationConfig


@dataclass(frozen=True)
class SpatialRelation:
    """Unique ordered graph edge with its supporting measurements."""

    relation: str
    subject_id: str
    anchor_id: str
    confidence: float
    evidence: RelationEvidence
    implied_by: str | None = None


class RelationGraph:
    """Recomputed graph enforcing inverse and support implications."""

    def __init__(
        self,
        vertical_config: VerticalRelationConfig | None = None,
        support_config: SupportRelationConfig | None = None,
    ) -> None:
        self.vertical_config = vertical_config or VerticalRelationConfig()
        self.support_config = support_config or SupportRelationConfig()
        self._edges: dict[tuple[str, str, str], SpatialRelation] = {}
        self._support_hypotheses: dict[str, tuple[RelationEvidence, ...]] = {}
        self._contradictions: list[str] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return the count of complete geometry recomputations."""
        return self._revision

    @property
    def edges(self) -> tuple[SpatialRelation, ...]:
        """Return deterministic unique graph edges."""
        return tuple(self._edges[key] for key in sorted(self._edges))

    @property
    def contradictions(self) -> tuple[str, ...]:
        """Return detected high-confidence vertical contradictions."""
        return tuple(self._contradictions)

    @property
    def support_hypotheses(
        self,
    ) -> Mapping[str, tuple[RelationEvidence, ...]]:
        """Return ranked accepted and uncertain support candidates."""
        return MappingProxyType(dict(self._support_hypotheses))

    def recompute(self, entities: list[SupportGeometry]) -> None:
        """Replace all derived relations after any fused geometry change."""
        self._edges.clear()
        self._support_hypotheses.clear()
        self._contradictions.clear()
        ordered = sorted(entities, key=lambda item: item.entity_id)
        for subject in ordered:
            for anchor in ordered:
                if subject.entity_id == anchor.entity_id:
                    continue
                evidence = above_evidence(
                    subject, anchor, self.vertical_config
                )
                if evidence.accepted:
                    self._insert_evidence(evidence)
                    inverse = RelationEvidence(
                        'below', anchor.entity_id, subject.entity_id,
                        evidence.confidence, True, evidence.status,
                        evidence.vertical_gap_m,
                        evidence.subject_support_overlap,
                        evidence.horizontal_distance_m,
                        evidence.geometry_confidence,
                        'inverse',
                    )
                    self._insert_evidence(inverse, implied_by='above')
            hypotheses = ranked_support_hypotheses(
                subject, ordered, self.support_config
            )
            if hypotheses:
                self._support_hypotheses[subject.entity_id] = hypotheses
            for evidence in hypotheses:
                if not evidence.accepted:
                    continue
                self._insert_evidence(evidence)
                supports = RelationEvidence(
                    'supports', evidence.anchor_id, evidence.subject_id,
                    evidence.confidence, True, evidence.status,
                    evidence.vertical_gap_m, evidence.subject_support_overlap,
                    evidence.horizontal_distance_m,
                    evidence.geometry_confidence, 'inverse',
                )
                self._insert_evidence(supports, implied_by='on')
                above = RelationEvidence(
                    'above', evidence.subject_id, evidence.anchor_id,
                    evidence.confidence, True, evidence.status,
                    evidence.vertical_gap_m, evidence.subject_support_overlap,
                    evidence.horizontal_distance_m,
                    evidence.geometry_confidence, 'implication',
                )
                self._insert_evidence(above, implied_by='on')
                below = RelationEvidence(
                    'below', evidence.anchor_id, evidence.subject_id,
                    evidence.confidence, True, evidence.status,
                    evidence.vertical_gap_m, evidence.subject_support_overlap,
                    evidence.horizontal_distance_m,
                    evidence.geometry_confidence, 'implication',
                )
                self._insert_evidence(below, implied_by='on')
        self._check_contradictions()
        self._revision += 1

    def add_for_diagnostic(self, relation: SpatialRelation) -> None:
        """Insert external evidence to exercise contradiction diagnostics."""
        if relation.subject_id == relation.anchor_id:
            raise ValueError('self-relations are not permitted')
        key = (relation.relation, relation.subject_id, relation.anchor_id)
        self._edges[key] = relation
        self._check_contradictions()

    def _insert_evidence(self, evidence, implied_by=None):
        key = (evidence.relation, evidence.subject_id, evidence.anchor_id)
        relation = SpatialRelation(
            evidence.relation, evidence.subject_id, evidence.anchor_id,
            evidence.confidence, evidence, implied_by,
        )
        previous = self._edges.get(key)
        if previous is None or relation.confidence > previous.confidence:
            self._edges[key] = relation

    def _check_contradictions(self):
        self._contradictions.clear()
        conflicting_keys = set()
        for key, edge in sorted(self._edges.items()):
            relation, subject, anchor = key
            if relation != 'above' or edge.confidence < 0.70:
                continue
            opposite = self._edges.get(('below', subject, anchor))
            if opposite is not None and opposite.confidence >= 0.70:
                self._contradictions.append(
                    f'high-confidence vertical contradiction: '
                    f'{subject} vs {anchor}'
                )
                conflicting_keys.update((
                    ('above', subject, anchor),
                    ('below', subject, anchor),
                ))
        for key in conflicting_keys:
            self._edges.pop(key, None)


__all__ = ['RelationGraph', 'SpatialRelation']
