"""Thin counting wrapper over Day 11 support-surface exploration."""

from dataclasses import dataclass
from dataclasses import replace
from math import isfinite

from qmapnav.counting.numerical_result import NumericalResult
from qmapnav.exploration.small_object_mode import decide_small_object_mode
from qmapnav.exploration.small_object_mode import is_support_surface
from qmapnav.exploration.small_object_mode import likely_supports
from qmapnav.exploration.support_surface_search import (
    generate_support_surface_viewpoints,
)
from qmapnav.exploration.support_surface_search import rank_support_surfaces
from qmapnav.exploration.support_surface_search import SupportSearchHistory
from qmapnav.exploration.viewpoint_generation import CandidateGenerationOutcome
from qmapnav.exploration.viewpoint_generation import ViewpointGenerationConfig
from qmapnav.exploration.viewpoint_generation import VisitedViewpoint
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import perceived_box


@dataclass(frozen=True)
class CountingSupportAssessment:
    """Coverage and negative evidence for plausible target supports."""

    target_class: str
    small_object_mode: bool
    support_instance_ids: tuple[str, ...]
    exhausted_support_ids: tuple[str, ...]
    all_plausible_supports_exhausted: bool
    negative_evidence_confidence: float

    def __post_init__(self) -> None:
        if not self.target_class.strip():
            raise ValueError('target_class must be non-empty')
        for name in ('small_object_mode', 'all_plausible_supports_exhausted'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be boolean')
        if not isfinite(self.negative_evidence_confidence) or not (
            0.0 <= self.negative_evidence_confidence <= 1.0
        ):
            raise ValueError('negative evidence confidence must lie in [0, 1]')

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe support coverage."""
        return {
            'target_class': self.target_class,
            'small_object_mode': self.small_object_mode,
            'support_instance_ids': list(self.support_instance_ids),
            'exhausted_support_ids': list(self.exhausted_support_ids),
            'all_plausible_supports_exhausted': (
                self.all_plausible_supports_exhausted
            ),
            'negative_evidence_confidence': (
                self.negative_evidence_confidence
            ),
        }


def assess_counting_supports(
    target_class: str,
    object_map: ObjectMap,
    history: SupportSearchHistory,
) -> CountingSupportAssessment:
    """Audit plausible supports and strong target-specific negative evidence."""
    canonical = canonicalize_class_name(target_class)
    if not isinstance(object_map, ObjectMap):
        raise TypeError('object_map must be ObjectMap')
    if not isinstance(history, SupportSearchHistory):
        raise TypeError('history must be SupportSearchHistory')
    preferred = likely_supports(canonical)
    supports = []
    for instance in object_map.active_instances():
        class_name = max(
            instance.class_scores,
            key=lambda value: (instance.class_scores[value], value),
        )
        class_name = canonicalize_class_name(class_name)
        if not is_support_surface(class_name):
            continue
        if preferred and class_name not in preferred:
            continue
        supports.append(perceived_box(instance, class_name=class_name))
    support_ids = tuple(sorted(item.object_id for item in supports))
    exhausted = []
    confidences = []
    for identifier in support_ids:
        record = history.record_for(identifier)
        searched = canonical in record.target_classes_searched
        if record.exhausted and searched:
            exhausted.append(identifier)
            confidences.append(record.search_confidence)
    all_exhausted = bool(support_ids) and len(exhausted) == len(support_ids)
    confidence = min(confidences) if all_exhausted else 0.0
    mode = decide_small_object_mode(
        canonical,
        detected=False,
        support_instance_ids=support_ids,
    )
    return CountingSupportAssessment(
        canonical,
        mode.active,
        support_ids,
        tuple(exhausted),
        all_exhausted,
        confidence,
    )


def strengthen_zero_with_support_evidence(
    result: NumericalResult,
    assessment: CountingSupportAssessment,
) -> NumericalResult:
    """Raise confidence in zero only after every plausible support is exhausted."""
    if not isinstance(result, NumericalResult):
        raise TypeError('result must be NumericalResult')
    if not isinstance(assessment, CountingSupportAssessment):
        raise TypeError('assessment has the wrong type')
    if result.count != 0 or not assessment.all_plausible_supports_exhausted:
        return result
    return replace(
        result,
        count_confidence=max(
            result.count_confidence,
            assessment.negative_evidence_confidence,
        ),
        stability_reason='strong_negative_evidence_on_all_plausible_supports',
    )


def counting_support_viewpoints(
    target_class: str,
    object_map: ObjectMap,
    history: SupportSearchHistory,
    *,
    grid: OccupancyGrid2D,
    current_pose_xy_yaw: tuple[float, float, float],
    config: ViewpointGenerationConfig | None = None,
    visited: tuple[VisitedViewpoint, ...] = (),
    max_travel_m: float | None = None,
) -> CandidateGenerationOutcome:
    """Reuse Day 11 ranking and viewpoint generation for the next support."""
    assessment = assess_counting_supports(target_class, object_map, history)
    by_id = {
        str(item.instance_id): item for item in object_map.active_instances()
    }
    supports = tuple(
        perceived_box(by_id[identifier])
        for identifier in assessment.support_instance_ids
    )
    ranked = rank_support_surfaces(
        canonicalize_class_name(target_class),
        supports,
        current_pose_xy_yaw=current_pose_xy_yaw,
        history=history,
    )
    if not ranked:
        return CandidateGenerationOutcome(candidates=(), rejected_counts={})
    return generate_support_surface_viewpoints(
        ranked[0],
        grid=grid,
        current_pose_xy_yaw=current_pose_xy_yaw,
        config=config,
        visited=visited,
        max_travel_m=max_travel_m,
    )


__all__ = [
    'CountingSupportAssessment',
    'assess_counting_supports',
    'counting_support_viewpoints',
    'strengthen_zero_with_support_evidence',
]
