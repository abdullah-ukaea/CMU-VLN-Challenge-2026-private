"""Footprint-aware near, distance ranking, and object-level between."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Sequence

import numpy as np

from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation
from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_geometry import SupportGeometry


@dataclass(frozen=True)
class SpatialRelationConfig:
    """Tunable Day 9 geometric relation policy."""

    minimum_geometry_confidence: float = 0.20
    near_base_margin_m: float = 0.40
    near_size_scale: float = 0.75
    between_projection_tolerance: float = 0.05
    between_max_relative_perpendicular_distance: float = 0.35
    between_min_anchor_separation_m: float = 0.30

    def __post_init__(self) -> None:
        unit = (
            'minimum_geometry_confidence',
            'between_projection_tolerance',
            'between_max_relative_perpendicular_distance',
        )
        for name in unit:
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        for name in (
            'near_base_margin_m',
            'near_size_scale',
            'between_min_anchor_separation_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class DistanceMeasurement:
    """Symmetric horizontal distance with explicit geometry fallback."""

    first_id: str
    second_id: str
    xy_distance_m: float
    footprint_distance_m: float | None
    selected_distance_m: float
    used_footprints: bool
    geometry_confidence: float


@dataclass(frozen=True)
class RankedDistance:
    """One target-anchor distance combination in an exhaustive ranking."""

    target_id: str
    anchor_id: str
    distance_m: float
    score: float
    confidence: float
    measurement: DistanceMeasurement


@dataclass(frozen=True)
class DistanceRanking:
    """Complete closest or farthest target-anchor product."""

    operator: str
    ranked: tuple[RankedDistance, ...]
    raw_margin: float
    normalized_margin: float
    confidence: float


def measure_distance(
    first: SupportGeometry,
    second: SupportGeometry,
    config: SpatialRelationConfig | None = None,
) -> DistanceMeasurement:
    """Prefer reliable footprint distance and otherwise use centres."""
    _require_geometry(first, 'first')
    _require_geometry(second, 'second')
    policy = config or SpatialRelationConfig()
    centre = float(np.linalg.norm(
        first.centre_xyz[:2] - second.centre_xyz[:2]
    ))
    confidence = min(first.confidence, second.confidence)
    reliable = confidence >= policy.minimum_geometry_confidence
    footprint = footprint_metrics(first, second).edge_distance_m
    selected = footprint if reliable else centre
    return DistanceMeasurement(
        first.entity_id,
        second.entity_id,
        centre,
        footprint,
        selected,
        reliable,
        confidence,
    )


def evaluate_near(
    first: SupportGeometry,
    second: SupportGeometry,
    config: SpatialRelationConfig | None = None,
) -> ConstraintEvaluation:
    """Evaluate a symmetric, smooth, size-aware near relation."""
    policy = config or SpatialRelationConfig()
    measurement = measure_distance(first, second, policy)
    first_scale = float(np.sqrt(np.prod(first.dimensions_xyz[:2])))
    second_scale = float(np.sqrt(np.prod(second.dimensions_xyz[:2])))
    threshold = policy.near_base_margin_m + policy.near_size_scale * (
        first_scale + second_scale
    ) / 2.0
    score = exp(
        -(measurement.selected_distance_m ** 2) / (2.0 * threshold ** 2)
    )
    satisfied = measurement.selected_distance_m <= threshold
    if measurement.geometry_confidence < policy.minimum_geometry_confidence:
        satisfied = None
    return ConstraintEvaluation(
        'near',
        score,
        False,
        satisfied,
        measurement.geometry_confidence,
        {
            'xy_distance_m': measurement.xy_distance_m,
            'footprint_distance_m': measurement.footprint_distance_m,
            'selected_distance_m': measurement.selected_distance_m,
            'near_threshold_m': threshold,
            'geometry_confidence': measurement.geometry_confidence,
            'used_footprints': float(measurement.used_footprints),
        },
    )


def rank_distances(
    targets: Sequence[SupportGeometry],
    anchors: Sequence[SupportGeometry],
    operator: str,
    config: SpatialRelationConfig | None = None,
) -> DistanceRanking:
    """Rank every target-anchor combination for closest or farthest."""
    if operator not in {'closest', 'farthest'}:
        raise ValueError('operator must be closest or farthest')
    if not targets or not anchors:
        return DistanceRanking(operator, (), 0.0, 0.0, 0.0)
    policy = config or SpatialRelationConfig()
    measurements = [
        measure_distance(target, anchor, policy)
        for target in targets
        for anchor in anchors
        if target.entity_id != anchor.entity_id
    ]
    reverse = operator == 'farthest'
    ordered = sorted(
        measurements,
        key=lambda item: (
            -item.selected_distance_m if reverse else item.selected_distance_m,
            item.first_id,
            item.second_id,
        ),
    )
    if not ordered:
        return DistanceRanking(operator, (), 0.0, 0.0, 0.0)
    best = ordered[0].selected_distance_m
    scale = max(best, 0.25)
    ranked = []
    for item in ordered:
        delta = (
            best - item.selected_distance_m
            if reverse else item.selected_distance_m - best
        )
        relation_score = exp(-max(0.0, delta) / scale)
        confidence = item.geometry_confidence
        ranked.append(RankedDistance(
            item.first_id,
            item.second_id,
            item.selected_distance_m,
            relation_score,
            confidence,
            item,
        ))
    if len(ordered) == 1:
        raw_margin = best
        normalized = 1.0
    else:
        second = ordered[1].selected_distance_m
        raw_margin = abs(second - best)
        normalized = raw_margin / (max(abs(best), abs(second)) + 1.0e-9)
    confidence = min(ordered[0].geometry_confidence, normalized)
    return DistanceRanking(
        operator, tuple(ranked), raw_margin, normalized, confidence
    )


def evaluate_between(
    subject: SupportGeometry,
    first_anchor: SupportGeometry,
    second_anchor: SupportGeometry,
    config: SpatialRelationConfig | None = None,
) -> ConstraintEvaluation:
    """Evaluate subject position relative to the finite anchor segment."""
    for name, geometry in (
        ('subject', subject),
        ('first_anchor', first_anchor),
        ('second_anchor', second_anchor),
    ):
        _require_geometry(geometry, name)
    if first_anchor.entity_id == second_anchor.entity_id:
        raise ValueError('between requires distinct anchors')
    policy = config or SpatialRelationConfig()
    point = subject.centre_xyz[:2]
    start = first_anchor.centre_xyz[:2]
    end = second_anchor.centre_xyz[:2]
    connector = end - start
    squared = float(np.dot(connector, connector))
    separation = float(np.sqrt(squared))
    confidence = min(
        subject.confidence,
        first_anchor.confidence,
        second_anchor.confidence,
    )
    if separation < policy.between_min_anchor_separation_m:
        return ConstraintEvaluation(
            'between', 0.0, False, False, confidence,
            {
                'projection_t': 0.0,
                'projection_x_m': float(start[0]),
                'projection_y_m': float(start[1]),
                'anchor_separation_m': separation,
                'perpendicular_distance_m': separation,
                'relative_perpendicular_distance': 1.0,
                'endpoint_margin': 0.0,
                'footprint_corridor_support': 0.0,
                'geometry_confidence': confidence,
            },
        )
    projection_t = float(np.dot(point - start, connector) / squared)
    projection = start + projection_t * connector
    perpendicular = float(np.linalg.norm(point - projection))
    relative = perpendicular / separation
    tolerance = policy.between_projection_tolerance
    inside_segment = -tolerance <= projection_t <= 1.0 + tolerance
    not_endpoint = tolerance < projection_t < 1.0 - tolerance
    corridor = _footprint_corridor_support(
        subject, first_anchor, second_anchor, connector, separation
    )
    centred = max(0.0, 1.0 - 2.0 * abs(projection_t - 0.5))
    perpendicular_score = exp(
        -(relative ** 2)
        / (2.0 * policy.between_max_relative_perpendicular_distance ** 2)
    )
    score = centred * perpendicular_score * (0.7 + 0.3 * corridor)
    satisfied = (
        inside_segment
        and not_endpoint
        and relative <= policy.between_max_relative_perpendicular_distance
    )
    if confidence < policy.minimum_geometry_confidence:
        satisfied = None
    return ConstraintEvaluation(
        'between',
        score,
        False,
        satisfied,
        confidence,
        {
            'projection_t': projection_t,
            'projection_x_m': float(projection[0]),
            'projection_y_m': float(projection[1]),
            'anchor_separation_m': separation,
            'perpendicular_distance_m': perpendicular,
            'relative_perpendicular_distance': relative,
            'endpoint_margin': min(projection_t, 1.0 - projection_t),
            'footprint_corridor_support': corridor,
            'geometry_confidence': confidence,
        },
    )


def _footprint_corridor_support(subject, first, second, connector, separation):
    direction = connector / separation
    normal = np.array((-direction[1], direction[0]), dtype=np.float64)
    centres = np.array((first.centre_xyz[:2], second.centre_xyz[:2]))
    longitudinal = np.dot(subject.footprint_xy - centres[0], direction)
    anchor_length = float(np.dot(centres[1] - centres[0], direction))
    within = np.logical_and(longitudinal >= 0.0, longitudinal <= anchor_length)
    normal_offsets = np.abs(np.dot(
        subject.footprint_xy - subject.centre_xyz[:2], normal
    ))
    subject_half_width = max(float(np.max(normal_offsets)), 1.0e-6)
    centre_projection = abs(float(np.dot(
        subject.centre_xyz[:2] - centres.mean(axis=0), normal
    )))
    width_score = exp(-centre_projection / (subject_half_width + separation))
    return float(np.mean(within)) * width_score


def _require_geometry(value, name):
    if not isinstance(value, SupportGeometry):
        raise TypeError(f'{name} must be SupportGeometry')


__all__ = [
    'DistanceMeasurement',
    'DistanceRanking',
    'RankedDistance',
    'SpatialRelationConfig',
    'evaluate_between',
    'evaluate_near',
    'measure_distance',
    'rank_distances',
]
