"""Conservative, auditable association of Day 6 object candidates."""

from dataclasses import dataclass
from enum import Enum
from math import exp
from math import isfinite
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.bounding_boxes import rectangle_yaw_difference
from qmapnav.mapping.geometry_evaluation import aabb_iou_3d
from qmapnav.mapping.object_candidate import ObjectCandidate3D


CLASS_ALIASES = {
    'computer_display': 'computer_monitor',
    'computer_screen': 'computer_monitor',
    'couch': 'sofa',
    'display': 'computer_monitor',
    'garbage_bin': 'trash_can',
    'garbage_can': 'trash_can',
    'monitor': 'computer_monitor',
    'television': 'tv',
    'trash_bin': 'trash_can',
    'waste_bin': 'trash_can',
    'wastebasket': 'trash_can',
}
COMPATIBLE_CLASSES = {
    frozenset({'chair', 'stool'}): 0.72,
    frozenset({'computer_monitor', 'tv'}): 0.58,
}
LARGE_CLASSES = frozenset({
    'bed', 'bookshelf', 'cabinet', 'desk', 'refrigerator', 'sofa',
    'table', 'wardrobe',
})
MEDIUM_CLASSES = frozenset({
    'chair', 'night_stand', 'potted_plant', 'stool', 'trash_can',
})


class AssociationDecision(str, Enum):
    """Three-band outcome for one candidate-instance comparison."""

    MERGE = 'merge'
    UNCERTAIN = 'uncertain'
    CREATE_NEW = 'create_new'
    REJECT = 'reject'


@dataclass(frozen=True)
class AssociationConfig:
    """Tunable gates and score weights for conservative association."""

    accept_threshold: float = 0.62
    uncertain_threshold: float = 0.55
    yaw_confidence_threshold: float = 0.50
    small_spatial_gate_m: float = 0.45
    medium_spatial_gate_m: float = 0.75
    large_spatial_gate_m: float = 1.25
    class_weight: float = 0.20
    distance_weight: float = 0.32
    overlap_weight: float = 0.18
    size_weight: float = 0.20
    yaw_weight: float = 0.10

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if not all(isfinite(value) for value in values):
            raise ValueError('association configuration must be finite')
        if not 0.0 < self.uncertain_threshold < self.accept_threshold <= 1.0:
            raise ValueError('association thresholds are invalid')
        if not 0.0 <= self.yaw_confidence_threshold <= 1.0:
            raise ValueError('yaw_confidence_threshold must lie in [0, 1]')
        if min(
            self.small_spatial_gate_m,
            self.medium_spatial_gate_m,
            self.large_spatial_gate_m,
        ) <= 0.0:
            raise ValueError('spatial gates must be positive')
        weights = (
            self.class_weight,
            self.distance_weight,
            self.overlap_weight,
            self.size_weight,
            self.yaw_weight,
        )
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError('association weights must be non-negative')


@dataclass(frozen=True)
class AssociationScore:
    """Component scores, gates, and final decision for one comparison."""

    candidate_id: str
    instance_id: int
    components: Mapping[str, float | None]
    final_score: float
    decision: AssociationDecision
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError('candidate_id must be non-empty')
        if self.instance_id < 0:
            raise ValueError('instance_id must be non-negative')
        if not isfinite(self.final_score) or not 0.0 <= self.final_score <= 1.0:
            raise ValueError('final_score must lie in [0, 1]')
        checked = {}
        for name, value in self.components.items():
            if value is not None and (
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f'component {name} must lie in [0, 1]')
            checked[str(name)] = None if value is None else float(value)
        object.__setattr__(self, 'components', MappingProxyType(checked))
        object.__setattr__(
            self, 'rejection_reasons', tuple(self.rejection_reasons)
        )


def canonicalize_class_name(class_name: str) -> str:
    """Normalize detector labels and apply the shared Day 7 aliases."""
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError('class_name must be non-empty')
    token = re.sub(
        r'[^a-z0-9]+', '_', class_name.strip().casefold()
    ).strip('_')
    if not token:
        raise ValueError('class_name has no alphanumeric content')
    return CLASS_ALIASES.get(token, token)


def class_compatibility(first: str, second: str) -> float:
    """Return configured semantic compatibility in ``[0, 1]``."""
    first_name = canonicalize_class_name(first)
    second_name = canonicalize_class_name(second)
    if first_name == second_name:
        return 1.0
    return COMPATIBLE_CLASSES.get(
        frozenset({first_name, second_name}), 0.0
    )


def score_candidate_instance(
    candidate: ObjectCandidate3D,
    instance: ObjectInstance,
    config: AssociationConfig | None = None,
) -> AssociationScore:
    """Score one candidate against one persistent reasoning snapshot."""
    policy = config or AssociationConfig()
    instance_class = max(
        sorted(instance.class_scores),
        key=lambda name: instance.class_scores[name],
    )
    class_score = class_compatibility(candidate.class_name, instance_class)
    reasons = []
    if class_score <= 0.0:
        reasons.append('incompatible_class')
    distance = float(np.linalg.norm(
        candidate.obb_centre_xyz - instance.centroid_xyz
    ))
    gate = _spatial_gate(candidate, instance, policy)
    if distance > gate:
        reasons.append('centroid_outside_spatial_gate')
    if reasons:
        return AssociationScore(
            candidate.candidate_id,
            instance.instance_id,
            {
                'class': class_score,
                'distance': 0.0,
                'overlap': 0.0,
                'size': 0.0,
                'yaw': None,
            },
            0.0,
            AssociationDecision.REJECT,
            tuple(reasons),
        )
    sigma = max(gate / 2.0, 1e-6)
    distance_score = exp(-(distance ** 2) / (2.0 * sigma ** 2))
    overlap_score = aabb_iou_3d(
        candidate.aabb_min_xyz,
        candidate.aabb_max_xyz,
        instance.aabb_min_xyz,
        instance.aabb_max_xyz,
    )
    size_score = _dimension_similarity(
        candidate.obb_dimensions_xyz, instance.obb_dimensions
    )
    incomplete_geometry = (
        candidate.partial_geometry
        or candidate.geometry_confidence <= 0.45
        or instance.confidence <= 0.45
    )
    if incomplete_geometry:
        overlap_score = None
        size_score = max(size_score, 0.75)
    yaw_score = None
    if (
        candidate.orientation_confidence >= policy.yaw_confidence_threshold
        and instance.orientation_confidence >= policy.yaw_confidence_threshold
        and candidate.geometry_confidence >= 0.50
        and instance.confidence >= 0.50
    ):
        difference = rectangle_yaw_difference(
            candidate.obb_yaw_rad, instance.obb_yaw
        )
        yaw_score = exp(-((difference / 0.45) ** 2) / 2.0)
    components = {
        'class': class_score,
        'distance': distance_score,
        'overlap': overlap_score,
        'size': size_score,
        'yaw': yaw_score,
    }
    weighted = (
        ('class', policy.class_weight),
        ('distance', policy.distance_weight),
        ('overlap', policy.overlap_weight),
        ('size', policy.size_weight),
        ('yaw', policy.yaw_weight),
    )
    active = [(components[name], weight) for name, weight in weighted
              if components[name] is not None and weight > 0.0]
    final = sum(float(value) * weight for value, weight in active) / sum(
        weight for _, weight in active
    )
    if final >= policy.accept_threshold:
        decision = AssociationDecision.MERGE
    elif final >= policy.uncertain_threshold:
        decision = AssociationDecision.UNCERTAIN
    else:
        decision = AssociationDecision.CREATE_NEW
    return AssociationScore(
        candidate.candidate_id,
        instance.instance_id,
        components,
        float(final),
        decision,
    )


def _spatial_gate(
    candidate: ObjectCandidate3D,
    instance: ObjectInstance,
    config: AssociationConfig,
) -> float:
    class_name = canonicalize_class_name(candidate.class_name)
    if class_name in LARGE_CLASSES:
        base = config.large_spatial_gate_m
    elif class_name in MEDIUM_CLASSES:
        base = config.medium_spatial_gate_m
    else:
        base = config.small_spatial_gate_m
    footprint = max(
        float(np.max(candidate.obb_dimensions_xyz[:2])),
        float(np.max(instance.obb_dimensions[:2])),
    )
    confidence_scale = 1.0 + 0.25 * (
        1.0 - min(candidate.geometry_confidence, instance.confidence)
    )
    return max(base, 0.65 * footprint) * confidence_scale


def _dimension_similarity(first: np.ndarray, second: np.ndarray) -> float:
    difference = float(np.sum(np.abs(np.asarray(first) - np.asarray(second))))
    denominator = float(np.sum(first)) + 1e-9
    return exp(-difference / denominator)


__all__ = [
    'AssociationConfig',
    'AssociationDecision',
    'AssociationScore',
    'canonicalize_class_name',
    'class_compatibility',
    'score_candidate_instance',
]
