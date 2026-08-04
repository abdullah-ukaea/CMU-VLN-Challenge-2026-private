"""Auditable orientation confidence and conservative yaw fallback."""

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np

from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import rectangle_yaw_difference
from qmapnav.mapping.bounding_boxes import UprightOrientedBox
from qmapnav.mapping.object_candidate import ConfidenceComponents


@dataclass(frozen=True)
class OrientationConfidenceConfig:
    """Thresholds for yaw evidence and fallback behavior."""

    low_confidence: float = 0.40
    high_confidence: float = 0.70
    full_support_points: int = 100
    anisotropy_ratio_for_full_score: float = 4.0
    agreement_tolerance_rad: float = np.deg2rad(20.0)
    stability_tolerance_rad: float = np.deg2rad(15.0)
    subset_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence < self.high_confidence <= 1.0:
            raise ValueError('orientation confidence thresholds are invalid')
        if self.full_support_points <= 0:
            raise ValueError('full_support_points must be positive')
        if self.anisotropy_ratio_for_full_score <= 1.0:
            raise ValueError('anisotropy full-score ratio must exceed one')
        if not 0.0 < self.subset_fraction < 1.0:
            raise ValueError('subset_fraction must lie in (0, 1)')


@dataclass(frozen=True)
class OrientationConfidenceResult:
    """Confidence, components, and deterministic resampling diagnostics."""

    confidence: float
    components: ConfidenceComponents
    subset_yaws_rad: tuple[float, ...]
    estimator_disagreement_rad: float
    maximum_subset_error_rad: float


def estimate_orientation_confidence(
    points_xyz: np.ndarray,
    box: UprightOrientedBox,
    *,
    depth_iqr_m: float,
    cluster_purity: float,
    image_coverage: float,
    timing_quality: float,
    boundary_fraction: float,
    config: OrientationConfidenceConfig | None = None,
) -> OrientationConfidenceResult:
    """Estimate yaw confidence without treating point count as sufficient."""
    policy = config or OrientationConfidenceConfig()
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError('at least three finite points are required')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_xyz must be finite')
    for name, value in (
        ('depth_iqr_m', depth_iqr_m),
        ('cluster_purity', cluster_purity),
        ('image_coverage', image_coverage),
        ('timing_quality', timing_quality),
        ('boundary_fraction', boundary_fraction),
    ):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')

    point_score = min(1.0, points.shape[0] / policy.full_support_points)
    minor, major = np.sort(np.maximum(box.pca_eigenvalues, 0.0))
    ratio = major / max(minor, 1e-12)
    anisotropy = np.clip(
        (ratio - 1.0) / (policy.anisotropy_ratio_for_full_score - 1.0),
        0.0,
        1.0,
    )
    disagreement = rectangle_yaw_difference(box.yaw_rad, box.pca_yaw_rad)
    agreement = exp(-((disagreement / policy.agreement_tolerance_rad) ** 2))
    subset_yaws = _deterministic_subset_yaws(points, policy.subset_fraction)
    subset_errors = [
        rectangle_yaw_difference(yaw, box.yaw_rad) for yaw in subset_yaws
    ]
    maximum_subset_error = max(subset_errors, default=np.pi / 2.0)
    stability = exp(
        -((maximum_subset_error / policy.stability_tolerance_rad) ** 2)
    )
    depth_consistency = exp(-max(0.0, depth_iqr_m) / 1.0)
    boundary_support = float(np.clip(1.0 - boundary_fraction, 0.0, 1.0))
    components = ConfidenceComponents(
        point_support=float(point_score),
        anisotropy=float(anisotropy),
        estimator_agreement=float(agreement),
        resampling_stability=float(stability),
        depth_consistency=float(depth_consistency),
        cluster_purity=float(np.clip(cluster_purity, 0.0, 1.0)),
        image_coverage=float(np.clip(image_coverage, 0.0, 1.0)),
        timing_quality=float(np.clip(timing_quality, 0.0, 1.0)),
        boundary_support=boundary_support,
    )
    # A geometric mean makes one unsupported precision claim reduce confidence.
    factors = np.array(
        [point_score, anisotropy, agreement, stability, boundary_support],
        dtype=np.float64,
    )
    confidence = float(np.prod(np.maximum(factors, 1e-6)) ** (1.0 / len(factors)))
    confidence *= float(np.sqrt(depth_consistency * components.timing_quality))
    return OrientationConfidenceResult(
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        components=components,
        subset_yaws_rad=subset_yaws,
        estimator_disagreement_rad=float(disagreement),
        maximum_subset_error_rad=float(maximum_subset_error),
    )


def conservative_orientation(
    box: UprightOrientedBox,
    aabb_dimensions_xyz: np.ndarray,
    confidence: float,
    config: OrientationConfidenceConfig | None = None,
) -> tuple[float, np.ndarray, bool, str]:
    """Choose estimated or conservative map-axis marker geometry."""
    policy = config or OrientationConfidenceConfig()
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError('confidence must lie in [0, 1]')
    if confidence >= policy.high_confidence:
        return box.yaw_rad, box.dimensions_xyz.copy(), False, 'supported'
    if confidence >= policy.low_confidence:
        return box.yaw_rad, box.dimensions_xyz.copy(), False, 'uncertain'
    dimensions = np.asarray(aabb_dimensions_xyz, dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError('aabb_dimensions_xyz must contain three positive values')
    return 0.0, dimensions.copy(), True, 'aabb_fallback'


def _deterministic_subset_yaws(
    points_xyz: np.ndarray,
    fraction: float,
) -> tuple[float, ...]:
    count = points_xyz.shape[0]
    subset_count = max(3, int(np.floor(count * fraction)))
    if subset_count >= count:
        return (estimate_upright_obb(points_xyz).yaw_rad,)
    orderings = (
        np.arange(count),
        np.argsort(points_xyz[:, 0], kind='stable'),
        np.argsort(points_xyz[:, 1], kind='stable'),
    )
    yaws = []
    for index, ordering in enumerate(orderings):
        start = min(index * max(1, (count - subset_count) // 2), count - subset_count)
        subset = points_xyz[ordering[start:start + subset_count]]
        try:
            yaws.append(estimate_upright_obb(subset).yaw_rad)
        except ValueError:
            continue
    return tuple(yaws)


__all__ = [
    'conservative_orientation',
    'estimate_orientation_confidence',
    'OrientationConfidenceConfig',
    'OrientationConfidenceResult',
]
