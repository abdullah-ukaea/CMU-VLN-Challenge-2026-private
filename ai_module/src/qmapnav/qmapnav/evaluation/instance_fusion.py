"""Day 7 persistent-identity, fusion, and structural-anchor metrics."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.bounding_boxes import rectangle_yaw_difference
from qmapnav.mapping.geometry_evaluation import upright_box_iou_3d
from qmapnav.mapping.structural_map import StructuralAnchor


@dataclass(frozen=True)
class IdentityAssignment:
    """One observation's physical identity and predicted persistent ID."""

    physical_object_id: str
    predicted_instance_id: int

    def __post_init__(self) -> None:
        if not self.physical_object_id:
            raise ValueError('physical_object_id must be non-empty')
        if self.predicted_instance_id < 0:
            raise ValueError('predicted_instance_id must be non-negative')


@dataclass(frozen=True)
class IdentityMetrics:
    """Duplicate, false-merge, and persistent-ID consistency summary."""

    physical_object_count: int
    predicted_instance_count: int
    extra_instance_count: int
    duplicate_rate: float
    false_merge_count: int
    mean_ids_per_physical_object: float
    maximum_ids_per_physical_object: int


@dataclass(frozen=True)
class FusionGeometryMetrics:
    """First-view versus fused centre, box, and orientation quality."""

    first_centre_error_m: float
    fused_centre_error_m: float
    centre_error_improvement_m: float
    first_dimension_error_m: float
    fused_dimension_error_m: float
    dimension_error_improvement_m: float
    first_oriented_iou_3d: float
    fused_oriented_iou_3d: float
    oriented_iou_improvement: float
    first_yaw_error_rad: float
    fused_yaw_error_rad: float
    yaw_error_improvement_rad: float


@dataclass(frozen=True)
class AnchorStabilityMetrics:
    """Repeated-view anchor variance and supporting-wall consistency."""

    observation_count: int
    mean_position_xyz: np.ndarray
    position_standard_deviation_m: float
    maximum_position_error_m: float
    supporting_wall_consistent: bool

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_position_xyz, dtype=np.float64).copy()
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise ValueError('mean_position_xyz must be finite shape (3,)')
        mean.setflags(write=False)
        object.__setattr__(self, 'mean_position_xyz', mean)


def evaluate_identity_assignments(
    assignments: list[IdentityAssignment],
) -> IdentityMetrics:
    """Measure duplicates and false merges from labelled observations."""
    if not assignments:
        return IdentityMetrics(0, 0, 0, 0.0, 0, 0.0, 0)
    by_physical: dict[str, set[int]] = {}
    by_prediction: dict[int, set[str]] = {}
    for assignment in assignments:
        if not isinstance(assignment, IdentityAssignment):
            raise TypeError('assignments must contain IdentityAssignment')
        by_physical.setdefault(
            assignment.physical_object_id, set()
        ).add(assignment.predicted_instance_id)
        by_prediction.setdefault(
            assignment.predicted_instance_id, set()
        ).add(assignment.physical_object_id)
    counts = [len(values) for values in by_physical.values()]
    extra = sum(max(0, count - 1) for count in counts)
    false_merges = sum(
        1 for values in by_prediction.values() if len(values) > 1
    )
    return IdentityMetrics(
        physical_object_count=len(by_physical),
        predicted_instance_count=len(by_prediction),
        extra_instance_count=extra,
        duplicate_rate=extra / len(by_physical),
        false_merge_count=false_merges,
        mean_ids_per_physical_object=float(np.mean(counts)),
        maximum_ids_per_physical_object=max(counts),
    )


def evaluate_fusion_geometry(
    first_centre_xyz: np.ndarray,
    first_dimensions_xyz: np.ndarray,
    fused_instance: ObjectInstance,
    reference_centre_xyz: np.ndarray,
    reference_dimensions_xyz: np.ndarray,
    *,
    first_yaw_rad: float = 0.0,
    reference_yaw_rad: float = 0.0,
) -> FusionGeometryMetrics:
    """Compare first-view and fused geometry against a reference cuboid."""
    first_centre = _vector(first_centre_xyz, 'first_centre_xyz')
    first_dimensions = _positive_vector(
        first_dimensions_xyz, 'first_dimensions_xyz'
    )
    reference_centre = _vector(
        reference_centre_xyz, 'reference_centre_xyz'
    )
    reference_dimensions = _positive_vector(
        reference_dimensions_xyz, 'reference_dimensions_xyz'
    )
    if not isfinite(first_yaw_rad) or not isfinite(reference_yaw_rad):
        raise ValueError('fusion yaw inputs must be finite')
    first_centre_error = float(np.linalg.norm(
        first_centre - reference_centre
    ))
    fused_centre_error = float(np.linalg.norm(
        fused_instance.centroid_xyz - reference_centre
    ))
    first_dimension_error = float(np.linalg.norm(
        first_dimensions - reference_dimensions
    ))
    fused_dimension_error = float(np.linalg.norm(
        fused_instance.obb_dimensions - reference_dimensions
    ))
    first_iou = upright_box_iou_3d(
        first_centre,
        first_dimensions,
        first_yaw_rad,
        reference_centre,
        reference_dimensions,
        reference_yaw_rad,
    )
    fused_iou = upright_box_iou_3d(
        fused_instance.centroid_xyz,
        fused_instance.obb_dimensions,
        fused_instance.obb_yaw,
        reference_centre,
        reference_dimensions,
        reference_yaw_rad,
    )
    first_yaw_error = rectangle_yaw_difference(
        first_yaw_rad, reference_yaw_rad
    )
    fused_yaw_error = rectangle_yaw_difference(
        fused_instance.obb_yaw, reference_yaw_rad
    )
    return FusionGeometryMetrics(
        first_centre_error,
        fused_centre_error,
        first_centre_error - fused_centre_error,
        first_dimension_error,
        fused_dimension_error,
        first_dimension_error - fused_dimension_error,
        first_iou,
        fused_iou,
        fused_iou - first_iou,
        first_yaw_error,
        fused_yaw_error,
        first_yaw_error - fused_yaw_error,
    )


def evaluate_anchor_stability(
    observations: list[StructuralAnchor],
) -> AnchorStabilityMetrics:
    """Measure repeated structural position variance and wall consistency."""
    if not observations:
        raise ValueError('at least one structural observation is required')
    if not all(isinstance(item, StructuralAnchor) for item in observations):
        raise TypeError('observations must contain StructuralAnchor values')
    positions = np.vstack([item.position_xyz for item in observations])
    mean = np.mean(positions, axis=0)
    errors = np.linalg.norm(positions - mean, axis=1)
    wall_ids = {item.supporting_wall_id for item in observations}
    return AnchorStabilityMetrics(
        observation_count=len(observations),
        mean_position_xyz=mean,
        position_standard_deviation_m=float(np.std(errors)),
        maximum_position_error_m=float(np.max(errors)),
        supporting_wall_consistent=len(wall_ids) == 1,
    )


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite shape (3,)')
    return result


def _positive_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = _vector(value, name)
    if np.any(result <= 0.0) or not all(isfinite(item) for item in result):
        raise ValueError(f'{name} must be strictly positive')
    return result


__all__ = [
    'AnchorStabilityMetrics',
    'evaluate_anchor_stability',
    'evaluate_fusion_geometry',
    'evaluate_identity_assignments',
    'FusionGeometryMetrics',
    'IdentityAssignment',
    'IdentityMetrics',
]
