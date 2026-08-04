"""Validated single-observation 3D object-candidate contracts."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np


class GeometryStatus(str, Enum):
    """Outcome of one Day 6 lifting attempt."""

    GOOD = 'good'
    SPARSE = 'sparse'
    NO_POINTS = 'no_points'
    GROUND_DOMINATED = 'ground_dominated'
    BACKGROUND_CONTAMINATED = 'background_contaminated'
    MULTIPLE_CLUSTERS = 'multiple_clusters'
    UNSTABLE_ORIENTATION = 'unstable_orientation'
    INVALID_GEOMETRY = 'invalid_geometry'


class GeometrySource(str, Enum):
    """Projected cloud used for one single-observation candidate."""

    CURRENT = 'current'
    ACCUMULATED = 'accumulated'
    COMBINED = 'combined'


def readonly_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int | None, ...],
    *,
    dtype: np.dtype | type = np.float64,
    finite: bool = True,
) -> np.ndarray:
    """Validate and defensively copy one immutable numeric array."""
    array = np.asarray(value, dtype=dtype)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f'{name} must have shape {shape}')
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class LiftingCounts:
    """Point counts retained at every lifting stage."""

    projected: int
    box_selected: int
    mask_selected: int
    post_ground: int
    post_depth: int
    clustered: int
    final: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in values):
            raise ValueError('lifting counts must be non-negative integers')
        if self.final > self.clustered or self.clustered > self.post_depth:
            raise ValueError('lifting counts must not increase after clustering')


@dataclass(frozen=True)
class ConfidenceComponents:
    """Auditable components of geometry and orientation confidence."""

    point_support: float
    anisotropy: float
    estimator_agreement: float
    resampling_stability: float
    depth_consistency: float
    cluster_purity: float
    image_coverage: float
    timing_quality: float
    boundary_support: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be finite and in [0, 1]')


@dataclass(frozen=True)
class ObjectCandidate3D:
    """One detector-independent object candidate from one observation."""

    candidate_id: str
    detection_id: str
    class_name: str
    detection_confidence: float
    source: GeometrySource
    source_timestamp_ns: int
    image_timestamp_ns: int
    scan_timestamp_ns: int
    pose_timestamp_ns: int
    pose_mode: str
    image_scan_delta_ms: float
    pose_before_delta_ms: float | None
    pose_after_delta_ms: float | None
    timing_warning: bool
    points_map_xyz: np.ndarray
    source_projection_indices: np.ndarray
    point_centroid_xyz: np.ndarray
    aabb_min_xyz: np.ndarray
    aabb_max_xyz: np.ndarray
    obb_centre_xyz: np.ndarray
    obb_dimensions_xyz: np.ndarray
    obb_yaw_rad: float
    estimated_yaw_rad: float
    orientation_confidence: float
    geometry_confidence: float
    geometry_status: GeometryStatus
    partial_geometry: bool
    low_orientation_fallback: bool
    counts: LiftingCounts
    confidence_components: ConfidenceComponents
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ('candidate_id', 'detection_id', 'class_name'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')
        if not isfinite(self.detection_confidence) or not (
            0.0 <= self.detection_confidence <= 1.0
        ):
            raise ValueError('detection_confidence must lie in [0, 1]')
        if not isinstance(self.source, GeometrySource):
            raise TypeError('source must be GeometrySource')
        for name in (
            'source_timestamp_ns',
            'image_timestamp_ns',
            'scan_timestamp_ns',
            'pose_timestamp_ns',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if not isinstance(self.pose_mode, str) or not self.pose_mode:
            raise ValueError('pose_mode must be non-empty')
        for name in (
            'image_scan_delta_ms',
            'pose_before_delta_ms',
            'pose_after_delta_ms',
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f'{name} must be finite when provided')
        if not isinstance(self.timing_warning, bool):
            raise TypeError('timing_warning must be bool')
        points = readonly_array('points_map_xyz', self.points_map_xyz, (None, 3))
        indices = readonly_array(
            'source_projection_indices',
            self.source_projection_indices,
            (points.shape[0],),
            dtype=np.int64,
        )
        if len(set(indices.tolist())) != indices.size:
            raise ValueError('source_projection_indices must be unique')
        object.__setattr__(self, 'points_map_xyz', points)
        object.__setattr__(self, 'source_projection_indices', indices)
        for name in (
            'point_centroid_xyz',
            'aabb_min_xyz',
            'aabb_max_xyz',
            'obb_centre_xyz',
            'obb_dimensions_xyz',
        ):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (3,)),
            )
        if np.any(self.aabb_max_xyz <= self.aabb_min_xyz):
            raise ValueError('AABB maximum must exceed minimum on every axis')
        if np.any(self.obb_dimensions_xyz <= 0.0):
            raise ValueError('OBB dimensions must be strictly positive')
        if not all(
            isfinite(value)
            for value in (self.obb_yaw_rad, self.estimated_yaw_rad)
        ):
            raise ValueError('yaw values must be finite')
        for name in ('orientation_confidence', 'geometry_confidence'):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        if not isinstance(self.geometry_status, GeometryStatus):
            raise TypeError('geometry_status must be GeometryStatus')
        if not isinstance(self.counts, LiftingCounts):
            raise TypeError('counts must be LiftingCounts')
        if self.counts.final != points.shape[0]:
            raise ValueError('final count must equal candidate point count')
        if not isinstance(self.confidence_components, ConfidenceComponents):
            raise TypeError('confidence_components must be ConfidenceComponents')
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))

    @property
    def point_count(self) -> int:
        """Return the number of final cleaned points."""
        return int(self.points_map_xyz.shape[0])


@dataclass(frozen=True)
class LiftingResult:
    """Structured success or normal failure from one lifting attempt."""

    detection_id: str
    status: GeometryStatus
    candidate: ObjectCandidate3D | None
    counts: LiftingCounts
    reason: str
    processing_time_ms: float
    stage_indices: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, str) or not self.detection_id:
            raise ValueError('detection_id must be non-empty')
        if not isinstance(self.status, GeometryStatus):
            raise TypeError('status must be GeometryStatus')
        if self.candidate is not None and self.candidate.geometry_status is not self.status:
            raise ValueError('candidate and result statuses must agree')
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError('reason must be non-empty')
        if not isfinite(self.processing_time_ms) or self.processing_time_ms < 0.0:
            raise ValueError('processing_time_ms must be finite and non-negative')
        frozen_indices = {}
        for name, values in self.stage_indices.items():
            frozen_indices[str(name)] = readonly_array(
                str(name), values, (None,), dtype=np.int64
            )
        object.__setattr__(self, 'stage_indices', MappingProxyType(frozen_indices))
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))


__all__ = [
    'ConfidenceComponents',
    'GeometrySource',
    'GeometryStatus',
    'LiftingCounts',
    'LiftingResult',
    'ObjectCandidate3D',
    'readonly_array',
]
