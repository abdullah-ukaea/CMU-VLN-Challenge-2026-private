"""Vectorized projection of registered map points into panorama and crops."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.mapping.timed_buffers import AssociationResult
from qmapnav.mapping.transforms import camera_internal_from_map
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.mapping.transforms import transform_points
from qmapnav.mapping.transforms import validate_transform
from qmapnav.perception.contracts import PerspectiveGeometry
from qmapnav.perception.panorama_projection import camera_rays_to_crop_pixels
from qmapnav.perception.panorama_projection import camera_rays_to_panorama_pixels
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def _readonly(value: np.ndarray, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f'array must have shape {shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError('array must contain only finite values')
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ProjectionConfig:
    """Range, frame, and timing-warning policy for panorama projection."""

    expected_scan_frame: str = 'map'
    expected_pose_parent_frame: str = 'map'
    expected_pose_child_frame: str = 'sensor'
    min_range_m: float = 0.30
    max_range_m: float = 30.0
    timing_warning_ms: float = 100.0

    def __post_init__(self) -> None:
        for name in (
            'expected_scan_frame',
            'expected_pose_parent_frame',
            'expected_pose_child_frame',
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{name} must be non-empty')
        for name in ('min_range_m', 'max_range_m', 'timing_warning_ms'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.min_range_m >= self.max_range_m:
            raise ValueError('min_range_m must be smaller than max_range_m')


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Point filtering and source-time diagnostics for one projection."""

    input_point_count: int
    range_valid_count: int
    vertical_valid_count: int
    projected_point_count: int
    image_scan_delta_ms: float
    pose_mode: str
    pose_before_delta_ms: float | None
    pose_after_delta_ms: float | None
    timing_warning: bool

    @property
    def valid_fraction(self) -> float:
        """Return the projected fraction of the input cloud."""
        if self.input_point_count == 0:
            return 0.0
        return self.projected_point_count / self.input_point_count


@dataclass(frozen=True)
class ProjectionResult:
    """Array-based valid projection plus a mask into the source scan."""

    image_id: str
    image_timestamp_ns: int
    scan_timestamp_ns: int
    transform_camera_internal_from_map: np.ndarray
    source_valid_mask: np.ndarray
    source_point_indices: np.ndarray
    points_map_xyz: np.ndarray
    points_camera_xyz: np.ndarray
    panorama_uv: np.ndarray
    euclidean_range_m: np.ndarray
    forward_depth_m: np.ndarray
    intensity: np.ndarray | None
    diagnostics: ProjectionDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError('image_id must be non-empty')
        transform = validate_transform(self.transform_camera_internal_from_map)
        transform.setflags(write=False)
        object.__setattr__(self, 'transform_camera_internal_from_map', transform)
        mask = np.asarray(self.source_valid_mask, dtype=np.bool_).copy()
        if mask.ndim != 1:
            raise ValueError('source_valid_mask must be one-dimensional')
        mask.setflags(write=False)
        object.__setattr__(self, 'source_valid_mask', mask)
        indices = np.asarray(self.source_point_indices, dtype=np.int64).copy()
        if indices.ndim != 1:
            raise ValueError('source_point_indices must be one-dimensional')
        indices.setflags(write=False)
        object.__setattr__(self, 'source_point_indices', indices)
        count = indices.shape[0]
        for name, shape in (
            ('points_map_xyz', (count, 3)),
            ('points_camera_xyz', (count, 3)),
            ('panorama_uv', (count, 2)),
            ('euclidean_range_m', (count,)),
            ('forward_depth_m', (count,)),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), shape))
        if self.intensity is not None:
            object.__setattr__(self, 'intensity', _readonly(self.intensity, (count,)))
        if int(np.count_nonzero(mask)) != count:
            raise ValueError('source_valid_mask and source indices disagree')

    @property
    def point_count(self) -> int:
        """Return the number of valid panorama projections."""
        return int(self.source_point_indices.shape[0])


@dataclass(frozen=True)
class CropProjection:
    """Projected points visible inside one perception perspective crop."""

    crop_id: int
    crop_uv: np.ndarray
    source_projection_indices: np.ndarray
    source_point_indices: np.ndarray
    euclidean_range_m: np.ndarray
    forward_depth_m: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.source_projection_indices).shape[0]
        object.__setattr__(self, 'crop_uv', _readonly(self.crop_uv, (count, 2)))
        for name in ('source_projection_indices', 'source_point_indices'):
            array = np.asarray(getattr(self, name), dtype=np.int64).copy()
            if array.shape != (count,):
                raise ValueError(f'{name} must have shape ({count},)')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ('euclidean_range_m', 'forward_depth_m'):
            object.__setattr__(self, name, _readonly(getattr(self, name), (count,)))


def project_association(
    association: AssociationResult,
    transform_sensor_from_camera_optical: np.ndarray,
    panorama_model: PanoramaCameraModel,
    config: ProjectionConfig | None = None,
) -> ProjectionResult:
    """Project one synchronized registered scan at the panorama pose."""
    policy = config or ProjectionConfig()
    scan = association.scan
    pose = association.pose
    if scan.frame_id != policy.expected_scan_frame:
        raise ValueError(
            f'unexpected scan frame {scan.frame_id!r}; '
            f'expected {policy.expected_scan_frame!r}'
        )
    if pose.parent_frame_id != policy.expected_pose_parent_frame:
        raise ValueError('unexpected pose parent frame')
    if pose.child_frame_id != policy.expected_pose_child_frame:
        raise ValueError('unexpected pose child frame')
    transform_map_from_sensor = transform_from_pose(
        pose.position_xyz,
        pose.orientation_xyzw,
    )
    transform_camera_from_map = camera_internal_from_map(
        transform_map_from_sensor,
        transform_sensor_from_camera_optical,
    )
    return project_map_points(
        points_map_xyz=scan.points_xyz,
        transform_camera_internal_from_map=transform_camera_from_map,
        panorama_model=panorama_model,
        image_id=association.panorama.image_id,
        image_timestamp_ns=association.panorama.timestamp_ns,
        scan_timestamp_ns=scan.timestamp_ns,
        intensity=scan.intensity,
        pose_mode=association.pose_mode,
        pose_before_delta_ns=association.pose_before_delta_ns,
        pose_after_delta_ns=association.pose_after_delta_ns,
        config=policy,
    )


def project_map_points(
    *,
    points_map_xyz: np.ndarray,
    transform_camera_internal_from_map: np.ndarray,
    panorama_model: PanoramaCameraModel,
    image_id: str,
    image_timestamp_ns: int,
    scan_timestamp_ns: int,
    intensity: np.ndarray | None = None,
    pose_mode: str = 'provided',
    pose_before_delta_ns: int | None = None,
    pose_after_delta_ns: int | None = None,
    config: ProjectionConfig | None = None,
) -> ProjectionResult:
    """Project finite map points using an already composed image-time transform."""
    policy = config or ProjectionConfig()
    points_map = np.asarray(points_map_xyz, dtype=np.float64)
    if points_map.ndim != 2 or points_map.shape[1] != 3:
        raise ValueError('points_map_xyz must have shape (N, 3)')
    if not np.all(np.isfinite(points_map)):
        raise ValueError('points_map_xyz must contain only finite values')
    if intensity is not None:
        source_intensity = np.asarray(intensity, dtype=np.float64)
        if source_intensity.shape != (points_map.shape[0],):
            raise ValueError('intensity must have one value per map point')
    else:
        source_intensity = None
    points_camera = transform_points(
        transform_camera_internal_from_map,
        points_map,
    )
    ranges = np.linalg.norm(points_camera, axis=1)
    range_valid = (
        (ranges >= policy.min_range_m)
        & (ranges <= policy.max_range_m)
        & (ranges > 1e-12)
    )
    candidate_indices = np.flatnonzero(range_valid)
    candidate_camera = points_camera[candidate_indices]
    candidate_ranges = ranges[candidate_indices]
    if candidate_indices.size:
        rays = candidate_camera / candidate_ranges[:, None]
        panorama_uv, vertical_valid = camera_rays_to_panorama_pixels(
            rays,
            panorama_model,
        )
    else:
        panorama_uv = np.empty((0, 2), dtype=np.float64)
        vertical_valid = np.empty((0,), dtype=np.bool_)
    selected_indices = candidate_indices[vertical_valid]
    source_mask = np.zeros(points_map.shape[0], dtype=np.bool_)
    source_mask[selected_indices] = True
    selected_intensity = (
        source_intensity[selected_indices]
        if source_intensity is not None
        else None
    )
    image_scan_delta_ms = (scan_timestamp_ns - image_timestamp_ns) / 1_000_000.0
    diagnostics = ProjectionDiagnostics(
        input_point_count=int(points_map.shape[0]),
        range_valid_count=int(candidate_indices.shape[0]),
        vertical_valid_count=int(np.count_nonzero(vertical_valid)),
        projected_point_count=int(selected_indices.shape[0]),
        image_scan_delta_ms=image_scan_delta_ms,
        pose_mode=pose_mode,
        pose_before_delta_ms=(
            pose_before_delta_ns / 1_000_000.0
            if pose_before_delta_ns is not None
            else None
        ),
        pose_after_delta_ms=(
            pose_after_delta_ns / 1_000_000.0
            if pose_after_delta_ns is not None
            else None
        ),
        timing_warning=abs(image_scan_delta_ms) > policy.timing_warning_ms,
    )
    return ProjectionResult(
        image_id=image_id,
        image_timestamp_ns=image_timestamp_ns,
        scan_timestamp_ns=scan_timestamp_ns,
        transform_camera_internal_from_map=transform_camera_internal_from_map,
        source_valid_mask=source_mask,
        source_point_indices=selected_indices,
        points_map_xyz=points_map[selected_indices],
        points_camera_xyz=points_camera[selected_indices],
        panorama_uv=panorama_uv[vertical_valid],
        euclidean_range_m=ranges[selected_indices],
        forward_depth_m=points_camera[selected_indices, 0],
        intensity=selected_intensity,
        diagnostics=diagnostics,
    )


def project_result_into_crops(
    result: ProjectionResult,
    geometries: tuple[PerspectiveGeometry, ...],
) -> tuple[CropProjection, ...]:
    """Project every panorama-valid point into all covering perspective crops."""
    if result.point_count:
        rays = result.points_camera_xyz / result.euclidean_range_m[:, None]
    else:
        rays = np.empty((0, 3), dtype=np.float64)
    projections = []
    for geometry in geometries:
        crop_uv, visible = camera_rays_to_crop_pixels(rays, geometry)
        indices = np.flatnonzero(visible)
        projections.append(
            CropProjection(
                crop_id=geometry.crop_id,
                crop_uv=crop_uv[indices],
                source_projection_indices=indices,
                source_point_indices=result.source_point_indices[indices],
                euclidean_range_m=result.euclidean_range_m[indices],
                forward_depth_m=result.forward_depth_m[indices],
            )
        )
    return tuple(projections)


__all__ = [
    'CropProjection',
    'ProjectionConfig',
    'ProjectionDiagnostics',
    'ProjectionResult',
    'project_association',
    'project_map_points',
    'project_result_into_crops',
]
