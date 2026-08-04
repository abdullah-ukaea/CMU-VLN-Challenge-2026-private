"""Validated map-frame camera-ray and bounded-wall intersection."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array


@dataclass(frozen=True)
class RayWallIntersection:
    """One valid forward ray hit and wall-extent diagnostic."""

    position_xyz: np.ndarray
    distance_m: float
    within_extent: bool
    extent_overrun_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'position_xyz',
            readonly_array('position_xyz', self.position_xyz, (3,)),
        )
        if not isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError('distance_m must be finite and positive')
        if not isfinite(self.extent_overrun_m) or self.extent_overrun_m < 0.0:
            raise ValueError('extent_overrun_m must be finite and non-negative')


def transform_camera_ray_to_map(
    centre_camera_ray: np.ndarray,
    transform_map_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return map origin and unit ray, applying no translation to direction."""
    ray = np.asarray(centre_camera_ray, dtype=np.float64)
    transform = np.asarray(transform_map_from_camera, dtype=np.float64)
    if ray.shape != (3,) or not np.all(np.isfinite(ray)):
        raise ValueError('centre_camera_ray must be finite shape (3,)')
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('transform_map_from_camera must be finite shape (4, 4)')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError('camera transform bottom row is invalid')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError('camera transform rotation must be orthonormal')
    map_ray = rotation @ ray
    norm = float(np.linalg.norm(map_ray))
    if norm <= 1e-12:
        raise ValueError('transformed ray must be non-zero')
    return transform[:3, 3].copy(), map_ray / norm


def intersect_ray_with_wall(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    plane_parameters: np.ndarray,
    line_segment_xy: np.ndarray,
    *,
    parallel_epsilon: float = 1.0e-6,
    extent_margin_m: float = 0.25,
) -> RayWallIntersection | None:
    """Intersect a forward unit ray with a vertical bounded wall plane."""
    origin = np.asarray(origin_xyz, dtype=np.float64)
    direction = np.asarray(direction_xyz, dtype=np.float64)
    plane = np.asarray(plane_parameters, dtype=np.float64)
    segment = np.asarray(line_segment_xy, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError('ray origin and direction must have shape (3,)')
    if plane.shape != (4,) or segment.shape != (2, 2):
        raise ValueError('wall plane or line segment has invalid shape')
    if not all(np.all(np.isfinite(value)) for value in (
        origin, direction, plane, segment
    )):
        raise ValueError('ray and wall geometry must be finite')
    if not isfinite(parallel_epsilon) or parallel_epsilon <= 0.0:
        raise ValueError('parallel_epsilon must be finite and positive')
    if not isfinite(extent_margin_m) or extent_margin_m < 0.0:
        raise ValueError('extent_margin_m must be finite and non-negative')
    ray_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(plane[:3]))
    segment_vector = segment[1] - segment[0]
    segment_length = float(np.linalg.norm(segment_vector))
    if ray_norm <= 1e-12 or normal_norm <= 1e-12 or segment_length <= 1e-12:
        raise ValueError('ray, plane normal, and segment must be non-degenerate')
    direction = direction / ray_norm
    normal = plane[:3] / normal_norm
    offset = float(plane[3]) / normal_norm
    denominator = float(normal @ direction)
    if abs(denominator) <= parallel_epsilon:
        return None
    distance = -(float(normal @ origin) + offset) / denominator
    if not isfinite(distance) or distance <= 0.0:
        return None
    position = origin + distance * direction
    axis = segment_vector / segment_length
    along = float((position[:2] - segment[0]) @ axis)
    overrun = max(0.0, -along, along - segment_length)
    if overrun > extent_margin_m:
        return None
    return RayWallIntersection(
        position_xyz=position,
        distance_m=distance,
        within_extent=overrun <= 1e-9,
        extent_overrun_m=overrun,
    )


__all__ = [
    'intersect_ray_with_wall',
    'RayWallIntersection',
    'transform_camera_ray_to_map',
]
