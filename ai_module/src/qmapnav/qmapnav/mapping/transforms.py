"""Validated rigid transforms for map, sensor, and panorama camera frames."""

from math import isfinite

import numpy as np


_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# ROS optical camera: +X right, +Y down, +Z forward.
# Q-MapNav panorama camera: +X forward, +Y left, +Z up.
ROTATION_INTERNAL_FROM_OPTICAL = np.array(
    (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    dtype=np.float64,
)


def validate_rotation(rotation: np.ndarray, *, atol: float = 1e-7) -> np.ndarray:
    """Return a defensive rotation copy after right-handed SO(3) validation."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation must be a finite 3 x 3 matrix')
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=atol):
        raise ValueError('rotation must be orthonormal')
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=atol):
        raise ValueError('rotation determinant must be +1')
    return np.ascontiguousarray(matrix).copy()


def quaternion_xyzw_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert a finite non-zero XYZW quaternion into a rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError('quaternion_xyzw must contain four finite values')
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError('quaternion_xyzw must have non-zero norm')
    x, y, z, w = quaternion / norm
    return validate_rotation(
        np.array(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
    )


def make_transform(rotation: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a rotation and XYZ translation."""
    checked_rotation = validate_rotation(rotation)
    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError('translation_xyz must contain three finite values')
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = checked_rotation
    transform[:3, 3] = translation
    return transform


def transform_from_pose(
    position_xyz: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> np.ndarray:
    """Build ``T_parent_from_child`` from a ROS-style pose."""
    return make_transform(
        quaternion_xyzw_to_rotation(orientation_xyzw),
        position_xyz,
    )


def validate_transform(transform: np.ndarray, *, atol: float = 1e-7) -> np.ndarray:
    """Return a defensive SE(3) copy after complete homogeneous validation."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('transform must be a finite 4 x 4 matrix')
    if not np.allclose(matrix[3], _BOTTOM_ROW, atol=atol):
        raise ValueError('transform homogeneous bottom row is invalid')
    validate_rotation(matrix[:3, :3], atol=atol)
    return np.ascontiguousarray(matrix).copy()


def invert_transform(transform_a_from_b: np.ndarray) -> np.ndarray:
    """Invert ``T_A_from_B`` without a generic matrix inverse."""
    transform = validate_transform(transform_a_from_b)
    rotation_b_from_a = transform[:3, :3].T
    translation_b_from_a = -rotation_b_from_a @ transform[:3, 3]
    return make_transform(rotation_b_from_a, translation_b_from_a)


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose direction-compatible transforms from left to right."""
    if not transforms:
        return np.eye(4, dtype=np.float64)
    result = validate_transform(transforms[0])
    for transform in transforms[1:]:
        result = result @ validate_transform(transform)
    return validate_transform(result)


def transform_points(
    transform_a_from_b: np.ndarray,
    points_b_xyz: np.ndarray,
) -> np.ndarray:
    """Vectorize ``p_A = T_A_from_B p_B`` for an ``(N, 3)`` point array."""
    transform = validate_transform(transform_a_from_b)
    points = np.asarray(points_b_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_b_xyz must have shape (N, 3)')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_b_xyz must contain only finite values')
    return np.ascontiguousarray(
        points @ transform[:3, :3].T + transform[:3, 3]
    )


def optical_to_internal_transform() -> np.ndarray:
    """Return ``T_internal_from_camera_optical`` with zero translation."""
    return make_transform(ROTATION_INTERNAL_FROM_OPTICAL, np.zeros(3))


def camera_internal_from_map(
    transform_map_from_sensor: np.ndarray,
    transform_sensor_from_camera_optical: np.ndarray,
) -> np.ndarray:
    """Compose the verified map-to-Day-4-panorama-camera transform."""
    transform_sensor_from_map = invert_transform(transform_map_from_sensor)
    transform_camera_optical_from_sensor = invert_transform(
        transform_sensor_from_camera_optical
    )
    return compose_transforms(
        optical_to_internal_transform(),
        transform_camera_optical_from_sensor,
        transform_sensor_from_map,
    )


def is_finite_timestamp_seconds(value: float) -> bool:
    """Return whether a floating timestamp is finite and non-negative."""
    return isfinite(value) and value >= 0.0


__all__ = [
    'ROTATION_INTERNAL_FROM_OPTICAL',
    'camera_internal_from_map',
    'compose_transforms',
    'invert_transform',
    'make_transform',
    'optical_to_internal_transform',
    'quaternion_xyzw_to_rotation',
    'transform_from_pose',
    'transform_points',
    'validate_rotation',
    'validate_transform',
]
