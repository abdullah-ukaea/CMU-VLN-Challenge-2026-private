"""Tests for explicit map, sensor, optical, and panorama transform directions."""

from math import pi

import numpy as np
import pytest

from qmapnav.mapping.transforms import camera_internal_from_map
from qmapnav.mapping.transforms import invert_transform
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import optical_to_internal_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.mapping.transforms import transform_points
from qmapnav.mapping.transforms import validate_transform


STATIC_CAMERA_QUATERNION = np.array([-0.5, 0.5, -0.5, 0.5])


def test_optical_to_internal_axis_conversion() -> None:
    optical = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    internal = transform_points(optical_to_internal_transform(), optical)

    np.testing.assert_allclose(internal, np.eye(3))


def test_live_static_quaternion_maps_optical_axes_into_sensor_axes() -> None:
    rotation_sensor_from_camera = quaternion_xyzw_to_rotation(
        STATIC_CAMERA_QUATERNION
    )

    np.testing.assert_allclose(
        rotation_sensor_from_camera,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        atol=1e-7,
    )


def test_full_non_identity_map_to_internal_camera_chain() -> None:
    yaw_quaternion = np.array([0.0, 0.0, np.sin(pi / 4.0), np.cos(pi / 4.0)])
    transform_map_from_sensor = transform_from_pose(
        np.array([2.0, 3.0, 0.75]),
        yaw_quaternion,
    )
    transform_sensor_from_camera = make_transform(
        quaternion_xyzw_to_rotation(STATIC_CAMERA_QUATERNION),
        np.array([0.0, 0.0, 0.1]),
    )
    transform_camera_from_map = camera_internal_from_map(
        transform_map_from_sensor,
        transform_sensor_from_camera,
    )
    point_one_metre_in_front = np.array([[2.0, 4.0, 0.85]])

    projected = transform_points(
        transform_camera_from_map,
        point_one_metre_in_front,
    )

    np.testing.assert_allclose(projected, [[1.0, 0.0, 0.0]], atol=1e-7)


def test_transform_inverse_round_trip() -> None:
    transform = transform_from_pose(
        np.array([1.0, -2.0, 0.3]),
        np.array([0.1, -0.2, 0.3, 0.9]),
    )
    points = np.array([[2.0, 3.0, 4.0], [-1.0, 0.5, 0.2]])

    round_trip = transform_points(
        invert_transform(transform),
        transform_points(transform, points),
    )

    np.testing.assert_allclose(round_trip, points, atol=1e-9)


@pytest.mark.parametrize(
    'invalid',
    [
        np.zeros((4, 4)),
        np.diag([1.0, 1.0, -1.0, 1.0]),
        np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_invalid_transforms_fail_loudly(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_transform(invalid)


def test_zero_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match='non-zero'):
        quaternion_xyzw_to_rotation(np.zeros(4))
