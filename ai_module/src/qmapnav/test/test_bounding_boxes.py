"""Synthetic tests for robust upright boxes and canonical yaw."""

from math import pi

import numpy as np
import pytest

from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.bounding_boxes import canonicalize_box
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import rectangle_yaw_difference
from qmapnav.mapping.bounding_boxes import robust_aabb
from qmapnav.mapping.bounding_boxes import wrap_half_pi


def _box_points(
    *,
    centre=(2.0, -1.0, 0.8),
    dimensions=(2.0, 0.8, 1.2),
    yaw=0.0,
    samples=(13, 9, 7),
) -> np.ndarray:
    axes = [
        np.linspace(-size / 2.0, size / 2.0, count)
        for size, count in zip(dimensions, samples)
    ]
    local = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    points = local.copy()
    points[:, :2] = local[:, :2] @ rotation.T
    return points + np.asarray(centre)


@pytest.mark.parametrize('degrees', [0.0, 30.0, 45.0, 80.0])
def test_upright_obb_recovers_rotated_box(degrees: float) -> None:
    yaw = np.deg2rad(degrees)
    points = _box_points(yaw=yaw)
    policy = BoxEstimationConfig(lower_percentile=0.0, upper_percentile=100.0)

    box = estimate_upright_obb(points, policy)

    np.testing.assert_allclose(box.centre_xyz, [2.0, -1.0, 0.8], atol=1e-10)
    np.testing.assert_allclose(box.dimensions_xyz, [2.0, 0.8, 1.2], atol=1e-10)
    assert rectangle_yaw_difference(box.yaw_rad, yaw) < 1e-10
    assert -pi / 2.0 <= box.yaw_rad < pi / 2.0


def test_boxes_are_deterministic_under_translation_shuffle_and_outliers() -> None:
    points = _box_points(yaw=np.deg2rad(32.0))
    outliers = np.array([[30.0, 30.0, 20.0], [-20.0, -20.0, -5.0]])
    contaminated = np.vstack((points, outliers))
    shuffled = contaminated[np.random.default_rng(42).permutation(len(contaminated))]

    first = estimate_upright_obb(contaminated)
    second = estimate_upright_obb(shuffled)
    aabb = robust_aabb(contaminated)

    np.testing.assert_allclose(first.centre_xyz, second.centre_xyz)
    np.testing.assert_allclose(first.dimensions_xyz, second.dimensions_xyz)
    assert first.dimensions_xyz[0] < 3.0
    assert aabb.dimensions_xyz[0] < 3.0


def test_canonicalization_and_half_pi_boundaries() -> None:
    length, width, yaw = canonicalize_box(0.5, 2.0, 0.0)
    assert (length, width) == (2.0, 0.5)
    assert yaw == -pi / 2.0
    assert wrap_half_pi(pi / 2.0) == -pi / 2.0
    assert wrap_half_pi(pi) == 0.0
    assert rectangle_yaw_difference(0.0, pi) == 0.0


@pytest.mark.parametrize(
    'points',
    [
        np.empty((0, 3)),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.0, 0.0, 1.0]]),
    ],
)
def test_obb_rejects_insufficient_or_degenerate_points(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        estimate_upright_obb(points)
