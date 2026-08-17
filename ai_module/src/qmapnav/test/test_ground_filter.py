"""Tests for local ground estimation and base-preserving filtering."""

import numpy as np

from qmapnav.mapping.ground_filter import estimate_local_ground_plane
from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.ground_filter import remove_ground_points


def test_estimates_level_and_sloped_ground_at_arbitrary_origin() -> None:
    xy = np.stack(
        np.meshgrid(np.linspace(-3.0, 3.0, 25), np.linspace(-2.0, 2.0, 19)),
        axis=-1,
    ).reshape(-1, 2)
    z = 1.2 + 0.02 * xy[:, 0] - 0.01 * xy[:, 1]
    floor = np.column_stack((xy, z))
    furniture = np.array([[0.0, 0.0, 2.0], [1.0, 0.5, 2.5]])

    estimate = estimate_local_ground_plane(
        np.vstack((floor, furniture)), timestamp_ns=10
    )

    assert estimate.plane is not None
    assert estimate.plane.confidence > 0.5
    assert estimate.plane.normal_xyz[2] > 0.99
    distances = estimate.plane.signed_distance(floor)
    assert np.median(np.abs(distances)) < 1e-6


def test_ground_clearance_preserves_object_above_requested_base_height() -> None:
    plane = GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0)
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.03], [0.2, 0.0, 0.06], [0.3, 0.0, 0.2]]
    )

    furniture = remove_ground_points(points, plane, clearance_m=0.07)
    floor_standing = remove_ground_points(points, plane, clearance_m=0.02)

    assert furniture.kept_indices.tolist() == [3]
    assert floor_standing.kept_indices.tolist() == [1, 2, 3]


def test_missing_ground_is_explicit_non_destructive_fallback() -> None:
    points = np.array([[0.0, 0.0, 0.0]])

    result = remove_ground_points(points, None)

    assert result.kept_indices.tolist() == [0]
    assert result.warning == 'ground_plane_unavailable'
