"""Tests for Day 6 centre, dimension, yaw, and overlap metrics."""

import numpy as np

from qmapnav.mapping.box_overlap import aabb_iou_3d
from qmapnav.mapping.box_overlap import point_count_bin
from qmapnav.mapping.box_overlap import upright_box_iou_3d


def test_aabb_iou_identical_disjoint_contained_and_offset_height() -> None:
    minimum = np.array([0.0, 0.0, 0.0])
    maximum = np.array([2.0, 2.0, 2.0])

    assert aabb_iou_3d(minimum, maximum, minimum, maximum) == 1.0
    assert aabb_iou_3d(minimum, maximum, minimum + 3.0, maximum + 3.0) == 0.0
    contained = aabb_iou_3d(
        minimum, maximum, np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])
    )
    assert np.isclose(contained, 1.0 / 8.0)
    half_height = aabb_iou_3d(
        minimum, maximum, np.array([0.0, 0.0, 1.0]), np.array([2.0, 2.0, 3.0])
    )
    assert np.isclose(half_height, 1.0 / 3.0)


def test_oriented_iou_identical_disjoint_and_rotated() -> None:
    centre = np.array([0.0, 0.0, 1.0])
    dimensions = np.array([2.0, 1.0, 2.0])

    identical = upright_box_iou_3d(
        centre, dimensions, 0.3, centre, dimensions, 0.3
    )
    disjoint = upright_box_iou_3d(
        centre, dimensions, 0.0, centre + np.array([5.0, 0.0, 0.0]), dimensions, 0.0
    )
    rotated = upright_box_iou_3d(
        centre, dimensions, 0.0, centre, dimensions, np.pi / 2.0
    )

    assert np.isclose(identical, 1.0)
    assert disjoint == 0.0
    assert 0.0 < rotated < 1.0
    assert np.isclose(rotated, 1.0 / 3.0)


def test_point_count_bins_cover_boundaries() -> None:
    assert [point_count_bin(value) for value in (0, 1, 5, 6, 10, 11, 30, 31, 100, 101)] == [
        '0', '1-5', '1-5', '6-10', '6-10', '11-30', '11-30',
        '31-100', '31-100', '>100',
    ]
