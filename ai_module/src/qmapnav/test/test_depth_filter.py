"""Tests for foreground depth-layer rejection."""

import numpy as np

from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.depth_filter import select_foreground_depth_layer


def test_nearest_substantial_object_wins_over_larger_wall() -> None:
    foreground = np.linspace(1.9, 2.3, 20)
    wall = np.linspace(4.8, 5.2, 100)
    outliers = np.array([0.3, 0.4])
    depths = np.concatenate((outliers, foreground, wall))

    result = select_foreground_depth_layer(depths)

    selected = depths[result.kept_indices]
    assert np.min(selected) >= 1.8
    assert np.max(selected) < 3.0
    assert result.mode_support >= 20
    assert result.contamination_fraction > 0.5


def test_sparse_depths_use_front_percentile_without_crashing() -> None:
    result = select_foreground_depth_layer(
        np.array([2.0, 2.1, 5.0]),
        DepthFilterConfig(minimum_mode_points=5),
    )

    assert result.used_fallback
    assert result.kept_indices.size >= 1


def test_empty_depth_result_is_valid() -> None:
    result = select_foreground_depth_layer(np.empty(0))
    assert result.kept_indices.size == 0
    assert result.reason == 'empty'
