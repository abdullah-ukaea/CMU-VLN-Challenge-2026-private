"""Tests for composing Day 6 lifting over Day 5 projection frames."""

import numpy as np
import pytest

from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.lifting_pipeline import combine_projection_results


def _projection(image_id: str, offset: float, count: int) -> ProjectionResult:
    points = np.column_stack(
        (
            np.full(count, 2.0 + offset),
            np.linspace(-0.1, 0.1, count),
            np.linspace(0.2, 0.8, count),
        )
    )
    return ProjectionResult(
        image_id=image_id,
        image_timestamp_ns=100,
        scan_timestamp_ns=90 + int(offset),
        transform_camera_internal_from_map=np.eye(4),
        source_valid_mask=np.ones(count, dtype=np.bool_),
        source_point_indices=np.arange(count),
        points_map_xyz=points,
        points_camera_xyz=points,
        panorama_uv=np.column_stack(
            (np.linspace(40.0, 60.0, count), np.full(count, 30.0))
        ),
        euclidean_range_m=np.linalg.norm(points, axis=1),
        forward_depth_m=points[:, 0],
        intensity=None,
        diagnostics=ProjectionDiagnostics(
            count, count, count, count, abs(offset), 'exact', 0.0, 0.0, False
        ),
    )


def test_combined_projection_preserves_both_sources() -> None:
    current = _projection('image', 0.0, 3)
    accumulated = _projection('image', 2.0, 4)

    combined = combine_projection_results(current, accumulated)

    assert combined.point_count == 7
    np.testing.assert_allclose(combined.points_map_xyz[:3], current.points_map_xyz)
    np.testing.assert_allclose(
        combined.points_map_xyz[3:], accumulated.points_map_xyz
    )
    np.testing.assert_array_equal(combined.source_point_indices, np.arange(7))
    assert combined.diagnostics.projected_point_count == 7


def test_combined_projection_rejects_different_images() -> None:
    with pytest.raises(ValueError, match='same image'):
        combine_projection_results(
            _projection('one', 0.0, 2),
            _projection('two', 0.0, 2),
        )
