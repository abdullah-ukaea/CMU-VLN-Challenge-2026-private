"""Tests for deterministic bounded Day 6 geometry diagnostics."""

import numpy as np

from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.lifting_visualisation import draw_candidate_orthographic
from qmapnav.mapping.lifting_visualisation import draw_depth_histogram
from qmapnav.mapping.lifting_visualisation import draw_lifting_stage_overlay
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _empty_projection() -> ProjectionResult:
    return ProjectionResult(
        'image', 1, 1, np.eye(4), np.empty(0, dtype=np.bool_),
        np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty((0, 3)),
        np.empty((0, 2)), np.empty(0), np.empty(0), None,
        ProjectionDiagnostics(0, 0, 0, 0, 0.0, 'exact', 0.0, 0.0, False),
    )


def _detection() -> Detection2D:
    box = PanoramaBox(
        360, 120, ((100.0, 200.0),), 20.0, 80.0,
        np.array([[100.0, 20.0], [200.0, 20.0], [200.0, 80.0], [100.0, 80.0]]),
    )
    return Detection2D(
        'chair:0', 'chair', 'chair', 0.8, box, (0,),
        ((1.0, 1.0, 10.0, 10.0),), (150.0, 50.0),
        np.array([1.0, 0.0, 0.0]),
    )


def _empty_result() -> LiftingResult:
    return LiftingResult(
        'chair:0', GeometryStatus.NO_POINTS, None,
        LiftingCounts(0, 0, 0, 0, 0, 0, 0), 'no points', 0.0, {}, {},
    )


def test_empty_visualisations_are_valid_and_deterministic() -> None:
    panorama = np.zeros((120, 360, 3), dtype=np.uint8)
    projection = _empty_projection()
    result = _empty_result()

    first = draw_lifting_stage_overlay(panorama, projection, _detection(), result)
    second = draw_lifting_stage_overlay(panorama, projection, _detection(), result)
    histogram = draw_depth_histogram(projection, result)
    geometry = draw_candidate_orthographic(result, np.zeros(3))

    np.testing.assert_array_equal(first, second)
    assert first.shape == panorama.shape
    assert histogram.shape == (360, 640, 3)
    assert geometry.shape == (700, 1400, 3)
    assert np.count_nonzero(first) > 0
