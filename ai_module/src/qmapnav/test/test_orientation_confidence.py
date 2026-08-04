"""Tests for orientation evidence and conservative fallback."""

import numpy as np

from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.orientation_confidence import conservative_orientation
from qmapnav.mapping.orientation_confidence import estimate_orientation_confidence


def _prism(length: float, width: float, count: int = 12) -> np.ndarray:
    x = np.linspace(-length / 2.0, length / 2.0, count)
    y = np.linspace(-width / 2.0, width / 2.0, count)
    z = np.linspace(0.1, 1.0, 4)
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)


def _confidence(points: np.ndarray):
    box = estimate_upright_obb(
        points,
        BoxEstimationConfig(lower_percentile=0.0, upper_percentile=100.0),
    )
    return box, estimate_orientation_confidence(
        points,
        box,
        depth_iqr_m=0.2,
        cluster_purity=1.0,
        image_coverage=0.8,
        timing_quality=1.0,
        boundary_fraction=0.05,
    )


def test_elongated_shape_has_more_orientation_evidence_than_square() -> None:
    elongated_box, elongated = _confidence(_prism(3.0, 0.5))
    _, square = _confidence(_prism(1.0, 1.0))

    assert elongated.confidence > square.confidence
    assert elongated.components.anisotropy > square.components.anisotropy
    yaw, dimensions, fallback, state = conservative_orientation(
        elongated_box,
        np.array([3.0, 0.5, 1.0]),
        elongated.confidence,
    )
    assert not fallback
    assert state in {'supported', 'uncertain'}
    assert yaw == elongated_box.yaw_rad
    np.testing.assert_allclose(dimensions, elongated_box.dimensions_xyz)


def test_low_confidence_uses_map_axis_aabb_without_false_yaw() -> None:
    box, _ = _confidence(_prism(1.0, 1.0))

    yaw, dimensions, fallback, state = conservative_orientation(
        box,
        np.array([1.2, 1.1, 1.0]),
        0.399,
    )

    assert yaw == 0.0
    assert fallback
    assert state == 'aabb_fallback'
    np.testing.assert_allclose(dimensions, [1.2, 1.1, 1.0])
