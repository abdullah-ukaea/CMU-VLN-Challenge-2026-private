"""Tests for overlapping perspective crop generation."""

from math import pi

import numpy as np
import pytest

from qmapnav.perception.crop_generator import eight_view_layout
from qmapnav.perception.crop_generator import PerspectiveCropGenerator
from qmapnav.perception.crop_generator import PerspectiveCropLayout
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def test_initial_layout_has_eight_views_and_25_percent_overlap() -> None:
    layout = eight_view_layout(output_width=32, output_height=24)

    assert layout.crop_count == 8
    assert layout.horizontal_overlap_fraction == pytest.approx(0.25)
    assert layout.yaw_centres_rad == pytest.approx(
        (0.0, pi / 4.0, pi / 2.0, 3.0 * pi / 4.0, -pi, -3.0 * pi / 4.0,
         -pi / 2.0, -pi / 4.0)
    )


def test_generator_produces_deterministic_views_and_retains_geometry() -> None:
    model = PanoramaCameraModel(360, 120)
    layout = eight_view_layout(output_width=32, output_height=24)
    generator = PerspectiveCropGenerator(model, layout)
    panorama = np.full((120, 360, 3), (7, 11, 13), dtype=np.uint8)

    first = generator.generate(panorama, source_image_id='frame_1')
    second = generator.generate(panorama, source_image_id='frame_1')

    assert len(first) == 8
    assert tuple(view.geometry.crop_id for view in first) == tuple(range(8))
    assert all(view.image_rgb.shape == (24, 32, 3) for view in first)
    assert all(np.all(view.valid_mask) for view in first)
    assert all(np.all(view.image_rgb == (7, 11, 13)) for view in first)
    assert all(np.array_equal(a.image_rgb, b.image_rgb) for a, b in zip(first, second))
    assert first[0].geometry is second[0].geometry
    assert not first[0].image_rgb.flags.writeable
    assert not first[0].valid_mask.flags.writeable


def test_horizontal_gradient_places_crop_centres_at_expected_panorama_yaws() -> None:
    model = PanoramaCameraModel(360, 120)
    layout = eight_view_layout(output_width=5, output_height=5)
    generator = PerspectiveCropGenerator(model, layout)
    panorama = np.zeros((120, 360, 3), dtype=np.float64)
    panorama[..., 0] = np.arange(360, dtype=np.float64)

    views = generator.generate(panorama, source_image_id='gradient')
    centre_values = [view.image_rgb[2, 2, 0] for view in views]

    assert centre_values == pytest.approx(
        (179.5, 134.5, 89.5, 44.5, 179.5, 314.5, 269.5, 224.5),
        abs=1e-9,
    )


def test_configurable_pitch_rows_have_stable_pitch_major_crop_ids() -> None:
    model = PanoramaCameraModel(360, 120)
    layout = PerspectiveCropLayout(
        yaw_centres_rad=(0.0, pi / 2.0),
        pitch_centres_rad=(-0.2, 0.2),
        horizontal_fov_rad=pi / 3.0,
        vertical_fov_rad=pi / 3.0,
        output_width=8,
        output_height=8,
    )
    generator = PerspectiveCropGenerator(model, layout)

    geometries = generator.geometries()

    assert [item.crop_id for item in geometries] == [0, 1, 2, 3]
    assert [item.pitch_rad for item in geometries] == pytest.approx(
        [-0.2, -0.2, 0.2, 0.2]
    )
    assert [item.yaw_rad for item in geometries] == pytest.approx(
        [0.0, pi / 2.0, 0.0, pi / 2.0]
    )


def test_pixels_outside_panorama_vertical_span_are_masked() -> None:
    model = PanoramaCameraModel(360, 120)
    layout = PerspectiveCropLayout(
        yaw_centres_rad=(0.0,),
        pitch_centres_rad=(0.8,),
        horizontal_fov_rad=pi / 3.0,
        vertical_fov_rad=pi / 2.0,
        output_width=16,
        output_height=16,
    )
    panorama = np.full((120, 360, 3), 255, dtype=np.uint8)

    view = PerspectiveCropGenerator(model, layout).generate(
        panorama,
        source_image_id='high_pitch',
    )[0]

    assert np.any(view.valid_mask)
    assert np.any(~view.valid_mask)
    assert np.all(view.image_rgb[~view.valid_mask] == 0)


def test_generator_rejects_wrong_panorama_shape() -> None:
    generator = PerspectiveCropGenerator(
        PanoramaCameraModel(360, 120),
        eight_view_layout(output_width=16, output_height=16),
    )

    with pytest.raises(ValueError, match='shape'):
        generator.generate(
            np.zeros((120, 359, 3), dtype=np.uint8),
            source_image_id='bad',
        )
