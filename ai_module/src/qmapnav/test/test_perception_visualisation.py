"""Tests for deterministic perception visualisations."""

import numpy as np

from qmapnav.perception import draw_crop_layout
from qmapnav.perception import eight_view_layout
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerspectiveCropGenerator
from qmapnav.perception.visualisation import draw_crop_detection_contact_sheet


def test_layout_and_empty_contact_sheet_have_expected_shapes() -> None:
    panorama = np.zeros((120, 360, 3), dtype=np.uint8)
    model = PanoramaCameraModel(360, 120)
    views = PerspectiveCropGenerator(
        model,
        eight_view_layout(output_width=64, output_height=64),
    ).generate(panorama, source_image_id='frame')

    layout = draw_crop_layout(panorama, views, model)
    contact_sheet = draw_crop_detection_contact_sheet(
        views,
        tuple(() for _ in views),
        cell_width=32,
    )

    assert layout.shape == panorama.shape
    assert np.any(layout != panorama)
    assert contact_sheet.shape == (64, 128, 3)
