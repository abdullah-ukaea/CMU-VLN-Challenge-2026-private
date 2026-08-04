"""Tests for spatial and photometric Day 8 pixel filtering."""

import numpy as np

from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels


def test_mask_path_keeps_central_component_and_erodes_boundary() -> None:
    image = np.full((30, 40, 3), [20, 80, 220], dtype=np.uint8)
    mask = np.zeros((30, 40), dtype=np.bool_)
    mask[5:25, 10:30] = True
    mask[0:3, 0:3] = True

    selected = select_object_pixels(image, segmentation_mask=mask)

    assert selected.source == 'segmentation_mask'
    assert not selected.selected_mask[1, 1]
    assert selected.stage_counts['mask_eroded'] < 400
    assert selected.stage_counts['selected'] >= 50


def test_geometry_support_precedes_contracted_box() -> None:
    image = np.full((30, 40, 3), [180, 70, 20], dtype=np.uint8)
    points = np.array([[19.0, 15.0], [20.0, 15.0], [21.0, 15.0]])

    selected = select_object_pixels(image, geometry_support_uv=points)

    assert selected.source == 'geometry_support'
    assert selected.selected_mask[15, 20]
    assert not selected.selected_mask[0, 0]


def test_uniform_loose_box_is_flagged_as_contaminated() -> None:
    image = np.full((30, 40, 3), [120, 120, 120], dtype=np.uint8)

    selected = select_object_pixels(image)

    assert selected.status == 'mask_contaminated'
    assert selected.contamination_score > 0.95


def test_shadow_and_highlight_tails_are_downweighted_not_dominant() -> None:
    image = np.full((30, 40, 3), [20, 70, 220], dtype=np.uint8)
    image[:3] = [2, 5, 15]
    image[-3:] = [255, 255, 255]
    mask = np.ones((30, 40), dtype=np.bool_)

    reliable = filter_reliable_pixels(
        select_object_pixels(image, segmentation_mask=mask)
    )

    assert reliable.stage_counts['shadow_downweighted'] > 0
    assert reliable.stage_counts['highlight_downweighted'] > 0
    assert np.median(reliable.hsv[:, 0]) > 200.0


def test_consistently_black_pixels_survive_shadow_filtering() -> None:
    image = np.full((30, 40, 3), 18, dtype=np.uint8)
    mask = np.ones((30, 40), dtype=np.bool_)

    reliable = filter_reliable_pixels(
        select_object_pixels(image, segmentation_mask=mask)
    )

    assert reliable.stage_counts['shadow_downweighted'] == 0
    assert reliable.rgb.shape[0] == 1200
    assert reliable.status == 'underexposed'
