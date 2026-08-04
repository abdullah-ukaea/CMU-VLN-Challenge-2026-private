"""Tests for HSV/Lab features, prototypes, and probabilities."""

import cv2
import numpy as np

from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import circular_hue_distance
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_prototypes import fit_colour_prototypes_from_rgb


RGB_PROTOTYPES = {
    'black': [[20, 20, 20], [30, 30, 30], [15, 15, 15]],
    'blue': [[30, 80, 220], [20, 60, 200], [50, 100, 230]],
    'brown': [[110, 65, 30], [130, 75, 35], [95, 55, 25]],
    'green': [[30, 160, 60], [40, 180, 70], [20, 140, 50]],
    'grey': [[110, 110, 110], [130, 130, 130], [150, 150, 150]],
    'orange': [[235, 120, 20], [220, 100, 15], [245, 135, 30]],
    'pink': [[240, 130, 170], [230, 110, 155], [250, 150, 185]],
    'purple': [[125, 50, 170], [145, 60, 190], [105, 40, 150]],
    'red': [[220, 25, 30], [200, 20, 25], [240, 35, 40]],
    'white': [[240, 240, 240], [250, 250, 250], [230, 230, 230]],
    'yellow': [[235, 215, 25], [250, 225, 35], [220, 200, 20]],
}


def _estimate(rgb: tuple[int, int, int]):
    image = np.full((30, 30, 3), rgb, dtype=np.uint8)
    mask = np.ones((30, 30), dtype=np.bool_)
    pixels = filter_reliable_pixels(
        select_object_pixels(image, segmentation_mask=mask)
    )
    features = extract_colour_features(pixels)
    prototypes = fit_colour_prototypes_from_rgb(RGB_PROTOTYPES)
    return classify_colour(features, pixels, prototypes)


def test_blue_and_neutral_grey_prefer_correct_classes() -> None:
    blue = _estimate((30, 80, 220))
    grey = _estimate((125, 125, 125))

    assert blue.dominant_colour == 'blue'
    assert grey.dominant_colour == 'grey'
    assert blue.probabilities['blue'] > blue.probabilities['grey']
    assert grey.probabilities['grey'] > grey.probabilities['blue']


def test_clear_chromatic_and_neutral_samples_prefer_expected_classes() -> None:
    for expected, rgb in (
        ('orange', (235, 120, 20)),
        ('brown', (110, 65, 30)),
        ('black', (20, 20, 20)),
        ('white', (245, 245, 245)),
    ):
        estimate = _estimate(rgb)
        assert estimate.dominant_colour == expected


def test_red_hue_wrap_distance_is_circular() -> None:
    assert circular_hue_distance(0.01, 2.0 * np.pi - 0.01) < 0.03
    first = np.uint8([[[255, 0, 5]]])
    second = np.uint8([[[255, 5, 0]]])
    first_hue = cv2.cvtColor(first, cv2.COLOR_RGB2HSV)[0, 0, 0]
    second_hue = cv2.cvtColor(second, cv2.COLOR_RGB2HSV)[0, 0, 0]
    assert abs(int(first_hue) - int(second_hue)) > 170
    assert _estimate((255, 0, 5)).dominant_colour == 'red'
    assert _estimate((255, 5, 0)).dominant_colour == 'red'


def test_probabilities_are_finite_normalized_and_complete() -> None:
    estimate = _estimate((235, 120, 20))

    assert set(estimate.probabilities) == set(COLOUR_CLASSES)
    assert all(np.isfinite(value) and value >= 0.0
               for value in estimate.probabilities.values())
    np.testing.assert_allclose(
        sum(estimate.probabilities.values()), 1.0, atol=1e-9
    )


def test_highlights_and_shadows_do_not_replace_chromatic_colour() -> None:
    red = np.full((40, 40, 3), [220, 25, 30], dtype=np.uint8)
    red[:5] = [255, 255, 255]
    blue = np.full((40, 40, 3), [30, 80, 220], dtype=np.uint8)
    blue[:5] = [3, 8, 20]
    mask = np.ones((40, 40), dtype=np.bool_)

    estimates = []
    for image in (red, blue):
        pixels = filter_reliable_pixels(
            select_object_pixels(image, segmentation_mask=mask)
        )
        estimates.append(classify_colour(
            extract_colour_features(pixels), pixels,
            fit_colour_prototypes_from_rgb(RGB_PROTOTYPES),
        ))

    assert estimates[0].dominant_colour == 'red'
    assert estimates[1].dominant_colour == 'blue'


def test_pairwise_colour_ambiguity_lowers_confidence() -> None:
    clear = _estimate((30, 80, 220))
    for first, second in (
        ('brown', 'orange'), ('pink', 'red'), ('purple', 'blue')
    ):
        prototypes = dict(RGB_PROTOTYPES)
        prototypes[second] = prototypes[first]
        image = np.full(
            (30, 30, 3), prototypes[first][0], dtype=np.uint8
        )
        mask = np.ones((30, 30), dtype=np.bool_)
        pixels = filter_reliable_pixels(
            select_object_pixels(image, segmentation_mask=mask)
        )
        ambiguous = classify_colour(
            extract_colour_features(pixels), pixels,
            fit_colour_prototypes_from_rgb(prototypes),
        )

        assert ambiguous.status == 'ambiguous'
        assert ambiguous.confidence < clear.confidence
        assert {
            name for name, _ in sorted(
                ambiguous.probabilities.items(), key=lambda item: -item[1]
            )[:2]
        } == {first, second}


def test_too_few_pixels_returns_explicit_failure() -> None:
    image = np.full((4, 4, 3), [30, 80, 220], dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.bool_)
    pixels = filter_reliable_pixels(
        select_object_pixels(image, segmentation_mask=mask)
    )
    estimate = classify_colour(
        extract_colour_features(pixels), pixels,
        fit_colour_prototypes_from_rgb(RGB_PROTOTYPES),
    )

    assert estimate.status == 'too_few_pixels'
    assert estimate.probabilities == {}
    assert estimate.dominant_colour is None
