"""Save representative Day 8 colour and relation diagnostic artifacts."""

import argparse
from pathlib import Path

import numpy as np

from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_prototypes import load_colour_prototypes
from qmapnav.reasoning.colour_visualisation import save_colour_diagnostic
from qmapnav.reasoning.relation_visualisation import save_relation_diagnostic
from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.support_relations import on_evidence


def _geometry(entity_id, semantic_class, centre, dimensions):
    centre = np.asarray(centre, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    half = dimensions[:2] / 2.0
    footprint = np.asarray([
        centre[:2] + [-half[0], -half[1]],
        centre[:2] + [half[0], -half[1]],
        centre[:2] + [half[0], half[1]],
        centre[:2] + [-half[0], half[1]],
    ])
    return SupportGeometry(
        entity_id, semantic_class, centre, dimensions, 0.0, footprint,
        centre[2] - dimensions[2] / 2.0,
        centre[2] + dimensions[2] / 2.0,
        0.95, 'active', 'object',
    )


def main() -> None:
    """Save deterministic blue-chair and book-on-table diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prototype_path', type=Path)
    parser.add_argument('output_directory', type=Path)
    arguments = parser.parse_args()
    crop = np.full((120, 160, 3), [30, 80, 210], dtype=np.uint8)
    crop[:12] = [3, 8, 20]
    crop[-12:] = [255, 255, 255]
    mask = np.zeros(crop.shape[:2], dtype=np.bool_)
    mask[5:-5, 10:-10] = True
    selection = select_object_pixels(crop, segmentation_mask=mask)
    pixels = filter_reliable_pixels(selection)
    features = extract_colour_features(pixels)
    estimate = classify_colour(
        features,
        pixels,
        load_colour_prototypes(arguments.prototype_path),
        source_viewpoint_id='representative_view_a',
        source_detection_id='blue_chair',
    )
    save_colour_diagnostic(
        arguments.output_directory, 'blue_chair', selection, pixels,
        features, estimate, expected_colour='blue',
    )
    table = _geometry('table_1', 'table', (0, 0, 0.4), (1.5, 1, 0.8))
    book = _geometry('book_1', 'book', (0.1, 0, 0.87), (0.3, 0.2, 0.1))
    save_relation_diagnostic(
        arguments.output_directory, 'book_on_table', book, table,
        on_evidence(book, table),
    )


if __name__ == '__main__':
    main()
