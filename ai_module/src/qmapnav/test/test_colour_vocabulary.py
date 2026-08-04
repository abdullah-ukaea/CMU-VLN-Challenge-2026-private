"""Tests for the shared Day 8 colour vocabulary and split audit."""

import csv
import json

import pytest

from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.common.colour_vocabulary import normalize_released_colour_name
from qmapnav.language.extraction import extract_language_features
from tools.day8_colour_audit import audit_colour_data


def test_colour_aliases_share_query_vocabulary() -> None:
    assert normalize_colour_name('gray') == 'grey'
    assert normalize_colour_name('light blue') == 'blue'
    assert normalize_colour_name('maroon') == 'red'
    assert normalize_colour_name('navy') == 'blue'
    assert normalize_colour_name('olive') == 'green'
    assert 'white' in COLOUR_CLASSES
    with pytest.raises(ValueError, match='unsupported'):
        normalize_colour_name('chartreuse')


def test_released_vocabulary_preserves_palette_distinctions() -> None:
    assert normalize_released_colour_name('gray') == 'grey'
    assert normalize_released_colour_name('maroon') == 'maroon'
    assert normalize_released_colour_name('aqua') == 'aqua'


def test_language_extraction_consumes_the_shared_query_vocabulary() -> None:
    extraction = extract_language_features(
        'Find the light blue chair beside the maroon table.'
    )

    assert [mention.normalized for mention in extraction.colours] == [
        'blue', 'red'
    ]


def test_audit_rejects_leakage_and_counts_fit_and_holdout(tmp_path) -> None:
    root = tmp_path / 'Unity'
    fieldnames = [
        'object_color_scheme1',
        'object_color_scheme2',
        'object_color_scheme3',
    ]
    for scene, colours in (
        ('fit', ('navy', 'gray')),
        ('held', ('gray',)),
    ):
        directory = root / scene
        directory.mkdir(parents=True)
        with (directory / f'{scene}_object_result.csv').open(
            'w', newline='', encoding='utf-8'
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for colour in colours:
                writer.writerow({
                    'object_color_scheme1': colour,
                    'object_color_scheme2': '_',
                    'object_color_scheme3': '_',
                })
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({
        'fit_scenes': ['fit'],
        'held_out_scenes': ['held'],
    }), encoding='utf-8')

    result = audit_colour_data(root, split)

    assert result['fit_colour_instances'] == {'blue': 1, 'grey': 1}
    assert result['held_out_colour_instances'] == {'grey': 1}
    assert result['split_overlap'] == []
