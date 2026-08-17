"""Tests for the shared colour vocabulary and split audit."""

from collections import Counter
import csv
import json

import pytest

from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.common.colour_vocabulary import normalize_released_colour_name
from qmapnav.language.extraction import extract_language_features


_MISSING_VALUES = frozenset({'', '_', '-1', 'n/a', 'none', 'null'})


def _audit_colour_data(vla_root, split_path):
    """Collect the split counts needed by the vocabulary regression."""
    split = json.loads(split_path.read_text(encoding='utf-8'))
    fit = set(split['fit_scenes'])
    held_out = set(split['held_out_scenes'])
    if fit & held_out:
        raise ValueError('fit and held-out scene sets overlap')
    paths = sorted(vla_root.glob('*/*_object_result.csv'))
    if fit | held_out != {path.parent.name for path in paths}:
        raise ValueError('split does not cover the released scene set exactly')
    canonical_total = Counter()
    canonical_fit = Counter()
    canonical_held_out = Counter()
    for path in paths:
        canonical = Counter()
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                for index in (1, 2, 3):
                    raw = str(row.get(f'object_color_scheme{index}', '')).strip()
                    if raw.casefold() in _MISSING_VALUES:
                        continue
                    label = normalize_colour_name(
                        normalize_released_colour_name(raw)
                    )
                    canonical[label] += 1
        canonical_total.update(canonical)
        if path.parent.name in fit:
            canonical_fit.update(canonical)
        else:
            canonical_held_out.update(canonical)
    missing_fit = [
        colour for colour in canonical_total if canonical_fit[colour] == 0
    ]
    if missing_fit:
        raise ValueError(f'canonical colours absent from fit: {missing_fit}')
    return {
        'fit_colour_instances': dict(sorted(canonical_fit.items())),
        'held_out_colour_instances': dict(sorted(canonical_held_out.items())),
        'split_overlap': [],
    }


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

    result = _audit_colour_data(root, split)

    assert result['fit_colour_instances'] == {'blue': 1, 'grey': 1}
    assert result['held_out_colour_instances'] == {'grey': 1}
    assert result['split_overlap'] == []
