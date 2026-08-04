"""Audit released colour labels and validate the scene-disjoint Day 8 split."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.common.colour_vocabulary import normalize_released_colour_name


MISSING_VALUES = frozenset({'', '_', '-1', 'n/a', 'none', 'null'})


def audit_colour_data(
    vla_root: Path,
    split_path: Path,
    evidence_root: Path | None = None,
) -> dict[str, object]:
    """Return deterministic inventory, split, and saved-evidence diagnostics."""
    split = json.loads(split_path.read_text(encoding='utf-8'))
    fit = set(split['fit_scenes'])
    held_out = set(split['held_out_scenes'])
    if fit & held_out:
        raise ValueError('fit and held-out scene sets overlap')
    per_scene = {}
    released_total = Counter()
    canonical_total = Counter()
    canonical_fit = Counter()
    canonical_held_out = Counter()
    object_counts = {}
    paths = sorted(vla_root.glob('*/*_object_result.csv'))
    available_scenes = {path.parent.name for path in paths}
    if fit | held_out != available_scenes:
        raise ValueError('split does not cover the released scene set exactly')
    for path in paths:
        scene = path.parent.name
        released = Counter()
        canonical = Counter()
        object_count = 0
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                object_count += 1
                for index in (1, 2, 3):
                    raw = str(row.get(
                        f'object_color_scheme{index}', ''
                    )).strip()
                    if raw.casefold() in MISSING_VALUES:
                        continue
                    released_name = normalize_released_colour_name(raw)
                    canonical_name = normalize_colour_name(released_name)
                    released[released_name] += 1
                    canonical[canonical_name] += 1
        released_total.update(released)
        canonical_total.update(canonical)
        if scene in fit:
            canonical_fit.update(canonical)
        else:
            canonical_held_out.update(canonical)
        object_counts[scene] = object_count
        per_scene[scene] = {
            'object_count': object_count,
            'released_colour_instances': dict(sorted(released.items())),
            'canonical_colour_instances': dict(sorted(canonical.items())),
        }
    missing_fit = sorted(
        colour for colour in canonical_total if canonical_fit[colour] == 0
    )
    if missing_fit:
        raise ValueError(f'canonical colours absent from fit: {missing_fit}')
    evidence = _audit_saved_evidence(evidence_root)
    return {
        'schema_version': 1,
        'released_scene_count': len(paths),
        'released_object_count': sum(object_counts.values()),
        'released_colour_instances': dict(sorted(released_total.items())),
        'canonical_colour_instances': dict(sorted(canonical_total.items())),
        'fit_colour_instances': dict(sorted(canonical_fit.items())),
        'held_out_colour_instances': dict(sorted(canonical_held_out.items())),
        'fit_scenes': sorted(fit),
        'held_out_scenes': sorted(held_out),
        'split_overlap': [],
        'per_scene': per_scene,
        'saved_evidence': evidence,
    }


def _audit_saved_evidence(root: Path | None) -> dict[str, int]:
    if root is None or not root.exists():
        return {}
    panoramas = list(root.glob('*/*/panorama.png'))
    detections = list(root.glob('*/*/detections.json'))
    inputs = list(root.glob('*/*/inputs.npz'))
    return {
        'panorama_count': len(panoramas),
        'detection_manifest_count': len(detections),
        'projected_support_archive_count': len(inputs),
    }


def main() -> None:
    """Run the command-line audit and print stable JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vla_root', type=Path)
    parser.add_argument('split_path', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    arguments = parser.parse_args()
    payload = audit_colour_data(
        arguments.vla_root,
        arguments.split_path,
        arguments.evidence_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
