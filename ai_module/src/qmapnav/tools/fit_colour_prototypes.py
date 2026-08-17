"""Fit colour prototypes from the fixed development split."""

import argparse
import json
from pathlib import Path

from qmapnav.reasoning.colour_prototypes import fit_colour_prototypes
from qmapnav.reasoning.colour_prototypes import prototypes_to_json


def main() -> None:
    """Fit prototypes and print stable JSON for persistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vla_root', type=Path)
    parser.add_argument('split_path', type=Path)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / 'data/colour_prototypes.json',
    )
    arguments = parser.parse_args()
    split = json.loads(arguments.split_path.read_text(encoding='utf-8'))
    prototypes = fit_colour_prototypes(arguments.vla_root, split['fit_scenes'])
    payload = prototypes_to_json(prototypes)
    payload['fit_scenes'] = sorted(split['fit_scenes'])
    payload['white_policy'] = (
        'brightest 5% tail of released fit-scene grey metadata, capped at '
        'CIELAB L*=82; query white remains distinct from grey'
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(f'Wrote {arguments.output}')


if __name__ == '__main__':
    main()
