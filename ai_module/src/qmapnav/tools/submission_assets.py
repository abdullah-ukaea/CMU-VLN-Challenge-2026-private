#!/usr/bin/env python3
"""Inventory submission assets and prove that each is locally loadable."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def _digest(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inspect_asset(name: str, path: Path, loader=None) -> dict[str, object]:
    """Return existence, checksum, size, and optional load evidence."""
    path = path.expanduser().resolve()
    record = {
        'name': name,
        'path': str(path),
        'exists': path.is_file(),
        'size_bytes': path.stat().st_size if path.is_file() else None,
        'sha256': _digest(path) if path.is_file() else None,
        'offline_load': 'not_run',
    }
    if path.is_file() and loader is not None:
        try:
            loader(path)
        except Exception as error:
            record['offline_load'] = f'failed:{type(error).__name__}:{error}'
        else:
            record['offline_load'] = 'passed'
    return record


def _load_yolo(path: Path) -> None:
    from qmapnav.perception import YOLOEDetector

    YOLOEDetector(path, device='cpu', half_precision=False)


def _load_mobileclip(path: Path) -> None:
    import torch

    torch.jit.load(str(path), map_location='cpu')


def _load_json(path: Path) -> None:
    json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    """Write a fail-closed inventory for every required submission asset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--model-root', type=Path, default=Path('/home/docker/models')
    )
    arguments = parser.parse_args()
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory('qmapnav'))
    assets = (
        inspect_asset(
            'yoloe_checkpoint',
            arguments.model_root / 'yoloe-11s-seg.pt',
            _load_yolo,
        ),
        inspect_asset(
            'mobileclip_prompt_encoder',
            arguments.model_root / 'mobileclip_blt.ts',
            _load_mobileclip,
        ),
        inspect_asset(
            'colour_prototypes',
            share / 'data' / 'colour_prototypes.json',
            _load_json,
        ),
        inspect_asset(
            'submission_config',
            share / 'configs' / 'submission_v1.yaml',
        ),
    )
    failures = [
        item['name'] for item in assets
        if not item['exists'] or str(item['offline_load']).startswith('failed')
    ]
    report = {'passed': not failures, 'failures': failures, 'assets': assets}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
