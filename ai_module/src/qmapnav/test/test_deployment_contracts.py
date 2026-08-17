"""Submission configuration and packaged-asset regression contracts."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def _declared_parameters() -> set[str]:
    tree = ast.parse(
        (PACKAGE_ROOT / 'qmapnav/mission/runtime_config.py').read_text(
            encoding='utf-8'
        )
    )
    for statement in ast.walk(tree):
        if not (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == 'declarations'
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Tuple)
        ):
            continue
        return {
            item.elts[0].value
            for item in statement.value.elts
            if (
                isinstance(item, ast.Tuple)
                and item.elts
                and isinstance(item.elts[0], ast.Constant)
                and isinstance(item.elts[0].value, str)
            )
        }
    raise AssertionError('runtime parameter declarations are missing')


def test_submission_config_freezes_every_runtime_parameter() -> None:
    payload = yaml.safe_load(
        (PACKAGE_ROOT / 'configs/submission_v1.yaml').read_text(
            encoding='utf-8'
        )
    )
    configured = set(payload['qmapnav']['ros__parameters'])
    assert configured == _declared_parameters()


def test_launch_uses_only_the_frozen_submission_config() -> None:
    source = (PACKAGE_ROOT / 'launch/qmapnav.launch.py').read_text(
        encoding='utf-8'
    )
    assert "'submission_v1.yaml'" in source
    assert 'parameters=[str(config)]' in source


def test_dockerfile_packages_both_detector_assets() -> None:
    dockerfile = next(
        (
            parent / 'docker/Dockerfile'
            for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents)
            if (parent / 'docker/Dockerfile').is_file()
        ),
        None,
    )
    if dockerfile is not None:
        source = dockerfile.read_text(encoding='utf-8')
        assert "attempt_download_asset('yoloe-11s-seg.pt')" in source
        assert "attempt_download_asset('mobileclip_blt.ts')" in source
        return
    model_root = Path('/home/docker/models')
    assert (model_root / 'yoloe-11s-seg.pt').is_file()
    assert (model_root / 'mobileclip_blt.ts').is_file()
