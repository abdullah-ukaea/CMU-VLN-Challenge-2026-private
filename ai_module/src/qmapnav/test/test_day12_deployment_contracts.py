"""Submission configuration and packaged-asset regression contracts."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def _declared_parameters() -> set[str]:
    tree = ast.parse(
        (PACKAGE_ROOT / 'qmapnav/mission/node.py').read_text(encoding='utf-8')
    )
    return {
        call.args[0].value
        for call in ast.walk(tree)
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'declare_parameter'
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        )
    }


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
