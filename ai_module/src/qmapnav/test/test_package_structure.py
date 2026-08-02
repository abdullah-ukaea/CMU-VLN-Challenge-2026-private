"""Protect the frozen top-level module boundaries."""

from importlib import import_module

import pytest


SUBSYSTEM_MODULES = (
    'qmapnav.common',
    'qmapnav.language',
    'qmapnav.perception',
    'qmapnav.mapping',
    'qmapnav.reasoning',
    'qmapnav.navigation',
    'qmapnav.mission',
    'qmapnav.evaluation',
)


@pytest.mark.parametrize('module_name', SUBSYSTEM_MODULES)
def test_frozen_subsystem_module_is_importable(module_name: str) -> None:
    assert import_module(module_name).__name__ == module_name
