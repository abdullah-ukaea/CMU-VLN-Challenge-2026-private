"""Protect the runtime/development and transport import boundaries."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parent.parent / 'qmapnav'
RUNTIME_PACKAGES = tuple(
    path for path in sorted(PACKAGE_ROOT.iterdir())
    if path.is_dir() and path.name != 'evaluation'
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module is not None:
                modules.append(node.module)
    return tuple(modules)


def _runtime_imports() -> tuple[tuple[Path, str], ...]:
    return tuple(
        (path, module)
        for package in RUNTIME_PACKAGES
        for path in sorted(package.rglob('*.py'))
        for module in _imports(path)
    )


def test_runtime_packages_do_not_import_evaluation() -> None:
    violations = [
        f'{path}: {module}'
        for path, module in _runtime_imports()
        if module == 'qmapnav.evaluation'
        or module.startswith('qmapnav.evaluation.')
    ]
    assert violations == []


def test_common_does_not_import_other_qmapnav_packages() -> None:
    common = PACKAGE_ROOT / 'common'
    violations = [
        f'{path}: {module}'
        for path in sorted(common.rglob('*.py'))
        for module in _imports(path)
        if module.startswith('qmapnav.')
        and not module.startswith('qmapnav.common')
    ]
    assert violations == []


def test_only_mission_imports_ros_transport() -> None:
    violations = [
        f'{path}: {module}'
        for path, module in _runtime_imports()
        if module == 'rclpy' or module.startswith('rclpy.')
        if path.relative_to(PACKAGE_ROOT).parts[0] != 'mission'
    ]
    assert violations == []
