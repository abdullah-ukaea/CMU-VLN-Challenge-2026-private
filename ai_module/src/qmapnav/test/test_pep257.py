"""Run the ROS 2 docstring checks as part of the package test suite."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257() -> None:
    assert main(argv=['.']) == 0
