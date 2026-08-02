"""Run the ROS 2 Python style checks as part of the package test suite."""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    return_code, errors = main_with_errors(argv=[])
    assert return_code == 0, '\n'.join(errors)
