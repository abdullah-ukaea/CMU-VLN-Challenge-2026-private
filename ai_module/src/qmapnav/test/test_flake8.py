"""Run the ROS 2 Python style checks as part of the package test suite."""

import warnings

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    # The ROS integration module may leave middleware threads alive until
    # process exit. Ignore only Python's fork advisory from flake8 workers.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'This process .* is multi-threaded, use of fork.*',
        )
        return_code, errors = main_with_errors(argv=[])
    assert return_code == 0, '\n'.join(errors)
