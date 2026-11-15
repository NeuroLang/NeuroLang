"""Pytest configuration file."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-squall",
        action="store_true",
        default=False,
        help="run the SQUALL tests, skipped by default",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "squall: test of the SQUALL syntax, skipped without --run-squall",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_squall = pytest.mark.skip(
        reason="SQUALL is to be deprecated: need --run-squall option to run"
    )
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "squall" in item.keywords and not config.getoption("--run-squall"):
            item.add_marker(skip_squall)
