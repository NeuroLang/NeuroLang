"""Pytest configuration file."""
from pathlib import Path

import pytest

# Directories that a test leaves in the working directory when it writes to
# a relative path: a mock used as a path (``MagicMock/...``) or a data
# directory that is "unused". They are ignored by git, but a test that makes
# one is wrong: its files would be read by the next run.
STRAY_DIRECTORIES = ("MagicMock", "unused")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def no_stray_directories(request):
    """Fail the session whose tests leave a stray directory in the root."""
    root = Path(request.config.rootpath)
    before = {name for name in STRAY_DIRECTORIES if (root / name).exists()}
    yield
    leaked = [
        name for name in STRAY_DIRECTORIES
        if (root / name).exists() and name not in before
    ]
    assert not leaked, (
        f"The tests left {leaked} in {root}: a test wrote to a relative "
        "path, or to a mock that is a path."
    )
