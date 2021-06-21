import pytest


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


@pytest.fixture(autouse=True)
def clear_dask_context_after_test_module():
    """
    We use only one DaskContextManager for the application and its context gets
    clustered with objects quite fast when running the tests, so this fixture clears
    the context after each test function.
    """
    yield 0
    # For some unknown reason importing DaskContextManager at the top of the
    # file creates an error when running the tests so we import it here instead.
    from neurolang.utils.relational_algebra_set.dask_helpers import (
        DaskContextManager,
    )

    DaskContextManager._context = None