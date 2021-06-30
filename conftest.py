"""
Pytest configuration file.

NOTE: Do not remove the unused dask_sql import !

The dask_sql library uses jpype to start a JVM and access Java objects
from python. This JVM needs to be started before we import neurolang,
otherwise it can cause major side-effects which are hard to track. See
https://github.com/jpype-project/jpype/issues/933 for reference.
"""
import pytest
import dask_sql
from neurolang import config
from neurolang.utils.relational_algebra_set.dask_helpers import (
    DaskContextManager,
)


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


def pytest_sessionstart(session: pytest.Session):
    """
    Hook called after the pytest Session object has been created and
    before performing collection and entering the run test loop.

    The dask-sql library uses the jpype library which starts a JVM and allows
    us to use Java classes from Python. But the JVM will trigger a
    segmentation fault when starting and when interrupting threads and Pythons
    fault handler can intercept these operations and interpret these as
    real faults. So we need to disable faulthandlers which pytest starts
    otherwise we get segmentation faults when running the tests.
    See (https://jpype.readthedocs.io/en/latest/userguide.html#errors-reported-by-python-fault-handler)
    """
    try:
        import faulthandler

        faulthandler.enable()
        faulthandler.disable()
    except:
        pass


@pytest.fixture(autouse=config["RAS"].get("Backend", "pandas") == "dask")
def clear_dask_context_after_test_module():
    """
    We use only one DaskContextManager for the application and its context gets
    clustered with objects quite fast when running the tests, so this fixture clears
    the context after each test function.
    """
    yield 0

    DaskContextManager._context = None
