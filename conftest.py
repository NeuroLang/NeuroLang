import pytest


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
    from neurolang.utils.relational_algebra_set.dask_helpers import DaskContextManager
    DaskContextManager._context = None