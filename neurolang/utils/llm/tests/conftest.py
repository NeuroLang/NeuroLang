"""
Fixtures of the LLM assistant tests.

The tests are skipped when the ``llm`` extra (httpx) is not installed.
"""

import importlib.util

import pytest

if importlib.util.find_spec("httpx") is None:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture
def nl():
    """A small engine: studies and the terms they mention."""
    from neurolang.frontend import NeurolangPDL

    engine = NeurolangPDL()
    engine.add_tuple_set([(1,), (2,), (3,)], name="study")
    engine.add_tuple_set(
        [(1, "pain"), (2, "chronic pain"), (3, "memory"), (1, "fear")],
        name="mentions",
    )
    return engine
