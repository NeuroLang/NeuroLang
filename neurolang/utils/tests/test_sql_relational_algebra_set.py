from neurolang.utils.relational_algebra_set.sql import (
    NamedRelationalAlgebraFrozenSet,
    SQLAEngineFactory,
)
from sqlalchemy import create_engine
from unittest.mock import patch
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sql_engine():
    """yields a SQLAlchemy engine which is suppressed after the test session"""
    engine_ = create_engine("sqlite:///test.db", echo=False)

    with patch.object(SQLAEngineFactory, "get_engine") as _fixture:
        _fixture.return_value = engine_
        yield _fixture

    engine_.dispose()


def test_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]

    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)
    ras_c = NamedRelationalAlgebraFrozenSet(("u", "v"), b)

    res = ras_a.naturaljoin(ras_b)
    res2 = ras_a.naturaljoin(ras_c)
    # res3 = ras_a.naturaljoin(ras_c).naturaljoin(ras_b)
