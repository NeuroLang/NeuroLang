from neurolang.utils.relational_algebra_set import (
    NamedSQLARelationalAlgebraFrozenSet,
)
from sqlalchemy import create_engine
import pytest


@pytest.fixture(scope="session")
def engine(request):
    """yields a SQLAlchemy engine which is suppressed after the test session"""
    engine_ = create_engine("sqlite:///test.db", echo=True)

    yield engine_

    engine_.dispose()


def test_set_init(engine):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]

    ras_a = NamedSQLARelationalAlgebraFrozenSet(engine, ("z", "y"), a)
    ras_b = NamedSQLARelationalAlgebraFrozenSet(engine, ("y", "x"), b)
    ras_c = NamedSQLARelationalAlgebraFrozenSet(engine, ("u", "v"), b)

    res = ras_a.naturaljoin(ras_b)
    res2 = ras_a.naturaljoin(ras_c)
    res3 = ras_a.naturaljoin(ras_c).naturaljoin(ras_b)
