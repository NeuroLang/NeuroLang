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

    print(ras_a)
    print(ras_b)
    print(ras_c)


if __name__ == "__main__":
    engine = create_engine("sqlite:///test.db", echo=True)
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]

    ra = NamedSQLARelationalAlgebraFrozenSet(engine, ("z", "y"), a)
    rb = NamedSQLARelationalAlgebraFrozenSet(engine, ("y", "x"), b)
    rc = NamedSQLARelationalAlgebraFrozenSet(engine, ("u", "v"), b)
    print(ra)
    print(rb)
    print(rc)
    res = ra.naturaljoin(rb)
    res2 = ra.naturaljoin(rc)
    print(res)
    print(res2)
    res3 = ra.naturaljoin(rc).naturaljoin(rb)
    print(res3)
