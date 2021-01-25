from ..relational_algebra_set.abstract import (
    RelationalAlgebraColumnInt,
    RelationalAlgebraColumnStr,
    RelationalAlgebraStringExpression,
)
from neurolang.utils.relational_algebra_set.sql import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    SQLAEngineFactory,
)
from sqlalchemy import create_engine, Table, MetaData
from unittest.mock import patch
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sql_engine():
    """yields a SQLAlchemy engine which is suppressed after the test session"""
    engine_ = create_engine("sqlite:///test.db", echo=False)

    with patch.object(SQLAEngineFactory, "_create_engine") as _fixture:
        _fixture.return_value = engine_
        yield _fixture

    # meta = MetaData()
    # meta.reflect(bind=engine_, views=True)
    # meta.drop_all(engine_)
    engine_.dispose()


def get_table_from_engine(table_name):
    return Table(
        table_name,
        MetaData(),
        autoload=True,
        autoload_with=SQLAEngineFactory.get_engine(),
    )


def test_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_b = RelationalAlgebraFrozenSet(b)
    assert ras_a.is_empty() == False
    assert ras_b.is_empty() == False


def test_named_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)

    assert ras_a.is_empty() == False
    assert (4, 8) in ras_a
    assert ras_b.is_empty() == False
    assert (4, 6) in ras_b


def test_try_to_create_index():
    """
    try_to_create_index should not create duplicate indexes.
    """
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)

    ras_a._try_to_create_index(["y", "z"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"y", "z"}

    ras_a._try_to_create_index(["z", "y"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1

    ras_a._try_to_create_index(["y"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 2


def test_try_to_create_index_on_views():
    """
    try_to_create_index should create indexes on parent tables when
    called on views.
    """
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("v", "u"), a)
    ab = ras_a.cross_product(ras_b)
    aba = ab.naturaljoin(ras_a)

    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"y", "z"}

    table = get_table_from_engine(ras_b._table_name)
    assert len(table.indexes) == 0

    abab = aba.naturaljoin(ras_b)
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    table = get_table_from_engine(ras_b._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"u", "v"}


def test_natural_join_creates_view():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)

    res = ras_a.naturaljoin(ras_b)
    assert res._is_view == True


def test_aggregate_with_duplicates():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9), (7, 8, 1)]
    )
    expected_sum = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set

    initial_set2 = NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9), (2, 7, 8, 9)]
    )
    expected_op2 = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "t"), [(7, 8, 13)]
    )
    new_set = initial_set2.aggregate(["x", "y"], {"t": lambda t: sum(t.w + t.z)})
    assert expected_op2 == new_set