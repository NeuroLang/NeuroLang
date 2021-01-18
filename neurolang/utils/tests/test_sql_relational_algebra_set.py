from neurolang.utils.relational_algebra_set.sql import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    RelationalAlgebraSet,
    SQLAEngineFactory,
    RelationalAlgebraStringExpression
)
from sqlalchemy import create_engine, Table, MetaData
from unittest.mock import patch
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sql_engine():
    """yields a SQLAlchemy engine which is suppressed after the test session"""
    engine_ = create_engine("sqlite:///test.db", echo=False)

    with patch.object(SQLAEngineFactory, "get_engine") as _fixture:
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
    ras_c = NamedRelationalAlgebraFrozenSet(("u", "v"), b)

    res = ras_a.naturaljoin(ras_b)
    assert res._is_view == True


def test_rename_column():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ras_a.rename_column("y", "z")
    assert all(
        el_a.x == el_b.x and el_a.y == el_b.z
        for el_a, el_b in zip(ras_a, ras_b)
    )


def test_iter_and_fetch_one():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    ras_a = RelationalAlgebraFrozenSet(a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res

    res_dee = RelationalAlgebraFrozenSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()


def test_named_iter_and_fetch_one():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res
    print(ras_a.fetch_one())

    res_dee = NamedRelationalAlgebraFrozenSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()

def test_relational_algebra_ra_projection():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = RelationalAlgebraFrozenSet(a)

    ras_0 = ras.projection(0)
    assert (0,) in ras_0 and (1,) in ras_0
    assert len(ras_0) == 2
    assert RelationalAlgebraFrozenSet.dum().projection(0).is_empty()
    assert RelationalAlgebraFrozenSet.dee().projection().is_dee()

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))
    assert ras.projection() == RelationalAlgebraFrozenSet.dee()

def test_relational_algebra_ra_selection():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = RelationalAlgebraSet(a)

    # Select elements where col0 == 1
    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    # Select elements where col0 == 1 and col1 == 2. Result should be empty.
    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set()
    assert ras_0 == a_sel

    # Select elements where the first parameter is 0
    # ras_0 = ras.selection(lambda x: x[0] == 0)
    # assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 0)

    # # Select elements where the col1 has values that are odd
    # ras_0 = ras.selection({1: lambda x: x % 2 == 1})
    # assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)

    # # Select elements where the col0 is 1 and col2 > 2
    # ras_0 = ras.selection({0: 1, 2: lambda x: x > 2})
    # assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i > 1)

    assert RelationalAlgebraSet.dum().selection({0: 1}).is_empty()


def test_named_relational_algebra_ra_selection():
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = NamedRelationalAlgebraFrozenSet(("x", "y", "z"), a)

    ras_0 = ras.selection({"x": 1})
    print(ras_0)
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns, set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    )
    print(a_sel)
    assert ras_0 == a_sel

    ras_0 = ras.selection({"x": 1, "y": 2})
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2),
    )
    assert ras_0 == a_sel

    # ras_0 = ras.selection({"x": lambda x: x == 1, "y": lambda y: y == 2})
    # assert ras_0 == a_sel

    # ras_0 = ras.selection(lambda t: t.x == 1 and t.y == 2)
    # assert ras_0 == a_sel

    ras_0 = ras.selection(
        RelationalAlgebraStringExpression("x == 1 and y == 2")
    )
    assert ras_0 == a_sel