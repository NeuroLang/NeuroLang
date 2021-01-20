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

def test_relational_algebra_ra_equijoin():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, i, i * 2) for i in range(5)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)
    ras_d = RelationalAlgebraSet(d)
    ras_empty = ras_d.selection({0: 1000})
    dee = RelationalAlgebraSet.dee()
    dum = RelationalAlgebraSet.dum()

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c

    res = ras_a.equijoin(ras_a, [(0, 0)])
    assert res == ras_d

    res = ras_a.equijoin(dee, [(0, 0)])
    assert res == ras_a

    res = ras_a.equijoin(dum, [(0, 0)])
    assert res == dum

    res = ras_a.equijoin(ras_empty, [(0, 0)])
    assert res.is_empty()


def test_relational_algebra_ra_cross_product():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)
    ras_empty = ras_a.selection({0: 1000})
    dee = RelationalAlgebraSet.dee()
    dum = RelationalAlgebraSet.dum()

    res = ras_a.cross_product(ras_b)
    assert res == ras_c

    res = ras_a.cross_product(dee)
    assert res == ras_a

    res = ras_a.cross_product(dum)
    assert res == dum

    res = ras_a.cross_product(ras_empty)
    assert res.is_empty()

    res = ras_empty.cross_product(ras_a)
    assert res.is_empty()


def test_relational_algebra_ra_equijoin_mixed_types():
    a = [(chr(ord("a") + i), i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(chr(ord("a") + i), i * 2, i * 2, i * 3) for i in range(5)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c

def test_named_relational_algebra_ra_naturaljoin():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, j * 2, j * 3) for i in range(5) for j in range(5)]

    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)
    ras_b2 = NamedRelationalAlgebraFrozenSet(("u", "v"), b)
    ras_c = NamedRelationalAlgebraFrozenSet(("z", "y", "x"), c)
    ras_d = NamedRelationalAlgebraFrozenSet(("z", "y", "u", "v"), d)
    empty = NamedRelationalAlgebraFrozenSet(("z", "y"), [])
    dee = NamedRelationalAlgebraFrozenSet.dee()
    dum = NamedRelationalAlgebraFrozenSet.dum()

    assert len(ras_a.naturaljoin(empty)) == 0
    assert len(empty.naturaljoin(ras_a)) == 0
    assert ras_a.naturaljoin(dee) == ras_a
    assert dee.naturaljoin(ras_a) == ras_a
    assert ras_a.naturaljoin(dum) == dum
    assert dum.naturaljoin(ras_a) == dum

    res = ras_a.naturaljoin(ras_b)
    assert res == ras_c

    res = ras_a.naturaljoin(ras_a)
    assert res == ras_a

    res = ras_a.naturaljoin(ras_b2)
    assert res == ras_d

def test_relational_algebra_set_semantics_empty():
    ras = RelationalAlgebraSet()

    assert len(ras) == 0
    assert ras.is_empty()
    assert ras.arity == 0
    assert 0 not in ras
    assert list(iter(ras)) == []
    assert ras == RelationalAlgebraSet.dum()

    ras.add((0, 1))
    assert (0, 1) in ras
    assert len(ras) == 1
    assert ras.arity == 2


def test_relational_algebra_set_semantics():
    a = [5, 4, 3, 2, 3, 1]
    ras = RelationalAlgebraSet(a)
    ras_ = RelationalAlgebraSet(a)
    ras__ = set((e,) for e in a)

    # assert ras.columns == [0]

    assert ras == ras_
    assert ras == ras__

    assert len(ras) == len(a) - 1
    ras.discard(5)
    assert 5 not in ras
    assert len(ras) == len(a) - 2
    ras.add(10)
    assert len(ras) == len(a) - 1
    assert 10 in ras
    assert [10] in ras
    assert (5, 2) not in ras
    assert all(a_ in ras for a_ in a if a_ != 5)
    assert ras.fetch_one() in ras__

    dee = RelationalAlgebraSet.dee()
    dum = RelationalAlgebraSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0

    r = RelationalAlgebraSet.create_view_from(ras)
    assert r == ras
    assert r is not ras

    r = RelationalAlgebraSet(ras)
    assert r == ras
    assert r is not ras

def test_relational_algebra_ra_union():
    first = RelationalAlgebraFrozenSet(
        [(7, 8), (9, 2)]
    )
    second = RelationalAlgebraFrozenSet(
        [(9, 2), (42, 0)]
    )
    assert first | first == first
    expected = RelationalAlgebraFrozenSet(
        [(7, 8), (9, 2), (42, 0)]
    )
    assert first | second == expected
    empty = RelationalAlgebraFrozenSet([])
    dee = RelationalAlgebraFrozenSet.dee()
    dum = RelationalAlgebraFrozenSet.dum()

    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | dum == first
    assert dum | first == first
    assert first | empty | second == first | second

    assert first | set() == first

def test_relational_algebra_ra_union_update():
    first = RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )
    second = RelationalAlgebraSet(
        [(9, 2), (42, 0)]
    )
    f = first.copy()
    f |= first
    assert f == first
    expected = RelationalAlgebraSet(
        [(7, 8), (9, 2), (42, 0)]
    )
    f = first.copy()
    f |= second
    assert f == expected
    empty = RelationalAlgebraSet([])
    dee = RelationalAlgebraSet.dee()
    dum = RelationalAlgebraSet.dum()

    f = first.copy()
    f |= empty
    assert f == first

    e = empty.copy()
    e |= first
    assert e == first

    d = dee.copy()
    d |= dee
    assert d == dee

    f = first.copy()
    f |= dum
    assert f == first

    d = dum.copy()
    d |= first
    assert d == first

    f = first.copy()
    f |= empty
    f |= second
    assert f == first | second

    f = first.copy()
    f |= set()
    assert f == first

def test_named_relational_algebra_difference():
    a = [(i, i * 2) for i in range(5)]
    b = [(i, i * 2) for i in range(1, 5)]
    c = [(i, i * 2) for i in range(1)]

    ras_a = NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("x", "y"), b)
    ras_b_inv = NamedRelationalAlgebraFrozenSet(
        ("y", "x"), [t[::-1] for t in b]
    )
    ras_c = NamedRelationalAlgebraFrozenSet(("x", "y"), c)

    empty = NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    dee = NamedRelationalAlgebraFrozenSet.dee()

    assert (ras_a - empty) == ras_a
    assert (empty - ras_a) == empty
    assert (empty - empty) == empty
    assert (dee - empty) == dee
    assert (dee - dee) == NamedRelationalAlgebraFrozenSet.dum()

    res = ras_a - ras_b
    assert res == ras_c

    res = ras_b - ras_a
    assert len(res) == 0

    res = ras_a - ras_b_inv
    assert res == ras_c

    res = ras_b_inv - ras_a
    assert len(res) == 0

def test_groupby():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = list(ras_a.groupby(0))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)

def test_aggregate():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_sum = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )
    expected_str = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 2)]
    )
    expected_lambda = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    initial_set2 = NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9)]
    )
    expected_op2 = NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(2, 7, 8, 8)]
    )
    expected_op3 = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "qq"), [(7, 8, 13)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": "count"})
    assert expected_str == new_set

def test_named_relational_algebra_ra_left_naturaljoin():
    import numpy as np

    ras_a = NamedRelationalAlgebraFrozenSet(
        ("z", "y"), [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
    )

    ras_b = NamedRelationalAlgebraFrozenSet(
        ("z", "y", "v"), [(0, 0, 1), (2, 3, 2), (4, 6, 3), (6, 9, 4), (8, 12, 5)]
    )

    ras_c = NamedRelationalAlgebraFrozenSet(
        ("y", "v"), [(0, 0), (2, 6), (4, 9), (8, 4)]
    )

    empty = NamedRelationalAlgebraFrozenSet(("z", "y"), [])
    dee = NamedRelationalAlgebraFrozenSet.dee()
    dum = NamedRelationalAlgebraFrozenSet.dum()

    expected_a_b = NamedRelationalAlgebraFrozenSet(
        ("z", "y", "v")
        , [(0, 0, 1), (1, 2, np.nan), (2, 4, np.nan), (3, 6, np.nan), (4, 8, np.nan)]
    )

    expected_b_a = ras_b

    expected_a_c = NamedRelationalAlgebraFrozenSet(
        ("y", "z", "v"), [(0, 0, 0), (2, 1, 6), (4, 2, 9), (6, 3, np.nan), (8, 4, 4)]
    )

    res = ras_a.left_naturaljoin(ras_b)
    assert res == expected_a_b

    res = ras_b.left_naturaljoin(ras_a)
    assert res == expected_b_a

    res = ras_a.left_naturaljoin(ras_a)
    assert res == ras_a

    res = ras_a.left_naturaljoin(ras_c)
    assert res == expected_a_c


    assert len(ras_a.left_naturaljoin(empty)) == 5
    assert len(empty.left_naturaljoin(ras_a)) == 0
    assert ras_a.left_naturaljoin(dee) == ras_a
    assert dee.left_naturaljoin(ras_a) == dee
    assert ras_a.left_naturaljoin(dum) == ras_a
    assert dum.left_naturaljoin(ras_a) == dum