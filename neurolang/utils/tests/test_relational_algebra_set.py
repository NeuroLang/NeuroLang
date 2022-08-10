from typing import AbstractSet, Tuple
import pytest

from ..relational_algebra_set import (
    RelationalAlgebraColumnInt,
    RelationalAlgebraColumnStr,
    dask_sql,
    pandas,
)


@pytest.fixture(ids=["pandas", "dask_sql"], params=[(pandas,), (dask_sql,)])
def ra_module(request):
    return request.param[0]


def test_relational_algebra_set_semantics_empty(ra_module):
    ras = ra_module.RelationalAlgebraSet()

    assert len(ras) == 0
    assert ras.is_empty()
    assert ras.arity == 0
    assert 0 not in ras
    assert list(iter(ras)) == []
    assert ras == ra_module.RelationalAlgebraSet.dum()

    ras.add((0, 1))
    assert (0, 1) in ras
    assert len(ras) == 1
    assert ras.arity == 2


def test_relational_algebra_set_semantics(ra_module):
    a = [5, 4, 3, 2, 3, 1]
    ras = ra_module.RelationalAlgebraSet(a)
    ras_ = ra_module.RelationalAlgebraSet(a)
    ras__ = set((e,) for e in a)

    assert list(map(int, ras.columns)) == [0]

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

    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0

    r = ra_module.RelationalAlgebraSet.create_view_from(ras)
    assert r == ras
    assert r is not ras

    r = ra_module.RelationalAlgebraSet(ras)
    assert r == ras
    assert r is not ras

    r = ra_module.RelationalAlgebraSet([()])
    assert r.is_dee()


def test_iter_and_fetch_one(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    ras_a = ra_module.RelationalAlgebraSet(a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res

    res_dee = ra_module.RelationalAlgebraSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()


def test_as_numpy_array(ra_module):
    a = set((i % 2, i, i * 2) for i in range(5))
    ras = ra_module.RelationalAlgebraSet(a)
    ras_array = ras.as_numpy_array()
    assert ras_array.shape == (5, 3)
    assert ras_array.dtype == int
    assert set(tuple(r) for r in ras_array) == a


def test_relational_algebra_ra_projection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    assert (0,) in ras_0 and (1,) in ras_0
    assert len(ras_0) == 2
    assert ra_module.RelationalAlgebraSet.dum().projection(0).is_empty()
    assert ra_module.RelationalAlgebraSet.dee().projection().is_dee()

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))
    assert ras.projection() == ra_module.RelationalAlgebraSet.dee()


def test_relational_algebra_ra_selection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.RelationalAlgebraSet(a)

    # Select elements where col0 == 1
    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    # Select elements where col0 == 1 and col1 == 2. Result should be empty.
    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set()
    assert ras_0 == a_sel

    # Select elements where the first parameter is 0
    ras_0 = ras.selection(lambda x: x[0] == 0)
    assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 0)

    # Select elements where the col1 has values that are odd
    ras_0 = ras.selection({1: lambda x: x % 2 == 1})
    assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)

    # Select elements where the col0 is 1 and col2 > 2
    ras_0 = ras.selection({0: 1, 2: lambda x: x > 2})
    assert ras_0 == set(
        (i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i > 1
    )

    assert ra_module.RelationalAlgebraSet.dum().selection({0: 1}).is_empty()


def test_relational_algebra_ra_selection_columns(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.RelationalAlgebraSet(a)

    # Select elements where col0 == col1 and col1 == col2.
    # Result should be set((0, 0, 0))
    ras_1 = ras.selection_columns({0: 1, 1: 2})
    assert ras_1 == set(t for t in a if t[0] == t[1] & t[1] == t[2])
    assert (
        ra_module.RelationalAlgebraSet.dum()
        .selection_columns({0: 1})
        .is_empty()
    )


def test_relational_algebra_ra_equijoin(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, i, i * 2) for i in range(5)]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)
    ras_d = ra_module.RelationalAlgebraSet(d)
    ras_empty = ras_d.selection({0: 1000})
    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

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


def test_relational_algebra_ra_cross_product(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)
    ras_empty = ras_a.selection({0: 1000})
    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

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


def test_relational_algebra_ra_equijoin_mixed_types(ra_module):
    a = [(chr(ord("a") + i), i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(chr(ord("a") + i), i * 2, i * 2, i * 3) for i in range(5)]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c


def test_groupby(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)

    res = list(ras_a.groupby(0))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)


def test_relational_algebra_difference(ra_module):
    first = ra_module.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    third = ra_module.RelationalAlgebraFrozenSet([(1, 2, 3), (4, 5, 6)])
    dee = ra_module.RelationalAlgebraFrozenSet.dee()
    dum = ra_module.RelationalAlgebraFrozenSet.dum()

    assert first - second == ra_module.RelationalAlgebraFrozenSet([(7, 8)])
    assert second - first == ra_module.RelationalAlgebraFrozenSet([(42, 0)])
    assert (first - first).is_empty()
    assert dee - dee == dum
    assert dum - dee == dum
    with pytest.raises(
        ValueError,
        match="Relational algebra set operators can"
        " only be used on sets with same columns.",
    ):
        first - third


def test_relational_algebra_ra_union(ra_module):
    first = ra_module.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    assert first | first == first
    expected = ra_module.RelationalAlgebraFrozenSet([(7, 8), (9, 2), (42, 0)])
    assert first | second == expected
    empty = ra_module.RelationalAlgebraFrozenSet([])
    dee = ra_module.RelationalAlgebraFrozenSet.dee()
    dum = ra_module.RelationalAlgebraFrozenSet.dum()

    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | dum == first
    assert dum | first == first
    assert first | empty | second == first | second

    assert first | set() == first


def test_relational_algebra_ra_intersection(ra_module):
    first = ra_module.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    assert first & first == first
    expected = ra_module.RelationalAlgebraFrozenSet([(9, 2)])
    assert first & second == expected
    empty = ra_module.RelationalAlgebraFrozenSet([])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty

    assert first & set() == empty


def test_relational_algebra_ra_union_update(ra_module):
    first = ra_module.RelationalAlgebraSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraSet([(9, 2), (42, 0)])
    f = first.copy()
    f |= first
    assert f == first
    expected = ra_module.RelationalAlgebraSet([(7, 8), (9, 2), (42, 0)])
    f = first.copy()
    f |= second
    assert f == expected
    empty = ra_module.RelationalAlgebraSet([])
    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

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


@pytest.mark.skip("Not implemented yet")
def test_relational_algebra_ra_intersection_update(ra_module):
    first = ra_module.RelationalAlgebraSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraSet([(9, 2), (42, 0)])
    f = first.copy()
    f |= first
    assert f == first
    expected = ra_module.RelationalAlgebraSet([(9, 2)])
    f = first.copy()
    f &= second
    assert f == expected
    empty = ra_module.RelationalAlgebraSet([])
    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

    f = first.copy()
    f &= empty
    assert f.is_empty()

    e = empty.copy()
    e &= first
    assert e.is_empty()

    d = dee.copy()
    d &= dee
    assert d == dee

    f = first.copy()
    f &= dum
    assert f == dum

    d = dum.copy()
    d &= first
    assert d == dum

    f = first.copy()
    f &= second
    assert f == first & second

    f = first.copy()
    f &= set()
    assert f.is_empty()


def test_relational_algebra_ra_difference_update(ra_module):
    first = ra_module.RelationalAlgebraSet([(7, 8), (9, 2)])
    second = ra_module.RelationalAlgebraSet([(9, 2), (42, 0)])
    f = first.copy()
    f -= first
    assert f.is_empty()
    expected = ra_module.RelationalAlgebraSet([(7, 8)])
    f = first.copy()
    f -= second
    assert f == expected
    empty = ra_module.RelationalAlgebraSet([])
    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

    f = first.copy()
    f -= empty
    assert f == first

    e = empty.copy()
    e -= first
    assert e.is_empty()

    d = dee.copy()
    d -= dee
    assert d.is_empty()
    assert d.is_dum()

    f = first.copy()
    f -= dum
    assert f == first

    d = dum.copy()
    d -= first
    assert d.is_dum()

    f = first.copy()
    f -= empty
    f -= second
    assert f == expected

    f = first.copy()
    f |= set()
    assert f == first


def test_columns(ra_module):
    first = ra_module.RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )

    assert tuple(int(c) for c in first.columns) == (0, 1)
    assert len(ra_module.RelationalAlgebraSet.dum().columns) == 0


def test_named_relational_algebra_set_semantics_empty(ra_module):
    ras = ra_module.NamedRelationalAlgebraFrozenSet(("y", "x"))

    assert ras.columns == ("y", "x")

    assert ras.is_empty()
    assert len(ras) == 0
    assert ras.arity == 2
    assert list(iter(ras)) == []
    assert ras != ra_module.NamedRelationalAlgebraFrozenSet.dum()
    assert ras.projection() == ra_module.NamedRelationalAlgebraFrozenSet.dum()

    ras = ra_module.NamedRelationalAlgebraFrozenSet(("y", "x"), [(0, 1)])
    assert (0, 1) in ras
    assert {"x": 1, "y": 0} in ras
    assert {"y": 1, "x": 1} not in ras
    assert len(ras) == 1
    assert ras.arity == 2

    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(("y", "z"), [(0, 1)])
    assert ras != ras_b

    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(1, 0)])
    assert ras == ras_c

    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0
    assert dee != dum

    r = ra_module.NamedRelationalAlgebraFrozenSet.create_view_from(ras)
    assert r == ras
    assert r is not ras

    r = ra_module.NamedRelationalAlgebraFrozenSet(ras)
    assert r == ras
    assert r is not ras

    r_unnamed = ra_module.RelationalAlgebraSet([(0, 1)])
    r = ra_module.NamedRelationalAlgebraFrozenSet(["y", "x"], r_unnamed)
    assert r == ras


def test_named_relational_algebra_ra_projection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), a)

    ras_x = ras.projection("x")
    assert (0,) in ras_x and (1,) in ras_x
    assert len(ras_x) == 2
    assert ras_x.columns == ("x",)

    ras_xz = ras.projection("x", "z")
    assert all((i % 2, i * 2) in ras_xz for i in range(5))

    ras_ = ras.projection()
    assert ras_.arity == 0
    assert len(ras_) > 0
    assert ras_.projection("x") == ras_

    assert ras_.projection() == ra_module.NamedRelationalAlgebraFrozenSet.dee()


def test_named_relational_algebra_ra_projection_to_unnamed(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), a)

    ras_x = ras.projection_to_unnamed("x")
    assert (0,) in ras_x and (1,) in ras_x
    assert len(ras_x) == 2
    assert tuple(map(int, ras_x.columns)) == (0,)

    ras_xz = ras.projection_to_unnamed("x", "z")
    assert all((i % 2, i * 2) in ras_xz for i in range(5))
    assert tuple(map(int, ras_xz.columns)) == (0, 1)

    ras_xx = ras.projection_to_unnamed("x", "x")
    assert all((i % 2, i % 2) in ras_xx for i in range(5))
    assert tuple(map(int, ras_xx.columns)) == (0, 1)

    ras_ = ras.projection_to_unnamed()
    assert ras_.arity == 0
    assert len(ras_) > 0
    assert (
        ras.projection_to_unnamed()
        == ra_module.RelationalAlgebraFrozenSet.dee()
    )


def test_named_relational_algebra_ra_selection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), a)

    ras_0 = ras.selection({"x": 1})
    a_sel = ra_module.NamedRelationalAlgebraFrozenSet(
        ras.columns, set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    )
    assert ras_0 == a_sel

    ras_0 = ras.selection({"x": 1, "y": 2})
    a_sel = ra_module.NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2),
    )
    assert ras_0 == a_sel

    ras_0 = ras.selection({"x": lambda x: x == 1, "y": lambda y: y == 2})
    assert ras_0 == a_sel

    ras_0 = ras.selection(lambda t: (t.x == 1) & (t.y == 2))
    assert ras_0 == a_sel

    ras_0 = ras.selection(
        ra_module.RelationalAlgebraStringExpression("x == 1 and y == 2")
    )
    assert ras_0 == a_sel


def test_named_relational_algebra_ra_naturaljoin(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, j * 2, j * 3) for i in range(5) for j in range(5)]

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(("y", "x"), b)
    ras_b2 = ra_module.NamedRelationalAlgebraFrozenSet(("u", "v"), b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(("z", "y", "x"), c)
    ras_d = ra_module.NamedRelationalAlgebraFrozenSet(("z", "y", "u", "v"), d)
    empty = ra_module.NamedRelationalAlgebraFrozenSet(("z", "y"), [])
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

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


def test_named_relational_algebra_ra_left_naturaljoin(ra_module):
    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z", "y"), [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
    )

    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z", "y", "v"),
        [(0, 0, 1), (2, 3, 2), (4, 6, 3), (6, 9, 4), (8, 12, 5)],
    )

    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "v"), [(0, 0), (2, 6), (4, 9), (8, 4)]
    )

    ras_d = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w",), [(2,)]
    )

    ras_e = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w",), []
    )

    ras_e_null = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w",), [(ra_module.NA,)]
    )

    empty = ra_module.NamedRelationalAlgebraFrozenSet(("z", "y"), [])
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

    expected_a_b = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z", "y", "v"),
        [
            (0, 0, 1),
            (1, 2, ra_module.NA),
            (2, 4, ra_module.NA),
            (3, 6, ra_module.NA),
            (4, 8, ra_module.NA),
        ],
    )

    expected_b_a = ras_b

    expected_a_c = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "z", "v"),
        [(0, 0, 0), (2, 1, 6), (4, 2, 9), (6, 3, ra_module.NA), (8, 4, 4)],
    )

    res = ras_a.left_naturaljoin(ras_b)
    assert res == expected_a_b

    res = ras_b.left_naturaljoin(ras_a)
    assert res == expected_b_a

    res = ras_a.left_naturaljoin(ras_a)
    assert res == ras_a

    res = ras_a.left_naturaljoin(ras_c)
    assert res == expected_a_c

    res = ras_a.left_naturaljoin(ras_d)
    assert res == ras_a.cross_product(ras_d)

    res = ras_a.left_naturaljoin(ras_e)
    assert res == ras_a.cross_product(ras_e_null)

    assert len(ras_a.left_naturaljoin(empty)) == 5
    assert len(empty.left_naturaljoin(ras_a)) == 0
    assert ras_a.left_naturaljoin(dee) == ras_a
    assert dee.left_naturaljoin(ras_a) == dee
    assert ras_a.left_naturaljoin(dum) == ras_a
    assert dum.left_naturaljoin(ras_a) == dum


def test_named_relational_algebra_ra_cross_product(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(("u", "v"), b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "u", "v"), c)
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

    res = ras_a.cross_product(ras_b)
    assert res == ras_c

    assert ras_a.naturaljoin(dee) == ras_a
    assert dee.naturaljoin(ras_a) == ras_a
    assert ras_a.naturaljoin(dum) == dum
    assert dum.naturaljoin(ras_a) == dum


def test_named_relational_algebra_difference(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i, i * 2) for i in range(1, 5)]
    c = [(i, i * 2) for i in range(1)]

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), b)
    ras_b_inv = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "x"), [t[::-1] for t in b]
    )
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), c)

    empty = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()

    assert (ras_a - empty) == ras_a
    assert (empty - ras_a) == empty
    assert (empty - empty) == empty
    assert (dee - empty) == dee
    assert (dee - dee) == ra_module.NamedRelationalAlgebraFrozenSet.dum()

    res = ras_a - ras_b
    assert res == ras_c

    res = ras_b - ras_a
    assert len(res) == 0

    res = ras_a - ras_b_inv
    assert res == ras_c

    res = ras_b_inv - ras_a
    assert len(res) == 0


def test_named_groupby(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    cols = ("x", "y")

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(cols, b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(cols, c)
    empty = ra_module.NamedRelationalAlgebraFrozenSet(cols, [])

    res = list(ras_a.groupby("x"))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)

    assert list(empty.groupby("x")) == []


def test_named_iter_and_fetch_one(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res

    res_dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()


def test_rename_column(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ras_a.rename_column("y", "z")
    assert all(
        el_a.x == el_b.x and el_a.y == el_b.z
        for el_a, el_b in zip(ras_a, ras_b)
    )

    ras_c = ra_module.NamedRelationalAlgebraFrozenSet.dum()
    ras_c = ras_c.rename_column("x", "y")
    assert ras_c.is_dum()


def test_named_to_unnamed(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ra_module.RelationalAlgebraFrozenSet(a)
    assert ras_a.to_unnamed() == ras_b


def test_named_ra_set_from_other(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "n"), [(56, "bonjour"), (42, "aurevoir")]
    )
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        first.columns,
        first,
    )
    assert first == second
    for tuple_a, tuple_b in zip(first, second):
        assert tuple_a == tuple_b

    third = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x",), ra_module.NamedRelationalAlgebraFrozenSet(tuple())
    )

    assert len(third) == 0
    assert third.columns == ("x",)

    fourth = ra_module.NamedRelationalAlgebraFrozenSet(
        (), ra_module.RelationalAlgebraFrozenSet.dee()
    )
    assert fourth.is_dee()


def test_named_ra_union(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(9, 2), (42, 0)]
    )
    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2), (42, 0)]
    )
    assert first | second == expected
    empty = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | dum == first
    assert dum | first == first
    assert first | empty | second == first | second


def test_named_ra_intersection(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(9, 2), (42, 0)]
    )
    expected = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2)])
    assert first & second == expected
    empty = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty


def test_aggregate(ra_module):
    initial_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_sum = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )
    expected_str = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 2)]
    )
    expected_lambda = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    initial_set2 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9)]
    )
    expected_op2 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(2, 7, 8, 8)]
    )
    expected_op3 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "qq"), [(7, 8, 13)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": "count"})
    assert expected_str == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": lambda x: max(x) - 1})
    assert expected_lambda == new_set
    new_set = initial_set.aggregate(
        ["x", "y"],
        [
            ("z", "z", lambda x: max(x) - 1),
        ],
    )
    assert expected_lambda == new_set
    new_set = initial_set2.aggregate(
        ["x", "y"], {"z": lambda x: max(x) - 1, "w": "count"}
    )
    assert expected_op2 == new_set

    new_set = initial_set2.aggregate(
        ["x", "y"], {"qq": lambda t: sum(t.w + t.z)}
    )

    assert new_set == expected_op3


def test_aggregate_with_duplicates(ra_module):
    initial_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9), (7, 8, 1)]
    )
    expected_sum = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set

    initial_set2 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9), (2, 7, 8, 9)]
    )
    expected_op2 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "t"), [(7, 8, 13)]
    )
    new_set = initial_set2.aggregate(
        ["x", "y"], {"t": lambda t: sum(t.w + t.z)}
    )
    assert expected_op2 == new_set


def test_aggregate_with_pandas_builtin_functions(ra_module):
    initial_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(i, i * j) for i in range(3) for j in range(3)]
    )
    expected_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(0, 9)]
    )
    agg_set = initial_set.aggregate(
        [],
        {
            "x": ra_module.RelationalAlgebraStringExpression("first"),
            "y": lambda x: sum(x),
        },
    )
    assert agg_set == expected_set


def test_aggregate_with_empty_sets(ra_module):
    expected_set = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=("x", "y")
    )
    agg_set = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=("x", "y")
    ).aggregate(("x",), [("y", "y", sum)])
    assert agg_set == expected_set

    agg_set = ra_module.NamedRelationalAlgebraFrozenSet.dum().aggregate(
        ("x",), [("y", "y", sum)]
    )
    assert agg_set == expected_set

    with pytest.raises(ValueError):
        agg_set = ra_module.NamedRelationalAlgebraFrozenSet.dee().aggregate(
            ("x",), [("y", "y", sum)]
        )


def test_relational_algebra_set_python_type_support(ra_module):
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)})),
        (10, "cat", frozenset({(5, 6), (8, 9)})),
    ]
    ras_a = ra_module.RelationalAlgebraFrozenSet(data)
    assert set(data) == set(ras_a)


def test_extended_projection(ra_module):
    initial_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    expected_sum = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z",), [(15,), (11,)]
    )
    expected_lambda = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z",), [(14,), (10,)]
    )
    expected_lambda2 = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z", "x"), [(14, 8), (10, 10)]
    )
    expected_new_colum_str = ra_module.NamedRelationalAlgebraFrozenSet(
        (
            "x",
            "z",
        ),
        [
            (
                7,
                "a",
            ),
            (
                9,
                "a",
            ),
        ],
    )
    expected_new_colum_int = ra_module.NamedRelationalAlgebraFrozenSet(
        ("z",), [(1,), (1,)]
    )
    new_set = initial_set.extended_projection({"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.extended_projection(
        {"z": ra_module.RelationalAlgebraStringExpression("x+y")}
    )
    assert expected_sum == new_set
    new_set = initial_set.extended_projection({"z": lambda r: r.x + r.y - 1})
    assert expected_lambda == new_set
    new_set = initial_set.extended_projection(
        {
            "z": lambda r: (r.x + r.y - 1),
            "x": ra_module.RelationalAlgebraStringExpression("x+1"),
        }
    )
    assert expected_lambda2 == new_set
    new_set = initial_set.extended_projection(
        {"z": "a", "x": ra_module.RelationalAlgebraStringExpression("x")}
    )
    assert expected_new_colum_str == new_set
    new_set = initial_set.extended_projection({"z": 1})
    assert expected_new_colum_int == new_set

    new_set = initial_set.extended_projection(
        {"x": RelationalAlgebraColumnStr("x")}
    )
    assert initial_set.projection("x") == new_set

    base_set = ra_module.NamedRelationalAlgebraFrozenSet(
        (1, 2), [(7, 8), (9, 2)]
    )

    new_set = base_set.extended_projection(
        {
            "x": RelationalAlgebraColumnInt(1),
            "y": RelationalAlgebraColumnInt(2),
        }
    )

    assert initial_set == new_set


def test_extended_projection_on_dee(ra_module):
    ras_a = (
        ra_module.NamedRelationalAlgebraFrozenSet.dee().extended_projection(
            {"new_col": "b"}
        )
    )
    expected_set = ra_module.NamedRelationalAlgebraFrozenSet(
        ("new_col",), [("b",)]
    )
    assert ras_a == expected_set


def test_extended_projection_on_python_sets(ra_module):
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)})),
        (10, "cat", frozenset({(5, 6), (8, 9)})),
    ]
    ras = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), data)
    expected_len = ra_module.NamedRelationalAlgebraFrozenSet(
        ("l",), [(3,), (3,)]
    )

    new_set = ras.extended_projection({"l": lambda x: len(x)})
    assert expected_len == new_set


def test_rename_columns(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "x"}) == first
    assert id(first.rename_columns({"x": "x"})) != id(first)
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "x"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "y", "y": "x"}) == second
    with pytest.raises(ValueError, match=r"non-existing columns: {'z'}"):
        first.rename_columns({"z": "w"})

    ras_c = ra_module.NamedRelationalAlgebraFrozenSet.dum()
    ras_c = ras_c.rename_columns({"x": "y"})
    assert ras_c.is_dum()


def test_rename_columns_duplicates(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    with pytest.raises(ValueError, match=r"Duplicated.*{'z'}"):
        first.rename_columns({"x": "z", "y": "z"})


def test_equality(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    assert first == first
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "x"),
        [(0, 2), (0, 4)],
    )
    assert first != second
    assert second != first
    third = ra_module.NamedRelationalAlgebraFrozenSet(columns=())
    assert first != third
    assert third != first
    assert third == third


def test_relation_duplicated_columns(ra_module):
    with pytest.raises(ValueError, match=r".*Duplicated.*: {'x'}"):
        ra_module.NamedRelationalAlgebraFrozenSet(
            ("x", "x"),
            [(0, 2), (0, 4)],
        )


def test_extended_projection_ra_string_expression_empty_relation(ra_module):
    # reported in GH387
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[],
    )
    eval_expressions = {
        "z": ra_module.RelationalAlgebraStringExpression("(x / y)")
    }
    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["z"],
        iterable=[],
    )
    assert relation.extended_projection(eval_expressions) == expected


def test_replace_null(ra_module):
    relation_left = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x"],
        iterable=[(0,), (1,)],
    )

    relation_right = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[(1, 2)],
    )

    relation = relation_left.left_naturaljoin(relation_right)

    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[(0, -1), (1, 2)],
    )

    assert relation.replace_null("y", -1) == expected


def test_explode(ra_module):
    data = [
        (5, frozenset({1, 2, 5, 6}), "dog"),
        (10, frozenset({5, 9}), "cat"),
    ]
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z"],
        iterable=data,
    )

    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z", "t"],
        iterable=[
            (5, frozenset({1, 2, 5, 6}), "dog", 1),
            (5, frozenset({1, 2, 5, 6}), "dog", 2),
            (5, frozenset({1, 2, 5, 6}), "dog", 5),
            (5, frozenset({1, 2, 5, 6}), "dog", 6),
            (10, frozenset({5, 9}), "cat", 5),
            (10, frozenset({5, 9}), "cat", 9),
        ],
    )
    result = relation.explode("y", "t")
    assert result == expected
    if hasattr(result, "set_row_type"):
        assert result.set_row_type == Tuple[int, AbstractSet[int], str, int]


def test_explode_multi_columns(ra_module):
    data = [
        (0, 1, frozenset({(3, 4), (4, 5)})),
        (0, 2, frozenset({(5, 6)})),
    ]
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z"],
        iterable=data,
    )
    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z", "v", "w"],
        iterable=[
            (0, 1, frozenset({(3, 4), (4, 5)}), 3, 4),
            (0, 1, frozenset({(3, 4), (4, 5)}), 4, 5),
            (0, 2, frozenset({(5, 6)}), 5, 6),
        ],
    )
    result = relation.explode("z", ("v", "w"))
    assert result == expected


def test_aggregate_repeated_group_column(ra_module):
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[("a", 4), ("b", 5)],
    )
    with pytest.raises(ValueError, match="Cannot group on repeated columns"):
        relation.aggregate(["x", "x"], {"y": sum})


def test_unsupported_aggregation_function(ra_module):
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x"],
        iterable=[("a",), ("b",)],
    )
    with pytest.raises(ValueError, match="Unsupported aggregate_function"):
        relation.aggregate(["x"], None)


def test_hash_none_container(ra_module):
    # GH584: hash of RA set with None _container
    relation = ra_module.RelationalAlgebraSet()
    assert hash(relation) == hash((tuple(), None))
