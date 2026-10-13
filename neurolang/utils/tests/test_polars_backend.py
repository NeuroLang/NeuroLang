"""Tests for the Polars Relational Algebra Set backend.

Verifies that all core operations produce identical results
to the pandas backend on the same inputs.
"""
from typing import AbstractSet, Tuple

import pytest
import polars as pl
import numpy as np

from ..relational_algebra_set import (
    RelationalAlgebraColumnInt,
    RelationalAlgebraColumnStr,
    RelationalAlgebraStringExpression,
    pandas,
    polars,
)


NA = polars.NA


def test_relational_algebra_set_semantics_empty():
    ras = polars.RelationalAlgebraSet()

    assert len(ras) == 0
    assert ras.is_empty()
    assert ras.arity == 0
    assert 0 not in ras
    assert list(iter(ras)) == []
    assert ras == polars.RelationalAlgebraSet.dum()

    ras.add((0, 1))
    assert (0, 1) in ras
    assert len(ras) == 1
    assert ras.arity == 2


def test_relational_algebra_set_semantics():
    a = [5, 4, 3, 2, 3, 1]
    ras = polars.RelationalAlgebraSet(a)
    ras_ = polars.RelationalAlgebraSet(a)
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

    dee = polars.RelationalAlgebraSet.dee()
    dum = polars.RelationalAlgebraSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0

    r = polars.RelationalAlgebraSet.create_view_from(ras)
    assert r == ras
    assert r is not ras

    r = polars.RelationalAlgebraSet(ras)
    assert r == ras
    assert r is not ras

    r = polars.RelationalAlgebraSet([()])
    assert r.is_dee()


def test_iter_and_fetch_one():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    ras_a = polars.RelationalAlgebraSet(a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res

    res_dee = polars.RelationalAlgebraSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()


def test_as_numpy_array():
    a = set((i % 2, i, i * 2) for i in range(5))
    ras = polars.RelationalAlgebraSet(a)
    ras_array = ras.as_numpy_array()
    assert ras_array.shape == (5, 3)
    assert ras_array.dtype == int
    assert set(tuple(r) for r in ras_array) == a


def test_relational_algebra_ra_projection():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = polars.RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    assert (0,) in ras_0 and (1,) in ras_0
    assert len(ras_0) == 2
    assert polars.RelationalAlgebraSet.dum().projection(0).is_empty()
    assert polars.RelationalAlgebraSet.dee().projection().is_dee()

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))
    assert ras.projection() == polars.RelationalAlgebraSet.dee()


def test_relational_algebra_ra_selection():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = polars.RelationalAlgebraSet(a)

    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set()
    assert ras_0 == a_sel

    ras_0 = ras.selection(lambda x: x[0] == 0)
    assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 0)

    ras_0 = ras.selection({1: lambda x: x % 2 == 1})
    assert ras_0 == set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)

    ras_0 = ras.selection({0: 1, 2: lambda x: x > 2})
    assert ras_0 == set(
        (i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i > 1
    )

    assert polars.RelationalAlgebraSet.dum().selection({0: 1}).is_empty()


def test_relational_algebra_ra_selection_columns():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = polars.RelationalAlgebraSet(a)

    ras_1 = ras.selection_columns({0: 1, 1: 2})
    assert ras_1 == set(t for t in a if t[0] == t[1] & t[1] == t[2])
    assert (
        polars.RelationalAlgebraSet.dum()
        .selection_columns({0: 1})
        .is_empty()
    )


def test_relational_algebra_ra_equijoin():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, i, i * 2) for i in range(5)]

    ras_a = polars.RelationalAlgebraSet(a)
    ras_b = polars.RelationalAlgebraSet(b)
    ras_c = polars.RelationalAlgebraSet(c)
    ras_d = polars.RelationalAlgebraSet(d)
    ras_empty = ras_d.selection({0: 1000})
    dee = polars.RelationalAlgebraSet.dee()
    dum = polars.RelationalAlgebraSet.dum()

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

    ras_a = polars.RelationalAlgebraSet(a)
    ras_b = polars.RelationalAlgebraSet(b)
    ras_c = polars.RelationalAlgebraSet(c)
    ras_empty = ras_a.selection({0: 1000})
    dee = polars.RelationalAlgebraSet.dee()
    dum = polars.RelationalAlgebraSet.dum()

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

    ras_a = polars.RelationalAlgebraSet(a)
    ras_b = polars.RelationalAlgebraSet(b)
    ras_c = polars.RelationalAlgebraSet(c)

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c


def test_groupby():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    ras_a = polars.RelationalAlgebraSet(a)
    ras_b = polars.RelationalAlgebraSet(b)
    ras_c = polars.RelationalAlgebraSet(c)

    res = list(ras_a.groupby(0))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)


def test_relational_algebra_difference():
    first = polars.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = polars.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    third = polars.RelationalAlgebraFrozenSet([(1, 2, 3), (4, 5, 6)])
    dee = polars.RelationalAlgebraFrozenSet.dee()
    dum = polars.RelationalAlgebraFrozenSet.dum()

    assert first - second == polars.RelationalAlgebraFrozenSet([(7, 8)])
    assert second - first == polars.RelationalAlgebraFrozenSet([(42, 0)])
    assert (first - first).is_empty()
    assert dee - dee == dum
    assert dum - dee == dum
    with pytest.raises(
        ValueError,
        match="Relational algebra set operators can"
        " only be used on sets with same columns.",
    ):
        first - third


def test_relational_algebra_ra_union():
    first = polars.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = polars.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    assert first | first == first
    expected = polars.RelationalAlgebraFrozenSet([(7, 8), (9, 2), (42, 0)])
    assert first | second == expected
    empty = polars.RelationalAlgebraFrozenSet([])
    dee = polars.RelationalAlgebraFrozenSet.dee()
    dum = polars.RelationalAlgebraFrozenSet.dum()

    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | dum == first
    assert dum | first == first
    assert first | empty | second == first | second

    assert first | set() == first


def test_relational_algebra_ra_intersection():
    first = polars.RelationalAlgebraFrozenSet([(7, 8), (9, 2)])
    second = polars.RelationalAlgebraFrozenSet([(9, 2), (42, 0)])
    assert first & first == first
    expected = polars.RelationalAlgebraFrozenSet([(9, 2)])
    assert first & second == expected
    empty = polars.RelationalAlgebraFrozenSet([])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty

    assert first & set() == empty


def test_relational_algebra_ra_union_update():
    first = polars.RelationalAlgebraSet([(7, 8), (9, 2)])
    second = polars.RelationalAlgebraSet([(9, 2), (42, 0)])
    f = first.copy()
    f |= first
    assert f == first
    expected = polars.RelationalAlgebraSet([(7, 8), (9, 2), (42, 0)])
    f = first.copy()
    f |= second
    assert f == expected
    empty = polars.RelationalAlgebraSet([])
    dee = polars.RelationalAlgebraSet.dee()
    dum = polars.RelationalAlgebraSet.dum()

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


def test_relational_algebra_ra_difference_update():
    first = polars.RelationalAlgebraSet([(7, 8), (9, 2)])
    second = polars.RelationalAlgebraSet([(9, 2), (42, 0)])
    f = first.copy()
    f -= first
    assert f.is_empty()
    expected = polars.RelationalAlgebraSet([(7, 8)])
    f = first.copy()
    f -= second
    assert f == expected
    empty = polars.RelationalAlgebraSet([])
    dee = polars.RelationalAlgebraSet.dee()
    dum = polars.RelationalAlgebraSet.dum()

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


def test_columns():
    first = polars.RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )
    assert tuple(int(c) for c in first.columns) == (0, 1)
    assert len(polars.RelationalAlgebraSet.dum().columns) == 0



def test_named_iter_and_fetch_one():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = polars.NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res

    res_dee = polars.NamedRelationalAlgebraFrozenSet.dee()
    assert list(iter(res_dee)) == [tuple()]
    assert res_dee.fetch_one() == tuple()


def test_rename_column():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = polars.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ras_a.rename_column("y", "z")
    assert all(
        el_a.x == el_b.x and el_a.y == el_b.z
        for el_a, el_b in zip(ras_a, ras_b)
    )

    ras_c = polars.NamedRelationalAlgebraFrozenSet.dum()
    ras_c = ras_c.rename_column("x", "y")
    assert ras_c.is_dum()


def test_named_to_unnamed():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ("y", "x")

    ras_a = polars.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = polars.RelationalAlgebraFrozenSet(a)
    assert ras_a.to_unnamed() == ras_b


def test_named_ra_set_from_other():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "n"), [(56, "bonjour"), (42, "aurevoir")]
    )
    second = polars.NamedRelationalAlgebraFrozenSet(
        first.columns,
        first,
    )
    assert first == second
    for tuple_a, tuple_b in zip(first, second):
        assert tuple_a == tuple_b

    third = polars.NamedRelationalAlgebraFrozenSet(
        ("x",), polars.NamedRelationalAlgebraFrozenSet(tuple())
    )

    assert len(third) == 0
    assert third.columns == ("x",)

    fourth = polars.NamedRelationalAlgebraFrozenSet(
        (), polars.RelationalAlgebraFrozenSet.dee()
    )
    assert fourth.is_dee()


def test_named_ra_union():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    second = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(9, 2), (42, 0)]
    )
    expected = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2), (42, 0)]
    )
    assert first | second == expected
    empty = polars.NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    dee = polars.NamedRelationalAlgebraFrozenSet.dee()
    dum = polars.NamedRelationalAlgebraFrozenSet.dum()

    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | dum == first
    assert dum | first == first
    assert first | empty | second == first | second


def test_named_ra_intersection():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    second = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(9, 2), (42, 0)]
    )
    expected = polars.NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2)])
    assert first & second == expected
    empty = polars.NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty


def test_aggregate():
    initial_set = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_sum = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )
    expected_str = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 2)]
    )
    expected_lambda = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    initial_set2 = polars.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9)]
    )
    expected_op2 = polars.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(2, 7, 8, 8)]
    )
    expected_op3 = polars.NamedRelationalAlgebraFrozenSet(
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


def test_aggregate_with_duplicates():
    initial_set = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9), (7, 8, 1)]
    )
    expected_sum = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 10)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set

    initial_set2 = polars.NamedRelationalAlgebraFrozenSet(
        ("w", "x", "y", "z"), [(1, 7, 8, 1), (2, 7, 8, 9), (2, 7, 8, 9)]
    )
    expected_op2 = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y", "t"), [(7, 8, 13)]
    )
    new_set = initial_set2.aggregate(
        ["x", "y"], {"t": lambda t: sum(t.w + t.z)}
    )
    assert expected_op2 == new_set


def test_aggregate_with_pandas_builtin_functions():
    initial_set = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(i, i * j) for i in range(3) for j in range(3)]
    )
    expected_set = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(0, 9)]
    )
    agg_set = initial_set.aggregate(
        [],
        {
            "x": polars.RelationalAlgebraStringExpression("first"),
            "y": lambda x: sum(x),
        },
    )
    assert agg_set == expected_set


def test_aggregate_with_empty_sets():
    expected_set = polars.NamedRelationalAlgebraFrozenSet(
        columns=("x", "y")
    )
    agg_set = polars.NamedRelationalAlgebraFrozenSet(
        columns=("x", "y")
    ).aggregate(("x",), [("y", "y", sum)])
    assert agg_set == expected_set

    agg_set = polars.NamedRelationalAlgebraFrozenSet.dum().aggregate(
        ("x",), [("y", "y", sum)]
    )
    assert agg_set == expected_set

    with pytest.raises(ValueError):
        agg_set = polars.NamedRelationalAlgebraFrozenSet.dee().aggregate(
            ("x",), [("y", "y", sum)]
        )


def test_relational_algebra_set_python_type_support():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)})),
        (10, "cat", frozenset({(5, 6), (8, 9)})),
    ]
    ras_a = polars.RelationalAlgebraFrozenSet(data)
    assert set(data) == set(ras_a)


def test_extended_projection():
    initial_set = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    expected_sum = polars.NamedRelationalAlgebraFrozenSet(
        ("z",), [(15,), (11,)]
    )
    expected_lambda = polars.NamedRelationalAlgebraFrozenSet(
        ("z",), [(14,), (10,)]
    )
    expected_lambda2 = polars.NamedRelationalAlgebraFrozenSet(
        ("z", "x"), [(14, 8), (10, 10)]
    )
    expected_new_colum_str = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "z"),
        [(7, "a"), (9, "a")],
    )
    expected_new_colum_int = polars.NamedRelationalAlgebraFrozenSet(
        ("z",), [(1,), (1,)]
    )
    new_set = initial_set.extended_projection({"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.extended_projection(
        {"z": polars.RelationalAlgebraStringExpression("x+y")}
    )
    assert expected_sum == new_set
    new_set = initial_set.extended_projection({"z": lambda r: r.x + r.y - 1})
    assert expected_lambda == new_set
    new_set = initial_set.extended_projection(
        {
            "z": lambda r: (r.x + r.y - 1),
            "x": polars.RelationalAlgebraStringExpression("x+1"),
        }
    )
    assert expected_lambda2 == new_set
    new_set = initial_set.extended_projection(
        {"z": "a", "x": polars.RelationalAlgebraStringExpression("x")}
    )
    assert expected_new_colum_str == new_set
    new_set = initial_set.extended_projection({"z": 1})
    assert expected_new_colum_int == new_set

    new_set = initial_set.extended_projection(
        {"x": RelationalAlgebraColumnStr("x")}
    )
    assert initial_set.projection("x") == new_set

    base_set = polars.NamedRelationalAlgebraFrozenSet(
        (1, 2), [(7, 8), (9, 2)]
    )

    new_set = base_set.extended_projection(
        {
            "x": RelationalAlgebraColumnInt(1),
            "y": RelationalAlgebraColumnInt(2),
        }
    )

    assert initial_set == new_set


def test_extended_projection_on_dee():
    ras_a = (
        polars.NamedRelationalAlgebraFrozenSet.dee().extended_projection(
            {"new_col": "b"}
        )
    )
    expected_set = polars.NamedRelationalAlgebraFrozenSet(
        ("new_col",), [("b",)]
    )
    assert ras_a == expected_set


def test_extended_projection_on_python_sets():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)})),
        (10, "cat", frozenset({(5, 6), (8, 9)})),
    ]
    ras = polars.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), data)
    expected_len = polars.NamedRelationalAlgebraFrozenSet(
        ("l",), [(3,), (3,)]
    )

    new_set = ras.extended_projection({"l": lambda x: len(x)})
    assert expected_len == new_set


def test_rename_columns():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "x"}) == first
    assert id(first.rename_columns({"x": "x"})) != id(first)
    second = polars.NamedRelationalAlgebraFrozenSet(
        ("y", "x"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "y", "y": "x"}) == second
    with pytest.raises(ValueError, match=r"non-existing columns: {'z'}"):
        first.rename_columns({"z": "w"})

    ras_c = polars.NamedRelationalAlgebraFrozenSet.dum()
    ras_c = ras_c.rename_columns({"x": "y"})
    assert ras_c.is_dum()


def test_rename_columns_duplicates():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    with pytest.raises(ValueError, match=r"Duplicated.*{'z'}"):
        first.rename_columns({"x": "z", "y": "z"})


def test_equality():
    first = polars.NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    assert first == first
    second = polars.NamedRelationalAlgebraFrozenSet(
        ("y", "x"),
        [(0, 2), (0, 4)],
    )
    assert first != second
    assert second != first
    third = polars.NamedRelationalAlgebraFrozenSet(columns=())
    assert first != third
    assert third != first
    assert third == third


def test_relation_duplicated_columns():
    with pytest.raises(ValueError, match=r".*Duplicated.*: {'x'}"):
        polars.NamedRelationalAlgebraFrozenSet(
            ("x", "x"),
            [(0, 2), (0, 4)],
        )


def test_extended_projection_ra_string_expression_empty_relation():
    relation = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[],
    )
    eval_expressions = {
        "z": polars.RelationalAlgebraStringExpression("(x / y)")
    }
    expected = polars.NamedRelationalAlgebraFrozenSet(
        columns=["z"],
        iterable=[],
    )
    assert relation.extended_projection(eval_expressions) == expected


def test_replace_null():
    relation_left = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x"],
        iterable=[(0,), (1,)],
    )

    relation_right = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[(1, 2)],
    )

    relation = relation_left.left_naturaljoin(relation_right)

    expected = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[(0, -1), (1, 2)],
    )

    assert relation.replace_null("y", -1) == expected


def test_explode():
    data = [
        (5, frozenset({1, 2, 5, 6}), "dog"),
        (10, frozenset({5, 9}), "cat"),
    ]
    relation = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z"],
        iterable=data,
    )

    expected = polars.NamedRelationalAlgebraFrozenSet(
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


def test_explode_multi_columns():
    data = [
        (0, 1, frozenset({(3, 4), (4, 5)})),
        (0, 2, frozenset({(5, 6)})),
    ]
    relation = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z"],
        iterable=data,
    )
    expected = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y", "z", "v", "w"],
        iterable=[
            (0, 1, frozenset({(3, 4), (4, 5)}), 3, 4),
            (0, 1, frozenset({(3, 4), (4, 5)}), 4, 5),
            (0, 2, frozenset({(5, 6)}), 5, 6),
        ],
    )
    result = relation.explode("z", ("v", "w"))
    assert result == expected


def test_aggregate_repeated_group_column():
    relation = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"],
        iterable=[("a", 4), ("b", 5)],
    )
    with pytest.raises(ValueError, match="Cannot group on repeated columns"):
        relation.aggregate(["x", "x"], {"y": sum})


def test_unsupported_aggregation_function():
    relation = polars.NamedRelationalAlgebraFrozenSet(
        columns=["x"],
        iterable=[("a",), ("b",)],
    )
    with pytest.raises(ValueError, match="Unsupported aggregate_function"):
        relation.aggregate(["x"], None)


def test_hash_none_container():
    relation = polars.RelationalAlgebraSet()
    assert hash(relation) == hash((tuple(), None))


# ── Cross-backend validation tests ────────────────────────────────────
# Verify Polars produces identical results to Pandas on same inputs


def _cross_validate(setup_func, check_func):
    """Helper to cross-validate a setup function on both backends."""
    # Build pandas and polars RAS objects, apply operations, compare
    pandas_setup_ras = setup_func(pandas)
    polars_setup_ras = setup_func(polars)
    check_func(pandas_setup_ras, polars_setup_ras)


def test_cross_projection():
    a = [(i % 2, i, i * 2) for i in range(5)]

    def setup(module):
        return module.RelationalAlgebraSet(a)

    def check(p_ras, po_ras):
        assert list(po_ras.projection(0)) == list(p_ras.projection(0))
        assert list(po_ras.projection(0, 2)) == list(p_ras.projection(0, 2))

    _cross_validate(setup, check)


def test_cross_selection():
    a = [(i % 2, i, i * 2) for i in range(5)]

    def setup(module):
        return module.RelationalAlgebraSet(a)

    def check(p_ras, po_ras):
        assert list(po_ras.selection({0: 1})) == list(p_ras.selection({0: 1}))
        assert list(po_ras.selection(lambda x: x[0] == 1)) == list(
            p_ras.selection(lambda x: x[0] == 1)
        )

    _cross_validate(setup, check)


def test_cross_equijoin():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]

    def setup(module):
        return module.RelationalAlgebraSet(a)

    def setup2(module):
        return module.RelationalAlgebraSet(b)

    def check(ras_a_p, ras_a_po):
        ras_b_p = setup2(pandas)
        ras_b_po = setup2(polars)
        res_p = list(ras_a_p.equijoin(ras_b_p, [(1, 0)]))
        res_po = list(ras_a_po.equijoin(ras_b_po, [(1, 0)]))
        assert res_po == res_p

    _cross_validate(setup, check)


def test_cross_union():
    first_data = [(7, 8), (9, 2)]
    second_data = [(9, 2), (42, 0)]

    def setup_first(module):
        return module.RelationalAlgebraFrozenSet(first_data)

    def setup_second(module):
        return module.RelationalAlgebraFrozenSet(second_data)

    def check(p_first, po_first):
        p_second = setup_second(pandas)
        po_second = setup_second(polars)
        assert list(po_first | po_second) == list(p_first | p_second)

    _cross_validate(setup_first, check)


def test_cross_difference():
    first_data = [(7, 8), (9, 2)]
    second_data = [(9, 2), (42, 0)]

    def setup_first(module):
        return module.RelationalAlgebraFrozenSet(first_data)

    def setup_second(module):
        return module.RelationalAlgebraFrozenSet(second_data)

    def check(p_first, po_first):
        p_second = setup_second(pandas)
        po_second = setup_second(polars)
        assert list(po_first - po_second) == list(p_first - p_second)

    _cross_validate(setup_first, check)


def test_cross_intersection():
    first_data = [(7, 8), (9, 2)]
    second_data = [(9, 2), (42, 0)]

    def setup_first(module):
        return module.RelationalAlgebraFrozenSet(first_data)

    def setup_second(module):
        return module.RelationalAlgebraFrozenSet(second_data)

    def check(p_first, po_first):
        p_second = setup_second(pandas)
        po_second = setup_second(polars)
        assert list(po_first & po_second) == list(p_first & p_second)

    _cross_validate(setup_first, check)


def test_cross_naturaljoin():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]

    def setup_a(module):
        return module.NamedRelationalAlgebraFrozenSet(("z", "y"), a)

    def setup_b(module):
        return module.NamedRelationalAlgebraFrozenSet(("y", "x"), b)

    def check(p_a, po_a):
        p_b = setup_b(pandas)
        po_b = setup_b(polars)
        assert list(po_a.naturaljoin(po_b)) == list(p_a.naturaljoin(p_b))

    _cross_validate(setup_a, check)


def test_cross_cross_product():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]

    def setup_a(module):
        return module.RelationalAlgebraSet(a)

    def setup_b(module):
        return module.RelationalAlgebraSet(b)

    def check(p_a, po_a):
        p_b = setup_b(pandas)
        po_b = setup_b(polars)
        assert list(po_a.cross_product(po_b)) == list(p_a.cross_product(p_b))

    _cross_validate(setup_a, check)


def test_cross_groupby():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    def setup(module):
        return module.RelationalAlgebraSet(a)

    def check(p_ras, po_ras):
        p_groups = [(k, sorted(list(v))) for k, v in p_ras.groupby(0)]
        po_groups = [(k, sorted(list(v))) for k, v in po_ras.groupby(0)]
        assert po_groups == p_groups

    _cross_validate(setup, check)


def test_cross_aggregate():
    data = [("x", "y", "z"), (7, 8, 1), (7, 8, 9)]

    def setup(module):
        return module.NamedRelationalAlgebraFrozenSet(
            ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
        )

    def check(p_ras, po_ras):
        p_agg = list(p_ras.aggregate(["x", "y"], {"z": sum}))
        po_agg = list(po_ras.aggregate(["x", "y"], {"z": sum}))
        assert po_agg == p_agg

    _cross_validate(setup, check)


def test_cross_extended_projection():
    def setup(module):
        return module.NamedRelationalAlgebraFrozenSet(
            ("x", "y"), [(7, 8), (9, 2)]
        )

    def check(p_ras, po_ras):
        p_proj = list(
            p_ras.extended_projection(
                {"z": pandas.RelationalAlgebraStringExpression("x+y")}
            )
        )
        po_proj = list(
            po_ras.extended_projection(
                {"z": polars.RelationalAlgebraStringExpression("x+y")}
            )
        )
        assert po_proj == p_proj

    _cross_validate(setup, check)


def test_cross_dee_dum():
    def setup(module):
        return (module.RelationalAlgebraFrozenSet.dee(),
                module.RelationalAlgebraFrozenSet.dum())

    def check(p_tup, po_tup):
        p_dee, p_dum = p_tup
        po_dee, po_dum = po_tup
        assert po_dee.is_dee() == p_dee.is_dee()
        assert po_dum.is_dum() == p_dum.is_dum()
        assert po_dee.arity == p_dee.arity
        assert po_dum.arity == p_dum.arity

    _cross_validate(setup, check)
