import pytest

from ..relational_algebra_set import RelationalAlgebraStringExpression, pandas


@pytest.fixture(ids=['pandas'], params=[(pandas,)])
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

    assert ras.columns == [0]

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

    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2)
    assert ras_0 == a_sel

    ras_1 = ras.selection_columns({0: 1, 1: 2})
    assert ras_1 == set(t for t in a if t[0] == t[1] & t[1] == t[2])
    assert ras.selection({0: 10000}).selection_columns({0: 1}).is_empty()

    assert ra_module.RelationalAlgebraSet.dum().selection({0: 1}).is_empty()


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


def test_relational_algebra_ra_union(ra_module):
    first = ra_module.RelationalAlgebraFrozenSet(
        [(7, 8), (9, 2)]
    )
    second = ra_module.RelationalAlgebraFrozenSet(
        [(9, 2), (42, 0)]
    )
    assert first | first == first
    expected = ra_module.RelationalAlgebraFrozenSet(
        [(7, 8), (9, 2), (42, 0)]
    )
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
    first = ra_module.RelationalAlgebraFrozenSet(
        [(7, 8), (9, 2)]
    )
    second = ra_module.RelationalAlgebraFrozenSet(
        [(9, 2), (42, 0)]
    )
    assert first & first == first
    expected = ra_module.RelationalAlgebraFrozenSet([(9, 2)])
    assert first & second == expected
    empty = ra_module.RelationalAlgebraFrozenSet([])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty

    assert first & set() == empty


def test_relational_algebra_ra_union_update(ra_module):
    first = ra_module.RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )
    second = ra_module.RelationalAlgebraSet(
        [(9, 2), (42, 0)]
    )
    f = first.copy()
    f |= first
    assert f == first
    expected = ra_module.RelationalAlgebraSet(
        [(7, 8), (9, 2), (42, 0)]
    )
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
    first = ra_module.RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )
    second = ra_module.RelationalAlgebraSet(
        [(9, 2), (42, 0)]
    )
    f = first.copy()
    f |= first
    assert f == first
    expected = ra_module.RelationalAlgebraSet(
        [(9, 2)]
    )
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
    first = ra_module.RelationalAlgebraSet(
        [(7, 8), (9, 2)]
    )
    second = ra_module.RelationalAlgebraSet(
        [(9, 2), (42, 0)]
    )
    f = first.copy()
    f -= first
    assert f.is_empty()
    expected = ra_module.RelationalAlgebraSet(
        [(7, 8)]
    )
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


def test_named_relational_algebra_set_semantics_empty(ra_module):
    ras = ra_module.NamedRelationalAlgebraFrozenSet(("y", "x"))

    assert ras.columns == ('y', 'x')

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
    r = ra_module.NamedRelationalAlgebraFrozenSet(['y', 'x'], r_unnamed)
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
    assert ras_x.columns == (0,)

    ras_xz = ras.projection_to_unnamed("x", "z")
    assert all((i % 2, i * 2) in ras_xz for i in range(5))
    assert tuple(ras_xz.columns) == (0, 1)

    ras_xx = ras.projection_to_unnamed("x", "x")
    assert all((i % 2, i % 2) in ras_xx for i in range(5))
    assert tuple(ras_xx.columns) == (0, 1)

    ras_ = ras.projection_to_unnamed()
    assert ras_.arity == 0
    assert len(ras_) > 0
    assert (
        ras.projection_to_unnamed() ==
        ra_module.RelationalAlgebraFrozenSet.dee()
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

    ras_0 = ras.selection(lambda t: t.x == 1 and t.y == 2)
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
    second = ra_module.NamedRelationalAlgebraFrozenSet(first.columns, first,)
    assert first == second
    for tuple_a, tuple_b in zip(first, second):
        assert tuple_a == tuple_b

    third = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x",), ra_module.NamedRelationalAlgebraFrozenSet(tuple())
    )

    assert len(third) == 0
    assert third.columns == ("x",)


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
        ["x", "y"], {'qq': lambda t: sum(t.w + t.z)}
    )

    assert new_set == expected_op3


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
        ("x", "z",), [(7, "a",), (9, "a",)]
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
        {"x": ra_module.RelationalAlgebraColumnStr("x")}
    )
    assert initial_set.projection("x") == new_set

    base_set = ra_module.NamedRelationalAlgebraFrozenSet(
        (1, 2), [(7, 8), (9, 2)]
    )

    new_set = base_set.extended_projection({
        "x": ra_module.RelationalAlgebraColumnInt(1),
        "y": ra_module.RelationalAlgebraColumnInt(2)
    })

    assert initial_set == new_set


def test_rename_columns(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "x"}) == first
    assert id(first.rename_columns({"x": "x"})) != id(first)
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "x"), [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "y", "y": "x"}) == second
    with pytest.raises(ValueError, match=r"non-existing columns: {'z'}"):
        first.rename_columns({"z": "w"})


def test_rename_columns_duplicates(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(0, 2), (0, 4)],
    )
    with pytest.raises(ValueError, match=r"Duplicated.*{'z'}"):
        first.rename_columns({"x": "z", "y": "z"})


def test_equality(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(0, 2), (0, 4)],
    )
    assert first == first
    second = ra_module.NamedRelationalAlgebraFrozenSet(
        ("y", "x"), [(0, 2), (0, 4)],
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
            ("x", "x"), [(0, 2), (0, 4)],
        )


def test_extended_projection_ra_string_expression_empty_relation(ra_module):
    # reported in GH387
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"], iterable=[],
    )
    eval_expressions = {"z": RelationalAlgebraStringExpression("(x / y)")}
    expected = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["z"], iterable=[],
    )
    assert relation.extended_projection(eval_expressions) == expected


def test_aggregate_repeated_group_column(ra_module):
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x", "y"], iterable=[("a", 4), ("b", 5)],
    )
    with pytest.raises(ValueError, match="Cannot group on repeated columns"):
        relation.aggregate(["x", "x"], {"y": sum})


def test_unsupported_aggregation_function(ra_module):
    relation = ra_module.NamedRelationalAlgebraFrozenSet(
        columns=["x"], iterable=[("a",), ("b",)],
    )
    with pytest.raises(ValueError, match="Unsupported aggregate_function"):
        relation.aggregate(["x"], None)
