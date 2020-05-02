import pytest

from ..relational_algebra_set import pandas


@pytest.fixture(params=[(pandas,)])
def ra_module(request):
    return request.param[0]


def test_relational_algebra_set_semantics_empty(ra_module):
    ras = ra_module.RelationalAlgebraSet()

    assert len(ras) == 0
    assert ras.is_empty()
    assert ras.arity == 0
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
    ras__ = set((e, ) for e in a)

    assert ras == ras_
    assert ras == ras__

    assert len(ras) == len(a) - 1
    ras.discard(5)
    assert 5 not in ras
    assert len(ras) == len(a) - 2
    ras.add(10)
    assert len(ras) == len(a) - 1
    assert 10 in ras
    assert all(a_ in ras for a_ in a if a_ != 5)
    assert ras.fetch_one() in ras__

    dee = ra_module.RelationalAlgebraSet.dee()
    dum = ra_module.RelationalAlgebraSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0


def test_relational_algebra_ra_projection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    assert (0, ) in ras_0 and (1, ) in ras_0
    assert len(ras_0) == 2

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


def test_relational_algebra_ra_equijoin(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, i, i * 2) for i in range(5)]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)
    ras_d = ra_module.RelationalAlgebraSet(d)

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c

    res = ras_a.equijoin(ras_a, [(0, 0)])
    assert res == ras_d


def test_relational_algebra_ra_cross_product(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = ra_module.RelationalAlgebraSet(a)
    ras_b = ra_module.RelationalAlgebraSet(b)
    ras_c = ra_module.RelationalAlgebraSet(c)

    res = ras_a.cross_product(ras_b)
    assert res == ras_c


def test_relational_algebra_ra_equijoin_mixed_types(ra_module):
    a = [(chr(ord('a') + i), i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(chr(ord('a') + i), i * 2, i * 2, i * 3) for i in range(5)]

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

    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    dum = ra_module.NamedRelationalAlgebraFrozenSet.dum()

    assert len(dee) > 0 and dee.arity == 0
    assert len(dum) == 0 and dum.arity == 0
    assert dee != dum


def test_named_relational_algebra_set_semantics_empty(ra_module):
    ras = ra_module.NamedRelationalAlgebraFrozenSet(('y', 'x'))

    assert ras.is_empty()
    assert len(ras) == 0
    assert ras.arity == 2
    assert list(iter(ras)) == []
    assert ras != ra_module.NamedRelationalAlgebraFrozenSet.dum()
    assert ras.projection() == ra_module.NamedRelationalAlgebraFrozenSet.dum()

    ras = ra_module.NamedRelationalAlgebraFrozenSet(('y', 'x'), [(0, 1)])
    assert (0, 1) in ras
    assert {'x': 1, 'y': 0} in ras
    assert {'y': 1, 'x': 1} not in ras
    assert len(ras) == 1
    assert ras.arity == 2

    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(('y', 'z'), [(0, 1)])
    assert ras != ras_b

    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), [(1, 0)])
    assert ras == ras_c


def test_named_relational_algebra_ra_projection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_x = ras.projection('x')
    assert (0, ) in ras_x and (1, ) in ras_x
    assert len(ras_x) == 2
    assert ras_x.columns == ('x', )

    ras_xz = ras.projection('x', 'z')
    assert all((i % 2, i * 2) in ras_xz for i in range(5))

    ras_ = ras.projection()
    assert ras_.arity == 0
    assert len(ras_) > 0
    assert ras_.projection('x') == ras_

    assert ras_.projection() == ra_module.NamedRelationalAlgebraFrozenSet.dee()


def test_named_relational_algebra_ra_selection(ra_module):
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_0 = ras.selection({'x': 1})
    a_sel = ra_module.NamedRelationalAlgebraFrozenSet(
        ras.columns, set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    )
    assert ras_0 == a_sel

    ras_0 = ras.selection({'x': 1, 'y': 2})
    a_sel = ra_module.NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2)
    )
    assert ras_0 == a_sel


def test_named_relational_algebra_ra_naturaljoin(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, j * 2, j * 3) for i in range(5) for j in range(5)]

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(('z', 'y'), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(('y', 'x'), b)
    ras_b2 = ra_module.NamedRelationalAlgebraFrozenSet(('u', 'v'), b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(('z', 'y', 'x'), c)
    ras_d = ra_module.NamedRelationalAlgebraFrozenSet(('z', 'y', 'u', 'v'), d)
    empty = ra_module.NamedRelationalAlgebraFrozenSet(('z', 'y'), [])
    empty_plus = ra_module.NamedRelationalAlgebraFrozenSet(
        ('z', 'y'), [(0, 1)]
    ).projection()

    assert len(ras_a.naturaljoin(empty)) == 0
    assert len(empty.naturaljoin(ras_a)) == 0
    assert ras_a.naturaljoin(empty_plus) == ras_a
    assert empty_plus.naturaljoin(ras_a) == ras_a

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

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(('u', 'v'), b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y', 'u', 'v'), c)

    res = ras_a.cross_product(ras_b)
    assert res == ras_c


def test_named_relational_algebra_difference(ra_module):
    a = [(i, i * 2) for i in range(5)]
    b = [(i, i * 2) for i in range(1, 5)]
    c = [(i, i * 2) for i in range(1)]

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), b)
    ras_b_inv = ra_module.NamedRelationalAlgebraFrozenSet(('y', 'x'),
                                                [t[::-1] for t in b])
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), c)

    empty = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
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

    cols = ('x', 'y')

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ra_module.NamedRelationalAlgebraFrozenSet(cols, b)
    ras_c = ra_module.NamedRelationalAlgebraFrozenSet(cols, c)

    res = list(ras_a.groupby('x'))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)


def test_named_iter_and_fecth_one(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res


def test_rename_column(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ras_a.rename_column('y', 'z')
    assert all(
        el_a.x == el_b.x and el_a.y == el_b.z
        for el_a, el_b in zip(ras_a, ras_b)
    )


def test_named_to_unnamed(ra_module):
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = ra_module.NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ra_module.RelationalAlgebraFrozenSet(a)
    assert ras_a.to_unnamed() == ras_b


def test_named_ra_set_from_other(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(("x", "n"), [
        (56, "bonjour"),
        (42, "aurevoir"),
    ])
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


def test_named_ra_union(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2)])
    second = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2), (42, 0)])
    expected = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2),
                                                            (42, 0)])
    assert first | second == expected
    empty = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
    dee = ra_module.NamedRelationalAlgebraFrozenSet.dee()
    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | empty | second == first | second


def test_named_ra_intersection(ra_module):
    first = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2)])
    second = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2), (42, 0)])
    expected = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2)])
    assert first & second == expected
    empty = ra_module.NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty


def test_aggregate(ra_module):
    initial_set = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"), [(7, 8, 1),
                                                                    (7, 8, 9)])
    expected_sum = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                   [(7, 8, 10)])
    expected_str = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                   [(7, 8, 2)])
    expected_lambda = ra_module.NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                      [(7, 8, 8)])

    initial_set2 = ra_module.NamedRelationalAlgebraFrozenSet(("w", "x", "y", "z"),
                                                   [(1, 7, 8, 1),
                                                    (2, 7, 8, 9)])
    expected_op2 = ra_module.NamedRelationalAlgebraFrozenSet(("w", "x", "y", "z"),
                                                   [(2, 7, 8, 8)])

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": "count"})
    assert expected_str == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": lambda x: max(x) - 1})
    assert expected_lambda == new_set
    new_set = initial_set.aggregate(
        ["x", "y"],
        [
            ("x", "x", lambda x: next(iter(x))),
            ("y", "y", lambda x: next(iter(x))),
            ("z", "z", lambda x: max(x) - 1)
        ]
    )
    assert expected_lambda == new_set
    new_set = initial_set2.aggregate(["x", "y"], {
        "z": lambda x: max(x) - 1,
        "w": "count"
    })
    assert expected_op2 == new_set


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
