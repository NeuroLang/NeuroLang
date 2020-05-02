from collections import namedtuple

from pytest import fixture, raises

from ..relational_algebra_set import pandas, sql


@fixture(
    params=(
        {
            'frozen': pandas.RelationalAlgebraFrozenSet,
            'mutable': pandas.RelationalAlgebraSet,
            'named': pandas.NamedRelationalAlgebraFrozenSet,
            'expression': pandas.RelationalAlgebraExpression,
            'operators': pandas.column_operators
        },
        {
            'frozen': sql.RelationalAlgebraFrozenSet,
            'mutable': sql.RelationalAlgebraSet,
            'named': sql.NamedRelationalAlgebraFrozenSet,
            'expression': sql.RelationalAlgebraExpression,
            'operators': sql.column_operators
        },
    ),
    ids=['pandas', 'sql']
)
def ras_class(request):
    print(request)
    return request.param


def test_relational_algebra_set_semantics_empty(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    ras = RelationalAlgebraSet()

    assert len(ras) == 0
    assert ras.arity == 0
    assert list(iter(ras)) == []

    ras.add((0, 1))
    assert (0, 1) in ras
    assert len(ras) == 1
    assert ras.arity == 2


def test_relational_algebra_set_semantics(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [5, 4, 3, 2, 3, 1]
    ras = RelationalAlgebraSet(a)
    ras_ = RelationalAlgebraSet(a)
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

    ras = RelationalAlgebraSet(a)
    ras_ = RelationalAlgebraSet([5, 4])
    ras -= ras_
    assert len(ras) == (len(set(a)) - 2)
    assert (5 not in ras) and (4 not in ras)


def test_object_column(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    T = namedtuple('T', ['x'])
    t = T(0)
    a = set([(t,)])
    ras = RelationalAlgebraSet(a)

    assert len(ras) == 1
    assert ras == a

    class T1:
        def __init__(self, a):
            self.a = a

        def __eq__(self, other):
            return self.a == other.a

        def __hash__(self):
            return hash(self.a)

    t = T1(0)
    a = set([(t,)])
    ras = RelationalAlgebraSet(a)

    assert len(ras) == 1
    assert ras == a


def test_relational_algebra_ra_projection(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    len(ras_0)
    assert (0, ) in ras_0 and (1, ) in ras_0
    assert len(ras_0) == 2

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))

    ras_null = ras.projection()
    assert ras_null.arity == 0 and len(ras_null) > 0

    ras_null2 = RelationalAlgebraSet().projection()
    assert ras_null2.arity == 0 and len(ras_null2) == 0


def test_relational_algebra_ra_selection(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = RelationalAlgebraSet(a)

    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2)
    assert ras_0 == a_sel

    ras_1 = ras.selection({0: 'a'})
    assert len(ras_1) == 0

    assert len(ras_1.selection({0: 1})) == 0

    ras = RelationalAlgebraSet()
    ras_1 = ras.selection({0: 1})
    assert len(ras_1) == 0 & ras_1.arity == 0


def test_relational_algebra_ra_selection_columns(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = RelationalAlgebraSet(a)

    ras_0 = ras.selection_columns({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == i)
    assert ras_0 == a_sel

    ras = RelationalAlgebraSet()
    ras_1 = ras.selection_columns({0: 1})
    assert len(ras_1) == 0 & ras_1.arity == 0


def test_relational_algebra_ra_equijoin(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, i, i * 2) for i in range(5)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)
    ras_d = RelationalAlgebraSet(d)
    ras_null = RelationalAlgebraSet()

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c

    res = ras_a.equijoin(ras_a, [(0, 0)])
    assert res == ras_d

    res = ras_a.equijoin(ras_null, [(0, 0)])
    assert len(res) == 0 and res.arity == 0

    res = ras_null.equijoin(ras_a, [(0, 0)])
    assert len(res) == 0 and res.arity == 0


def test_relational_algebra_ra_cross_product(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)
    ras_null = RelationalAlgebraSet()

    res = ras_a.cross_product(ras_b)
    assert res == ras_c

    res = ras_a.cross_product(ras_null)
    assert len(res) == 0 and res.arity == 0

    res = ras_null.cross_product(ras_a)
    assert len(res) == 0 and res.arity == 0


def test_relational_algebra_ra_equijoin_mixed_types(ras_class):
    RelationalAlgebraSet = ras_class['mutable']

    a = [(chr(ord('a') + i), i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(chr(ord('a') + i), i * 2, i * 2, i * 3) for i in range(5)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c


def test_relational_algebra_null_operator(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(chr(ord('a') + i), i * 2) for i in range(5)]
    ras = RelationalAlgebraSet(a)

    ras_null = ras.projection()
    assert ras_null.arity == 0
    assert len(ras_null) > 0


def test_groupby(ras_class):
    RelationalAlgebraSet = ras_class['mutable']

    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = list(ras_a.groupby(0))
    assert res[0] == ((1,), ras_b)
    assert res[1] == ((2,), ras_c)


def test_named_relational_algebra_set_semantics_empty(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    RelationalAlgebraSet = ras_class['mutable']

    ras = NamedRelationalAlgebraFrozenSet(('y', 'x'))

    assert len(ras) == 0
    assert ras.arity == 2
    assert list(iter(ras)) == []

    ras = NamedRelationalAlgebraFrozenSet(('y', 'x'), [(0, 1)])
    assert (0, 1) in ras
    assert {'x': 1, 'y': 0} in ras
    assert {'y': 1, 'x': 1} not in ras
    assert len(ras) == 1
    assert ras.arity == 2

    ras = RelationalAlgebraSet([(0, 1)]).projection()
    ras_n = NamedRelationalAlgebraFrozenSet([], ras)
    return ras_n.arity == 0 and len(ras_n) > 0


def test_named_relational_algebra_ra_projection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_x = ras.projection('x')
    assert (0, ) in ras_x and (1, ) in ras_x
    assert len(ras_x) == 2
    assert ras_x.columns == ('x', )

    ras_xz = ras.projection('x', 'z')
    assert all((i % 2, i * 2) in ras_xz for i in range(5))

    ras_null = ras.projection()
    assert ras_null.arity == 0 and len(ras_null) > 0

    ras_null2 = NamedRelationalAlgebraFrozenSet(columns=['a']).projection()
    assert ras_null2.arity == 0 and len(ras_null2) == 0

    ras_ = ras.projection()
    assert ras_.arity == 0
    assert len(ras_) > 0
    assert ras_.projection('x') == ras_


def test_named_relational_algebra_ra_selection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_0 = ras.selection({'x': 1})
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns, set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    )
    assert ras_0 == a_sel

    ras_0 = ras.selection({'x': 1, 'y': 2})
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1 and i == 2)
    )
    assert ras_0 == a_sel


def test_named_relational_algebra_ra_naturaljoin(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 3) for i in range(5)]
    d = [(i, i * 2, j * 2, j * 3) for i in range(5) for j in range(5)]

    ras_a = NamedRelationalAlgebraFrozenSet(('z', 'y'), a)
    ras_b = NamedRelationalAlgebraFrozenSet(('y', 'x'), b)
    ras_b2 = NamedRelationalAlgebraFrozenSet(('u', 'v'), b)
    ras_c = NamedRelationalAlgebraFrozenSet(('z', 'y', 'x'), c)
    ras_d = NamedRelationalAlgebraFrozenSet(('z', 'y', 'u', 'v'), d)
    empty = NamedRelationalAlgebraFrozenSet(('z', 'y'), [])
    empty_plus = NamedRelationalAlgebraFrozenSet(
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


def test_named_relational_algebra_ra_cross_product(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = NamedRelationalAlgebraFrozenSet(('x', 'y'), a)
    ras_b = NamedRelationalAlgebraFrozenSet(('u', 'v'), b)
    ras_c = NamedRelationalAlgebraFrozenSet(('x', 'y', 'u', 'v'), c)

    res = ras_a.cross_product(ras_b)
    assert res == ras_c


def test_named_relational_algebra_difference(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i, i * 2) for i in range(5)]
    b = [(i, i * 2) for i in range(1, 5)]
    c = [(i, i * 2) for i in range(1)]

    ras_a = NamedRelationalAlgebraFrozenSet(('x', 'y'), a)
    ras_b = NamedRelationalAlgebraFrozenSet(('x', 'y'), b)
    ras_b_inv = NamedRelationalAlgebraFrozenSet(('y', 'x'),
                                                [t[::-1] for t in b])
    ras_c = NamedRelationalAlgebraFrozenSet(('x', 'y'), c)

    empty = NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
    unit_empty = NamedRelationalAlgebraFrozenSet(
        ('x', 'y'), [(0, 1)]
    ).projection()

    assert (ras_a - empty) == ras_a
    assert (empty - ras_a) == empty
    assert (empty - empty) == empty
    assert (unit_empty - empty) == unit_empty
    assert (unit_empty - unit_empty) == NamedRelationalAlgebraFrozenSet(())

    res = ras_a - ras_b
    assert res == ras_c

    res = ras_b - ras_a
    assert len(res) == 0

    res = ras_a - ras_b_inv
    assert res == ras_c

    res = ras_b_inv - ras_a
    assert len(res) == 0

    res = ras_a - NamedRelationalAlgebraFrozenSet(
        columns=ras_a.columns, iterable=[]
    )
    assert ras_a == res


def test_named_groupby(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    cols = ('x', 'y')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = NamedRelationalAlgebraFrozenSet(cols, b)
    ras_c = NamedRelationalAlgebraFrozenSet(cols, c)

    res = list(ras_a.groupby('x'))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)


def test_named_iter(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a
    assert ras_a.fetch_one() in res


def test_rename_column(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = ras_a.rename_column('y', 'z')
    assert all(
        el_a.x == el_b.x and el_a.y == el_b.z
        for el_a, el_b in zip(ras_a, ras_b)
    )


def test_named_to_unnamed(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    RelationalAlgebraFrozenSet = ras_class['frozen']
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]

    cols = ('y', 'x')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = RelationalAlgebraFrozenSet(a)
    assert ras_a.to_unnamed() == ras_b

    ras_zero = NamedRelationalAlgebraFrozenSet(columns=cols, iterable=[])
    ras_zero_un = ras_zero.to_unnamed()
    assert ras_zero_un.arity == 2
    assert len(ras_zero_un) == 0


def test_named_ra_set_from_other(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    first = NamedRelationalAlgebraFrozenSet(("x", "n"), [
        (56, "bonjour"),
        (42, "aurevoir"),
    ])
    second = NamedRelationalAlgebraFrozenSet(
        first.columns,
        first,
    )
    assert first == second
    for tuple_a, tuple_b in zip(first, second):
        assert tuple_a == tuple_b

    third = NamedRelationalAlgebraFrozenSet(
        ("x",), NamedRelationalAlgebraFrozenSet(tuple())
    )

    assert len(third) == 0
    assert third.columns == ("x",)


def test_named_ra_union(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    first = NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2)])
    second = NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2), (42, 0)])
    expected = NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2),
                                                            (42, 0)])
    assert first | second == expected
    empty = NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
    dee = NamedRelationalAlgebraFrozenSet.dee()
    assert first | empty == first
    assert empty | first == first
    assert dee | dee == dee
    assert first | empty | second == first | second


def test_named_ra_intersection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    first = NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2)])
    second = NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2), (42, 0)])
    expected = NamedRelationalAlgebraFrozenSet(("x", "y"), [(9, 2)])
    assert first & second == expected
    empty = NamedRelationalAlgebraFrozenSet(('x', 'y'), [])
    assert first & empty == empty
    assert empty & first == empty
    assert first & empty & second == empty

    empty = NamedRelationalAlgebraFrozenSet(columns=tuple())
    assert first & empty == empty
    assert empty & first == empty


def test_aggregate(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    operators = ras_class['operators']

    initial_set = NamedRelationalAlgebraFrozenSet(("x", "y", "z"), [(7, 8, 1),
                                                                    (7, 8, 9)])
    expected_sum = NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                   [(7, 8, 10)])
    expected_str = NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                   [(7, 8, 2)])
    expected_lambda = NamedRelationalAlgebraFrozenSet(("x", "y", "z"),
                                                      [(7, 8, 8)])

    initial_set2 = NamedRelationalAlgebraFrozenSet(("w", "x", "y", "z"),
                                                   [(1, 7, 8, 1),
                                                    (2, 7, 8, 9)])
    expected_op2 = NamedRelationalAlgebraFrozenSet(("w", "x", "y", "z"),
                                                   [(2, 7, 8, 8)])

    new_set = initial_set.aggregate(["x", "y"], {"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.aggregate(["x", "y"], {"z": "count"})
    assert expected_str == new_set
    new_set = initial_set.aggregate(
        ["x", "y"], {"z": lambda x: operators.max(x) - 1}
    )
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
        "z": lambda x: operators.max(x) - 1,
        "w": "count"
    })
    assert expected_op2 == new_set


def test_mutable_built_from_frozen(ras_class):
    RelationalAlgebraFrozenSet = ras_class['frozen']
    RelationalAlgebraSet = ras_class['mutable']

    rafs = RelationalAlgebraFrozenSet([0, 1, 2, 3])
    ras = RelationalAlgebraSet(rafs)

    assert rafs == ras
    ras.discard(0)
    assert 0 not in ras
    assert len(rafs) - 1 == len(ras)


def test_extended_projection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    RelationalAlgebraStringExpression = ras_class['expression']

    initial_set = NamedRelationalAlgebraFrozenSet(("x", "y"), [(7, 8), (9, 2)])
    expected_sum = NamedRelationalAlgebraFrozenSet(("z",), [(15,), (11,)])
    expected_lambda = NamedRelationalAlgebraFrozenSet(("z",), [(14,), (10,)])
    expected_lambda2 = NamedRelationalAlgebraFrozenSet(
        ("z", "x"), [(14, 8), (10, 10)]
    )
    expected_new_colum_str = NamedRelationalAlgebraFrozenSet(
        ("x", "z",), [(7, "a",), (9, "a",)]
    )
    expected_new_colum_int = NamedRelationalAlgebraFrozenSet(
        ("z",), [(1,), (1,)]
    )
    new_set = initial_set.extended_projection({"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.extended_projection(
        {"z": RelationalAlgebraStringExpression("x+y")}
    )
    assert expected_sum == new_set
    new_set = initial_set.extended_projection({"z": lambda r: r.x + r.y - 1})
    assert expected_lambda == new_set
    new_set = initial_set.extended_projection(
        {
            "z": lambda r: (r.x + r.y - 1),
            "x": RelationalAlgebraStringExpression("x+1"),
        }
    )
    assert expected_lambda2 == new_set
    new_set = initial_set.extended_projection(
        {"z": "a", "x": RelationalAlgebraStringExpression("x")}
    )
    assert expected_new_colum_str == new_set
    new_set = initial_set.extended_projection({"z": 1})
    assert expected_new_colum_int == new_set


def test_rename_columns(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']
    first = NamedRelationalAlgebraFrozenSet(
        ("x", "y"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "x"}) == first
    assert id(first.rename_columns({"x": "x"})) != id(first)
    second = NamedRelationalAlgebraFrozenSet(
        ("y", "x"),
        [(0, 2), (0, 4)],
    )
    assert first.rename_columns({"x": "y", "y": "x"}) == second
    with raises(ValueError, match=r"non-existing columns: {'z'}"):
        first.rename_columns({"z": "w"})
