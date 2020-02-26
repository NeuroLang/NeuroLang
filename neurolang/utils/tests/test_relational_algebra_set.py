from pytest import fixture

from ..relational_algebra_set import pandas, sql


@fixture(
    params=(
        {
            'frozen': pandas.RelationalAlgebraFrozenSet,
            'mutable': pandas.RelationalAlgebraSet,
            'named': pandas.NamedRelationalAlgebraFrozenSet,
        },
        {
            'frozen': sql.RelationalAlgebraFrozenSet,
            'mutable': sql.RelationalAlgebraSet,
            'named': sql.NamedRelationalAlgebraFrozenSet,
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

    assert len(ras) == len(a) - 1
    ras.discard(5)
    assert 5 not in ras
    assert len(ras) == len(a) - 2
    ras.add(10)
    assert len(ras) == len(a) - 1
    assert 10 in ras
    assert all(a_ in ras for a_ in a if a_ != 5)


def test_relational_algebra_ra_projection(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    assert (0,) in ras_0 and (1,) in ras_0
    assert len(ras_0) == 2

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))


def test_relational_algebra_ra_selection(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = RelationalAlgebraSet(a)

    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set(
        (i % 2, i, i * 2) for i in range(5)
        if i % 2 == 1 and i == 2
    )
    assert ras_0 == a_sel


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

    res = ras_a.equijoin(ras_b, [(1, 0)])
    assert res == ras_c

    res = ras_a.equijoin(ras_a, [(0, 0)])
    assert res == ras_d


def test_relational_algebra_ra_cross_product(ras_class):
    RelationalAlgebraSet = ras_class['mutable']
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [u + v for u in a for v in b]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = ras_a.cross_product(ras_b)
    assert res == ras_c


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


def test_groupby(ras_class):
    RelationalAlgebraSet = ras_class['mutable']

    a = [
        (i, i * j)
        for i in (1, 2)
        for j in (2, 3, 4)
    ]

    b = [(1, j) for j in (2, 3, 4)]
    c = [(2, 2 * j) for j in (2, 3, 4)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = list(ras_a.groupby(0))
    assert res[0] == (1, ras_b)
    assert res[1] == (2, ras_c)


def test_named_relational_algebra_set_semantics_empty(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

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


def test_named_relational_algebra_ra_projection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_x = ras.projection('x')
    assert (0,) in ras_x and (1,) in ras_x
    assert len(ras_x) == 2
    assert ras_x.columns == ('x',)

    ras_xz = ras.projection('x', 'z')
    assert all((i % 2, i * 2) in ras_xz for i in range(5))


def test_named_relational_algebra_ra_selection(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = NamedRelationalAlgebraFrozenSet(('x', 'y', 'z'), a)

    ras_0 = ras.selection({'x': 1})
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    )
    assert ras_0 == a_sel

    ras_0 = ras.selection({'x': 1, 'y': 2})
    a_sel = NamedRelationalAlgebraFrozenSet(
        ras.columns,
        set(
            (i % 2, i, i * 2) for i in range(5)
            if i % 2 == 1 and i == 2
        )
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
    ras_c = NamedRelationalAlgebraFrozenSet(('x', 'y'), c)

    res = ras_a - ras_b
    assert res == ras_c


def test_named_groupby(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [
        (i, i * j)
        for i in (1, 2)
        for j in (2, 3, 4)
    ]

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

    a = [
        (i, i * j)
        for i in (1, 2)
        for j in (2, 3, 4)
    ]

    cols = ('y', 'x')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    res = list(iter(ras_a))
    assert res == a


def test_rename_column(ras_class):
    NamedRelationalAlgebraFrozenSet = ras_class['named']

    a = [
        (i, i * j)
        for i in (1, 2)
        for j in (2, 3, 4)
    ]

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

    a = [
        (i, i * j)
        for i in (1, 2)
        for j in (2, 3, 4)
    ]

    cols = ('y', 'x')

    ras_a = NamedRelationalAlgebraFrozenSet(cols, a)
    ras_b = RelationalAlgebraFrozenSet(a)
    assert ras_a.to_unnamed() == ras_b
