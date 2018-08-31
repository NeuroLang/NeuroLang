import pytest
from .. import relational_algebra_set as ras


def test_empty():
    a = ras.RelationalAlgebraSet()
    assert len(a) == 0

    assert 1 not in a
    assert (1,) not in a


def test_single_column():
    data = set((i,) for i in range(5))
    a = ras.RelationalAlgebraSet(data)

    assert len(a) == 5
    assert all(e in a for e in data)
    assert set(iter(a)) == data

    a.add((100,))
    assert len(a) == 6
    assert (100,) in a
    assert a & data == data


def test_multiple_columns():
    data = set((i, 2 * i) for i in range(5))
    a = ras.RelationalAlgebraSet(data)

    assert len(a) == 5
    assert all(e in a for e in data)
    assert set(iter(a)) == data

    a.add((100, 200))
    assert len(a) == 6
    assert (100, 200) in a


@pytest.mark.skip(msg="This should fail but tuple size is not validated")
def test_multiple_columns_different_sizes():
    data = set((str(i), str(2 * i)) for i in range(5))
    data.add(('a',))

    a = ras.RelationalAlgebraSet(data)

    assert len(a) == 6
    assert all(e in a for e in data)
    assert set(iter(a)) == data


def test_project():
    data = set((i, 2 * i) for i in range(5))
    a = ras.RelationalAlgebraSet(data)

    b = a.project(0)
    assert b == set((e[0],) for e in data)
    assert all(i == j for i, j in zip(b._set.columns, range(1)))

    b = a.project((0,))
    assert b == set((e[0],) for e in data)
    assert all(i == j for i, j in zip(b._set.columns, range(1)))

    b = a.project((0, 1))
    assert b == data
    assert all(i == j for i, j in zip(b._set.columns, range(2)))


def test_select_equality():
    data = set((i, 2 * i) for i in range(5))
    data |= set((i, 3 * i) for i in range(5))

    a = ras.RelationalAlgebraSet(data)

    b = a.select_equality({0: 1})
    assert b == {(1, 2), (1, 3)}
    assert all(i == j for i, j in zip(b._set.columns, range(2)))


def test_select_columns():
    data = set((i, 2 * i) for i in range(5))
    data |= set((i, 3 * i) for i in range(5))

    a = ras.RelationalAlgebraSet(data)

    b = a.select_columns({0: 1, 1: 0})
    assert set(iter(b)) == {(0, 0)}
    assert all(i == j for i, j in zip(b._set.columns, range(2)))


def test_join_columns():
    a = set((i, 2 * i) for i in range(5))
    b = set((i, 3 * i) for i in range(5))

    a = ras.RelationalAlgebraSet(a)
    b = ras.RelationalAlgebraSet(b)

    c = a.join_by_columns(b, (1,), (0,))
    assert c == {
        a_ + b_
        for a_ in a
        for b_ in b
        if a_[1] == b_[0]
    }
    assert all(i == j for i, j in zip(c._set.columns, range(4)))

    d = a.join_by_columns(b, (1, 0), (0, 1))
    assert d == {
        a_ + b_
        for a_ in a
        for b_ in b
        if (
            a_[1] == b_[0] and
            a_[0] == b_[1]
        )
    }

    assert all(i == j for i, j in zip(d._set.columns, range(4)))


def test_ior():
    a = ras.RelationalAlgebraSet([(0,)])
    b = ras.RelationalAlgebraSet([(1,)])

    a |= b

    assert a == {(0,), (1,)}


def test_iand():
    a = ras.RelationalAlgebraSet([(0,), (1,)])
    b = ras.RelationalAlgebraSet([(1,)])

    a &= b

    assert a == {(1,)}
