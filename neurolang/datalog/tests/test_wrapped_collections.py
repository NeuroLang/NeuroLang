
from ..expressions import Constant
from ..wrapped_collections import (RelationalAlgebraFrozenSet,
                                   NamedRelationalAlgebraFrozenSet,
                                   WrappedNamedRelationalAlgebraFrozenSet,
                                   WrappedRelationalAlgebraSet)

R1 = WrappedRelationalAlgebraSet([
    (i, i * 2)
    for i in range(10)
])

R2 = WrappedRelationalAlgebraSet([
    (i * 2, i * 3)
    for i in range(10)
])

R3 = WrappedNamedRelationalAlgebraFrozenSet(
    columns=('x', 'y'),
    iterable=[
        (i * 2, i * 3)
        for i in range(10)
    ]
)

C_ = Constant


def test_init_from_wrapped():
    r1 = WrappedRelationalAlgebraSet(R1)

    assert r1.row_type == R1.row_type
    assert set(r1) == set(R1)


def test_init_named_from_wrapped():
    rt = R3.row_type
    r3 = WrappedNamedRelationalAlgebraFrozenSet(iterable=R3)
    assert r3.row_type == R3.row_type
    assert r3.columns == R3.columns
    assert set(r3) == set(R3)


def test_unwrap():
    r2_u = R2.unwrap()
    r3_u = R3.unwrap()

    assert not isinstance(r2_u, WrappedRelationalAlgebraSet)
    assert isinstance(r2_u, RelationalAlgebraFrozenSet)
    assert r2_u == set(R2.unwrapped_iter())

    assert not isinstance(r3_u, WrappedNamedRelationalAlgebraFrozenSet)
    assert isinstance(r3_u, NamedRelationalAlgebraFrozenSet)
    assert r3_u == set(R3.unwrapped_iter())


def test_contains():
    assert C_((0, 0)) in R1
    assert not(C_((-1, -1)) in R1)


def test_equal():
    r1_ = WrappedRelationalAlgebraSet([
        (i, i * 2)
        for i in range(10)
    ])
    r1__ = {C_((i, i * 2)) for i in range(10)}
    assert R1 == r1_
    assert not (R1 == R2)
    assert not (R2 == R1)
    assert R1 == r1__


def test_intersection():
    r1 = WrappedRelationalAlgebraSet([(1, 2), (7, 8)])
    r2 = WrappedRelationalAlgebraSet([(5, 0), (7, 8)])
    res = r1 & r2
    assert res == WrappedRelationalAlgebraSet({(7, 8)})


def test_union():
    r1 = WrappedRelationalAlgebraSet([(1, 2), (7, 8)])
    r2 = WrappedRelationalAlgebraSet([(5, 0), (7, 8)])
    res = r1 | r2
    assert res == WrappedRelationalAlgebraSet([(1, 2), (7, 8), (5, 0)])
