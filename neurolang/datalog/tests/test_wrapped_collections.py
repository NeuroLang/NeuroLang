from typing import Tuple

import numpy as np
from pytest import mark

from ..expressions import Constant
from ..wrapped_collections import (NamedRelationalAlgebraFrozenSet,
                                   RelationalAlgebraFrozenSet,
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
        (i * 2, str(i * 3))
        for i in range(10)
    ]
)

C_ = Constant


@mark.xfail(reason="Need to implement type mappings between RA sets and python")
def test_row_types():
    assert R2.row_type == Tuple[int, int]
    assert R3.columns == ('x', 'y')
    assert R3.row_type == Tuple[int, str]


def test_init_from_wrapped():
    r1 = WrappedRelationalAlgebraSet(R1)

    assert r1.row_type == R1.row_type
    assert set(r1) == set(R1)


def test_init_named_from_wrapped():
    r3 = WrappedNamedRelationalAlgebraFrozenSet(iterable=R3)
    assert r3.row_type == R3.row_type
    assert r3.columns == R3.columns
    assert r3 == R3
    assert set(r3) == set(R3)


def test_init_named_from_iterator_and_collection():
    it = ((i,) for i in range(5))
    col = [(i,) for i in range(5)]
    r = WrappedRelationalAlgebraSet(iterable=it)
    assert r.arity == 1
    assert len(r) == 5
    assert set(r.unwrap()) == set(col)

    r = WrappedRelationalAlgebraSet(iterable=col)
    assert r.arity == 1
    assert len(r) == 5
    assert set(r.unwrap()) == set(col)


def test_unwrap():
    r2_u = R2.unwrap()
    r3_u = R3.unwrap()

    assert not isinstance(r2_u, WrappedRelationalAlgebraSet)
    assert isinstance(r2_u, RelationalAlgebraFrozenSet)
    assert r2_u == set(R2.unwrapped_iter())

    assert not isinstance(r3_u, WrappedNamedRelationalAlgebraFrozenSet)
    assert isinstance(r3_u, NamedRelationalAlgebraFrozenSet)
    r3_u_expected = NamedRelationalAlgebraFrozenSet(
        columns=('x', 'y'),
        iterable=[
            (i * 2, str(i * 3))
            for i in range(10)
        ]
    )
    assert r3_u == r3_u_expected
    assert r3_u.columns == r3_u_expected.columns


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


def test_update_difference():
    r1 = WrappedRelationalAlgebraSet([(1, 2), (7, 8)])
    r2 = WrappedRelationalAlgebraSet([(5, 0), (7, 8)])
    r1 -= r2
    assert r1 == WrappedRelationalAlgebraSet([(1, 2)])


def test_create_from_array():
    r1 = WrappedRelationalAlgebraSet(np.eye(2, dtype=int))

    assert len(r1) == 2
    assert {(1, 0), (0, 1)} == r1.unwrap()


def test_type_inference():
    items = {(0.2, 1), (0.8, 42)}
    r = WrappedRelationalAlgebraSet(items)
    assert r.row_type == Tuple[float, int]
