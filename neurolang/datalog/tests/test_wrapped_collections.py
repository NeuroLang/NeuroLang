
from typing import AbstractSet, Tuple

from ..expressions import Constant
from ..wrapped_collections import WrappedRelationalAlgebraSet

R1 = WrappedRelationalAlgebraSet([
    (i, i * 2)
    for i in range(10)
])

R2 = WrappedRelationalAlgebraSet([
    (i * 2, i * 3)
    for i in range(10)
])


C_ = Constant


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
