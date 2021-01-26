import operator

import pytest

from .....datalog.basic_representation import DatalogProgram
from .....exceptions import UnsupportedProgramError
from .....expressions import Constant, Symbol
from .....logic import Conjunction, Implication
from ..spatial import (
    DetectEuclideanDistanceBoundMatrix,
    TranslateEuclideanDistanceBoundMatrixMixin,
)

EQ = Constant(operator.eq)
LT = Constant(operator.lt)
LE = Constant(operator.le)
GT = Constant(operator.gt)
GE = Constant(operator.ge)
P = Symbol("P")
EUCLIDEAN = Symbol("EUCLIDEAN")
i1, j1, k1 = Symbol("i1"), Symbol("j1"), Symbol("k1")
i2, j2, k2 = Symbol("i2"), Symbol("j2"), Symbol("k2")
FirstRange = Symbol("FirstRange")
SecondRange = Symbol("SecondRange")
d = Symbol("d")


def test_euclidean_spatial_bound_detection():
    max_dist = Constant(2)
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                LT(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                LE(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                GE(max_dist, d),
            )
        ),
    )
    assert detector.walk(expression)


def test_reversed_equality():
    max_dist = Constant(10)
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                LT(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                GT(max_dist, d),
            )
        ),
    )
    assert detector.walk(expression)


def test_unsupported_unbounded_distance():
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
            )
        ),
    )
    with pytest.raises(UnsupportedProgramError):
        detector.walk(expression)


class ProgramTest(TranslateEuclideanDistanceBoundMatrixMixin, DatalogProgram):
    pass


def test_unsupported_non_extensional_safe_range_pred():
    max_dist = Constant(2)
    translator = ProgramTest()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                GT(max_dist, d),
            )
        ),
    )
    with pytest.raises(NotImplementedError):
        translator.walk(expression)
