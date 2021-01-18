import operator

from .....expressions import Constant, Symbol
from .....logic import Conjunction, Implication
from ..spatial import DetectEuclideanDistanceBoundMatrix


def test_euclidean_spatial_bound_detection():
    EQ = Constant(operator.eq)
    LT = Constant(operator.lt)
    P = Symbol("P")
    i1, j1, k1 = Symbol("i1"), Symbol("j1"), Symbol("k1")
    i2, j2, k2 = Symbol("i2"), Symbol("j2"), Symbol("k2")
    FirstRange = Symbol("FirstRange")
    SecondRange = Symbol("SecondRange")
    EUCLIDEAN = Symbol("EUCLIDEAN")
    d = Symbol("d")
    max_dist = Constant("10mm")
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
