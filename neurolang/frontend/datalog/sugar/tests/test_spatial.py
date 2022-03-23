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
MUL = Constant(operator.mul)
P = Symbol("P")
EUCLIDEAN = Symbol("EUCLIDEAN")
i1, j1, k1 = Symbol("i1"), Symbol("j1"), Symbol("k1")
i2, j2, k2 = Symbol("i2"), Symbol("j2"), Symbol("k2")
FirstRange = Symbol("FirstRange")
SecondRange = Symbol("SecondRange")
d = Symbol("d")
s = Symbol("s")


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

def test_bound_with_symbol_in_table():
    import pandas as pd
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()

    Vol1 = pd.DataFrame([[1, 2, 3, 6.8], [2, 2, 2, 0.2], [7, 4, 3, 8.5]], columns=['x', 'y', 'z', 'dist'])
    Vol2 = pd.DataFrame([[1, 5, 9], [2, 2, 1], [15, 6, 2]], columns=['x', 'y', 'z'])

    Vol1s = nl.add_tuple_set(Vol1.values, name='Vol1s')
    Vol2s = nl.add_tuple_set(Vol2.values, name='Vol2s')

    with nl.scope as e:
        e.res[e.x1, e.y1, e.z1] = (
            Vol1s[e.x2, e.y2, e.z2, e.dist]
            & Vol2s(e.x1, e.y1, e.z1)
            & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
            & (e.d < e.dist)
        )

        a1 = nl.query((e.x1, e.y1, e.z1), e.res[e.x1, e.y1, e.z1])

    with nl.scope as e:
        e.res[e.x1, e.y1, e.z1] = (
            Vol1s[e.x2, e.y2, e.z2, e.dist]
            & Vol2s(e.x1, e.y1, e.z1)
            & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
            & (e.d < 1)
        )

        a2 = nl.query((e.x1, e.y1, e.z1), e.res[e.x1, e.y1, e.z1])

    with nl.scope as e:
        e.res[e.x1, e.y1, e.z1] = (
            Vol1s[e.x2, e.y2, e.z2, e.dist]
            & Vol2s(e.x1, e.y1, e.z1)
            & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
            & (e.d < 7)
        )

        a3 = nl.query((e.x1, e.y1, e.z1), e.res[e.x1, e.y1, e.z1])

    a1 = a1.as_pandas_dataframe().values
    a2 = a2.as_pandas_dataframe().values
    a3 = a3.as_pandas_dataframe().values
    assert len(a1) == 3
    assert len(a2) == 1 and [2, 2, 1] in a2
    assert len(a3) == 2 and [2, 2, 1] in a3 and [1, 5, 9] in a3


def test_bound_with_symbol_in_table_inverted():
    import pandas as pd
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()

    Vol1 = pd.DataFrame([[1, 2, 3], [2, 2, 2], [7, 4, 3]], columns=['x', 'y', 'z'])
    Vol2 = pd.DataFrame([[1, 5, 9, 6.8], [2, 2, 1, 2]], columns=['x', 'y', 'z', 'dist'])

    Vol1s = nl.add_tuple_set(Vol1.values, name='Vol1s')
    Vol2s = nl.add_tuple_set(Vol2.values, name='Vol2s')

    with nl.scope as e:
        e.res[e.x1, e.y1, e.z1] = (
            Vol1s[e.x2, e.y2, e.z2]
            & Vol2s(e.x1, e.y1, e.z1, e.dist)
            & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
            & (e.d < e.dist)
        )

        a1 = nl.query((e.x1, e.y1, e.z1), e.res[e.x1, e.y1, e.z1])

    with nl.scope as e:
        e.res[e.x1, e.y1, e.z1] = (
            Vol1s[e.x2, e.y2, e.z2]
            & Vol2s(e.x1, e.y1, e.z1, e.dist)
            & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
            & (e.d < 2)
        )

        a2 = nl.query((e.x1, e.y1, e.z1), e.res[e.x1, e.y1, e.z1])

    a1 = a1.as_pandas_dataframe().values
    a2 = a2.as_pandas_dataframe().values
    assert len(a1) == 2 and [2, 2, 1] in a1 and [1,5,9] in a1
    assert len(a2) == 1 and [2, 2, 1] in a2




