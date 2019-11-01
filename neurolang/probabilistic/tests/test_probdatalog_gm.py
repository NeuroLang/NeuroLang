from typing import Mapping, AbstractSet, Tuple

import pytest
import pandas as pd
import numpy as np

from ...exceptions import NeuroLangException
from ..probdatalog_gm import (
    TranslateGroundedProbDatalogToGraphicalModel,
    AlgebraSet,
    bernoulli_vect_table_distrib,
    extensional_vect_table_distrib,
    ExtendedRelationalAlgebraSolver,
    _make_numerical_col_symb,
    _args_to_column_names,
)
from ..probdatalog import Grounding
from ...relational_algebra import NaturalJoin
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import (
    Symbol,
    Constant,
    ExpressionBlock,
    ExistentialPredicate,
)
from ...datalog.expressions import Implication, Conjunction
from ..expressions import (
    VectorisedTableDistribution,
    ProbabilisticPredicate,
    RandomVariableValuePointer,
    ConcatenateColumn,
    AddIndexColumn,
    SumColumns,
    MultiplyColumns,
    MultipleNaturalJoin,
)

P = Symbol("P")
Q = Symbol("Q")
T = Symbol("T")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
p = Symbol("p")
a = Constant[str]("a")
b = Constant[str]("b")
c = Constant[str]("c")
d = Constant[str]("d")


def test_extensional_grounding():
    grounding = Grounding(
        P(x, y),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=("x", "y"), iterable={(a, b), (c, d)}
            )
        ),
    )
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    translator.walk(grounding)
    assert not translator.edges
    assert translator.cpds == {
        P: VectorisedTableDistribution(
            Constant[Mapping](
                {
                    Constant[bool](False): Constant[float](0.0),
                    Constant[bool](True): Constant[float](1.0),
                }
            ),
            grounding,
        )
    }
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    gm = translator.walk(ExpressionBlock([grounding]))
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[bool](False): Constant[float](0.0),
                        Constant[bool](True): Constant[float](1.0),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_probabilistic_grounding():
    probfact = Implication(
        ProbabilisticPredicate(Constant[float](0.3), P(x)),
        Constant[bool](True),
    )
    grounding = Grounding(
        probfact,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(
        ExpressionBlock([grounding])
    )
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[bool](False): Constant[float](0.7),
                        Constant[bool](True): Constant[float](0.3),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_intensional_grounding():
    extensional_grounding = Grounding(
        T(x),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    probabilistic_grounding = Grounding(
        Implication(
            ProbabilisticPredicate(Constant[float](0.3), P(x)),
            Constant[bool](True),
        ),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    intensional_grounding = Grounding(
        Implication(Q(x), Conjunction([P(x), T(x)])),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    groundings = ExpressionBlock(
        [extensional_grounding, probabilistic_grounding, intensional_grounding]
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(groundings)
    assert gm.edges == Constant[Mapping]({Q: {P, T}})


def test_construct_abstract_set_from_pandas_dataframe():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    abstract_set = AlgebraSet(iterable=df, columns=df.columns)
    assert np.all(
        np.array(list(abstract_set.itervalues()))
        == np.array(
            [
                np.array([1, 5]),
                np.array([2, 6]),
                np.array([3, 7]),
                np.array([4, 8]),
            ]
        )
    )


def test_bernoulli_vect_table_distrib():
    grounding = Grounding(
        P(x),
        Constant[AbstractSet](AlgebraSet(iterable=[1, 2, 3], columns=["x"])),
    )
    distrib = bernoulli_vect_table_distrib(Constant[float](1.0), grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](1.0)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.0)
    distrib = bernoulli_vect_table_distrib(Constant[float](0.2), grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](0.2)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.8)

    with pytest.raises(NeuroLangException):
        bernoulli_vect_table_distrib(p, grounding)

    solver = ExtendedRelationalAlgebraSolver({})
    walked_distrib = solver.walk(
        bernoulli_vect_table_distrib(Constant[float](0.7), grounding)
    )
    assert isinstance(walked_distrib, Constant[AbstractSet])


def test_extensional_vect_table_distrib():
    grounding = Grounding(
        P(x),
        Constant[AbstractSet](AlgebraSet(iterable=[1, 2, 3], columns=["x"])),
    )
    distrib = extensional_vect_table_distrib(grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](1.0)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.0)


def test_rv_value_pointer():
    parent_values = {
        P: AlgebraSet(
            iterable=[("a", 1), ("b", 1), ("c", 0)],
            columns=["x", _make_numerical_col_symb().name],
        )
    }
    solver = ExtendedRelationalAlgebraSolver(parent_values)
    walked = solver.walk(RandomVariableValuePointer(P))
    assert isinstance(walked, Constant[AbstractSet])
    assert isinstance(walked.value, AlgebraSet)


def test_concatenate_column():
    solver = ExtendedRelationalAlgebraSolver({})
    relation = Constant[AbstractSet](
        AlgebraSet(iterable=range(100), columns=["x"])
    )
    column_name = y
    column_values = Constant[np.ndarray](np.arange(100) * 2)
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.hstack(
                [
                    relation.value.to_numpy(),
                    np.atleast_2d(column_values.value).T,
                ]
            ),
            columns=["x", "y"],
        )
    )
    result = solver.walk(
        ConcatenateColumn(relation, column_name, column_values)
    )
    assert _eq_tuple_sets(result, expected)


def test_add_index_column():
    solver = ExtendedRelationalAlgebraSolver({})
    relation = Constant[AbstractSet](
        AlgebraSet(iterable=np.arange(100) * 4, columns=["x"])
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.hstack(
                [np.atleast_2d(np.arange(100)).T, relation.value.to_numpy()]
            ),
            columns=["whatever", "x"],
        )
    )
    result = solver.walk(AddIndexColumn(relation, Constant[str]("whatever")))
    assert _eq_tuple_sets(result, expected)
    assert set(result.value.columns) == set(expected.value.columns)


def test_sum_columns():
    solver = ExtendedRelationalAlgebraSolver({})
    r1 = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 4,
            columns=[_make_numerical_col_symb().name],
        )
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 14,
            columns=[_make_numerical_col_symb().name],
        )
    )
    result = solver.walk(
        SumColumns(
            ConcatenateColumn(
                r1,
                _make_numerical_col_symb(),
                Constant[np.ndarray](np.arange(100) * 10),
            )
        )
    )
    assert _eq_tuple_sets(result, expected)


def test_args_to_column_names():
    assert _args_to_column_names(P(x, y, z)) == Constant[Tuple](
        tuple(["x", "y", "z"])
    )
    with pytest.raises(NeuroLangException):
        _args_to_column_names(P(x, a, y))


def _as_tuple_set(obj):
    if isinstance(obj, np.ndarray) and len(obj.shape) == 2:
        return set(tuple(obj[i]) for i in range(obj.shape[0]))
    elif isinstance(obj, AlgebraSet):
        return set(tuple(val) for val in obj.itervalues())
    elif isinstance(obj, Constant[AbstractSet]):
        return _as_tuple_set(obj.value)


def _eq_tuple_sets(first, second):
    return _as_tuple_set(first) == _as_tuple_set(second)
