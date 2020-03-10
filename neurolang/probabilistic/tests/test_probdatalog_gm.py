from typing import Mapping, AbstractSet

import pytest
import pandas as pd
import numpy as np

from ...exceptions import NeuroLangException
from ...utils.relational_algebra_set import RelationalAlgebraSet
from ..probdatalog_gm import (
    TranslateGroundedProbDatalogToGraphicalModel,
    AlgebraSet,
    bernoulli_vect_table_distrib,
    extensional_vect_table_distrib,
    ExtendedRelationalAlgebraSolver,
    make_numerical_col_symb,
    _split_numerical_cols,
    compute_marginal_probability,
    and_vect_table_distribution,
    SuccQuery,
    QueryGraphicalModelSolver,
    infer_pfact_params,
    succ_query,
)
from ..probdatalog import Grounding, ground_probdatalog_program
from ...relational_algebra import (
    NaturalJoin,
    RelationalAlgebraSolver,
    ColumnStr,
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import Symbol, Constant, ExpressionBlock
from ...datalog.expressions import Implication, Conjunction, Fact
from ..expressions import (
    VectorisedTableDistribution,
    ProbabilisticPredicate,
    RandomVariableValuePointer,
    ConcatenateColumn,
    AddIndexColumn,
    SumColumns,
    MultiplyColumns,
    AddRepeatedValueColumn,
    Aggregation,
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
T = Symbol("T")
H = Symbol("H")
Y = Symbol("Y")
w = Symbol("w")
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
        Implication(P(x, y), Constant[bool](True)),
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
        Implication(T(x), Constant[bool](True)),
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
        P: Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", 1), ("b", 1), ("c", 0)],
                columns=["x", make_numerical_col_symb().name],
            )
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
    _assert_relations_almost_equal(result, expected)


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
    _assert_relations_almost_equal(result, expected)
    assert set(result.value.columns) == set(expected.value.columns)


def test_sum_columns():
    solver = ExtendedRelationalAlgebraSolver({})
    r1 = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 4,
            columns=[make_numerical_col_symb().name],
        )
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 14,
            columns=[make_numerical_col_symb().name],
        )
    )
    result = solver.walk(
        SumColumns(
            ConcatenateColumn(
                r1,
                make_numerical_col_symb(),
                Constant[np.ndarray](np.arange(100) * 10),
            )
        )
    )
    _assert_relations_almost_equal(result, expected)


def test_multiply_columns():
    solver = ExtendedRelationalAlgebraSolver({})
    r1 = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 4,
            columns=[make_numerical_col_symb().name],
        )
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=np.arange(100) * 40 * np.arange(100),
            columns=[make_numerical_col_symb().name],
        )
    )
    result = solver.walk(
        MultiplyColumns(
            ConcatenateColumn(
                r1,
                make_numerical_col_symb(),
                Constant[np.ndarray](np.arange(100) * 10),
            )
        )
    )
    _assert_relations_almost_equal(result, expected)


def test_add_repeated_value_column():
    solver = ExtendedRelationalAlgebraSolver({})
    relation = Constant[AbstractSet](
        AlgebraSet(iterable=["a", "b", "c"], columns=["x"])
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=[("a", 0), ("b", 0), ("c", 0)],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    result = solver.walk(AddRepeatedValueColumn(relation, Constant[int](0)))
    _assert_relations_almost_equal(result, expected)


def test_compute_marginal_probability_single_parent():
    parent_symb = P
    parent_marginal_distrib = Constant[AbstractSet](
        AlgebraSet(
            iterable=[("a", 0.2), ("b", 0.8), ("c", 1.0)],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    grounding = Grounding(
        Implication(Q(x), P(x)),
        Constant[AbstractSet](AlgebraSet(iterable=["c", "a"], columns=["x"])),
    )
    cpd = and_vect_table_distribution(
        rule_grounding=grounding,
        parent_groundings={
            parent_symb: Grounding(
                P(y),
                Constant[AbstractSet](
                    AlgebraSet(iterable=["a", "b", "c"], columns=["y"])
                ),
            )
        },
    )
    marginal = compute_marginal_probability(
        cpd, {parent_symb: parent_marginal_distrib}, {parent_symb: grounding}
    )
    _assert_relations_almost_equal(
        marginal,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("c", 1.0), ("a", 0.2)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_compute_marginal_probability_two_parents():
    parent_marginal_distribs = {
        P: Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", 0.2), ("b", 0.8), ("c", 1.0)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
        Q: Constant[AbstractSet](
            AlgebraSet(
                iterable=[("b", 0.2), ("c", 0.0), ("d", 0.99)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    }
    parent_groundings = {
        P: Grounding(
            P(y),
            Constant[AbstractSet](
                AlgebraSet(iterable=["a", "b", "c"], columns=["y"])
            ),
        ),
        Q: Grounding(
            Q(z),
            Constant[AbstractSet](
                AlgebraSet(iterable=["b", "c", "d"], columns=["z"])
            ),
        ),
    }
    grounding = Grounding(
        Implication(Z(x), Conjunction([P(x), Q(x)])),
        Constant[AbstractSet](AlgebraSet(iterable=["c", "b"], columns=["x"])),
    )
    cpd = and_vect_table_distribution(
        rule_grounding=grounding, parent_groundings=parent_groundings
    )
    marginal = compute_marginal_probability(
        cpd, parent_marginal_distribs, parent_groundings
    )
    _assert_relations_almost_equal(
        marginal,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("c", 0.0), ("b", 0.16)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_simple():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), P(x)),
            # Implication(Q(x), Conjunction([P(x), ])),
        ]
    )
    result = succ_query(code, Q(x))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", 0.3), ("b", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_simple_const_in_antecedent():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(a)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), T(x), R(a)])),
        ]
    )
    result = succ_query(code, Q(x))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", 0.3), ("b", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_with_constant():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), T(x)])),
        ]
    )
    grounded = ground_probdatalog_program(code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    query = SuccQuery(Q(a))
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(query)
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_multiple_parents():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(b)),
            Fact(R(c)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(c)),
                Constant[bool](True),
            ),
            Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),
        ]
    )
    grounded = ground_probdatalog_program(code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(Q(x, y)))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[
                    ("a", "b", 0.1),
                    ("b", "b", 0.1),
                    ("a", "c", 0.1),
                    ("b", "c", 0.1),
                ],
                columns=["x", "y", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_multi_level():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(b)),
            Fact(R(c)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(c)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.7), Y(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.7), Y(b)),
                Constant[bool](True),
            ),
            Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),
            Implication(H(x, y), Conjunction([Q(x, y), T(y), Y(y)])),
        ]
    )
    grounded = ground_probdatalog_program(code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(H(x, y)))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            AlgebraSet(
                iterable=[("a", "b", 0.07), ("b", "b", 0.07)],
                columns=["x", "y", make_numerical_col_symb().name],
            )
        ),
    )


@pytest.mark.slow
def test_succ_query_hundreds_of_facts():
    facts_t = [Fact(T(Constant[int](i))) for i in range(1000)]
    facts_r = [Fact(R(Constant[int](i))) for i in range(300)]
    code = ExpressionBlock(
        facts_t
        + facts_r
        + [
            Implication(
                ProbabilisticPredicate(
                    Constant[float](0.5), P(Constant[int](i))
                ),
                Constant[bool](True),
            )
            for i in range(1000)
        ]
        + [
            Implication(
                ProbabilisticPredicate(
                    Constant[float](0.2), Z(Constant[int](i))
                ),
                Constant[bool](True),
            )
            for i in range(300)
        ]
        + [Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),]
    )
    grounded = ground_probdatalog_program(code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(Q(x)))


def test_succ_query_hundreds_of_facts_fast():
    extensional_sets = {
        T: [(i,) for i in range(1000)],
        R: [(i,) for i in range(300)],
    }
    probabilistic_sets = {
        P: [(0.2, i,) for i in range(1000)],
        Z: [(0.5, i,) for i in range(300)],
    }
    code = ExpressionBlock(
        [Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),]
    )
    grounded = ground_probdatalog_program(
        code,
        probabilistic_sets=probabilistic_sets,
        extensional_sets=extensional_sets,
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(Q(x)))


def test_sum_aggregate():
    relation = Constant[AbstractSet](
        AlgebraSet(
            iterable=[
                ("a", "b", 2),
                ("b", "a", 3),
                ("c", "a", 1),
                ("c", "a", 2),
                ("b", "a", 2),
            ],
            columns=["x", "y", "z"],
        )
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=[("a", "b", 2), ("b", "a", 5), ("c", "a", 3)],
            columns=["x", "y", "w"],
        )
    )
    sum_agg_op = Aggregation(
        Constant[str]("sum"),
        relation,
        [Constant(ColumnStr("x")), Constant(ColumnStr("y"))],
        Constant(ColumnStr("z")),
        Constant(ColumnStr("w")),
    )
    solver = ExtendedRelationalAlgebraSolver({})
    result = solver.walk(sum_agg_op)
    _assert_relations_almost_equal(result, expected)


def test_count_aggregate():
    relation = Constant[AbstractSet](
        AlgebraSet(
            iterable=[
                ("a", "b", 2),
                ("b", "a", 3),
                ("c", "a", 1),
                ("c", "a", 2),
                ("b", "a", 2),
            ],
            columns=["x", "y", "z"],
        )
    )
    expected = Constant[AbstractSet](
        AlgebraSet(
            iterable=[("a", "b", 1), ("b", "a", 2), ("c", "a", 2)],
            columns=["x", "y", "w"],
        )
    )
    count_agg_op = Aggregation(
        Constant[str]("count"),
        relation,
        [Constant(ColumnStr("x")), Constant(ColumnStr("y"))],
        Constant(ColumnStr("z")),
        Constant(ColumnStr("w")),
    )
    solver = ExtendedRelationalAlgebraSolver({})
    result = solver.walk(count_agg_op)
    _assert_relations_almost_equal(result, expected)


def test_exact_inference_pfact_params():
    param_symb = Symbol.fresh()
    pfact = Implication(
        ProbabilisticPredicate(param_symb, P(x, y)), Constant[bool](True)
    )
    relation = Constant[AbstractSet](
        AlgebraSet(
            iterable=[
                ("a", "b", "p1"),
                ("a", "c", "p2"),
                ("b", "b", "p1"),
                ("b", "c", "p2"),
            ],
            columns=("x", "y", param_symb.name),
        )
    )
    pfact_grounding = Grounding(pfact, relation)
    interpretations = {
        P: AlgebraSet(
            iterable=[
                ("a", "b", 1),
                ("a", "c", 1),
                ("a", "c", 2),
                ("b", "b", 2),
                ("b", "c", 2),
                ("a", "b", 3),
                ("a", "b", 3),
                ("b", "b", 3),
            ],
            columns=("x", "y", "__interpretation_id__"),
        )
    }
    estimations = infer_pfact_params(pfact_grounding, interpretations, 3)


def _assert_relations_almost_equal(r1, r2):
    assert len(r1.value) == len(r2.value)
    if r1.value.arity == 1 and r2.value.arity == 1:
        np.testing.assert_array_almost_equal(
            r1.value._container[r1.value.columns[0]].values,
            r2.value._container[r2.value.columns[0]].values,
        )
    else:
        joined = RelationalAlgebraSolver().walk(NaturalJoin(r1, r2))
        _, num_cols = _split_numerical_cols(joined)
        if len(num_cols) == 2:
            arr1 = joined.value._container[num_cols[0]].values
            arr2 = joined.value._container[num_cols[1]].values
            np.testing.assert_array_almost_equal(arr1, arr2)
        elif len(num_cols) == 0:
            assert len(joined.value) == len(r1.value)
