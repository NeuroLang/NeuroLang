from operator import eq
from typing import AbstractSet

import pytest

from ...datalog.translate_to_named_ra import TranslateToNamedRA
from nl_expr.exceptions import NonLiftableException
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation,
    Union,
)
from ...relational_algebra import (
    ColumnInt,
    ColumnStr,
    Difference,
    Materialise,
    NameColumns,
    Projection,
    str2columnstr_constant
)
from ...relational_algebra_provenance import (
    DisjointProjection,
    IndependentProjection
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .. import dalvi_suciu_lift, transforms
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet
)
from ..transforms import convert_rule_to_ucq

TNRA = TranslateToNamedRA()
EQ = Constant(eq)


def test_has_separator_variable_existential():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')
    y = Symbol('y')

    expression = ExistentialPredicate(
        y,
        Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))
    )

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression, {})
    assert sv == {y}

    expression = ExistentialPredicate(
        y,
        Disjunction((B(x, y), Conjunction((A(y), B(y, x))), C(y)))
    )

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression, {})
    assert sv == set()

    expression = Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression, {})
    assert sv == set()

    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')

    expression = ExistentialPredicate(x1, ExistentialPredicate(
        x2,
        ExistentialPredicate(y1, ExistentialPredicate(
            y2,
            Disjunction((
                Conjunction((R(x1), S(x1, y1))),
                Conjunction((T(x2), S(x2, y2)))
            ))
        ))
    ))

    assert dalvi_suciu_lift.has_separator_variables(expression, {})
    assert (
        dalvi_suciu_lift.find_separator_variables(expression, {})[0] &
        {x1, x2}
    )


def test_lifted_bcq_fig_4_4():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    )))
    cq = convert_rule_to_ucq(cq)
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_lifted_separator_variable():
    R = Symbol('R')
    x1 = Symbol('x1')
    z = Symbol('z')

    cq = ExistentialPredicate(x1, R(x1, z))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_lifted_join():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((R(x1, z), S(x1, x2), T(x1, z))))
    ucq = convert_rule_to_ucq(cq)
    plan = dalvi_suciu_lift.dalvi_suciu_lift(ucq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_another_liftable_join():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((R(x1, z), S(x1, x2), S(x1, z))))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(convert_rule_to_ucq(cq), {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_non_liftable_join():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((R(x1, z), S(x2, x1), S(x1, z))))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(convert_rule_to_ucq(cq), {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert not res


def test_lifted_cq_fig_4_5():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    y3 = Symbol('y3')
    z = Symbol('z')

    cq = Implication(Q(z), Disjunction((
        Conjunction((R(z, x1), S(x1, y1))),
        Conjunction((S(x2, y2), T(z, y2))),
        Conjunction((R(z, x3), T(z, y3))),
    )))

    plan = dalvi_suciu_lift.dalvi_suciu_lift(convert_rule_to_ucq(cq), {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_minimize_component_conjunction():
    Q = Symbol('Q')
    R = Symbol('R')
    T = Symbol('T')
    x1 = Symbol('x1')
    y1 = Symbol('y1')
    y2 = Symbol('y2')

    expr = Conjunction((Q(x1),))
    res = transforms.minimize_component_conjunction(expr)
    assert res == expr

    expr = Conjunction((Q(x1, y1), R(x1, y2)))
    res = transforms.minimize_component_conjunction(expr)
    assert res == expr

    expr = Conjunction((
        ExistentialPredicate(y1, Q(x1, y1)),
        ExistentialPredicate(y2, Q(x1, y2))
    ))
    res = transforms.minimize_component_conjunction(expr)
    assert res in (
        Conjunction((q,)) for q in
        (
            ExistentialPredicate(y1, Q(x1, y1)),
            ExistentialPredicate(y2, Q(x1, y2))
        )
    )

    expr = Conjunction((
        ExistentialPredicate(y2, Q(x1, y2)),
        Disjunction((
            ExistentialPredicate(y1, Q(x1, y1)),
            T(x1)
        ))
    ))
    res = transforms.minimize_component_conjunction(expr)
    assert res == Conjunction((ExistentialPredicate(y2, Q(x1, y2)),))

    expr = Conjunction((
        Disjunction((
            ExistentialPredicate(y1, Q(x1, y1)),
            T(x1)
        )),
        ExistentialPredicate(y2, Q(x1, y2)),
    ))
    res = transforms.minimize_component_conjunction(expr)
    assert res == Conjunction((ExistentialPredicate(y2, Q(x1, y2)),))


def test_minimize_component_disjunction():
    Q = Symbol('Q')
    R = Symbol('R')
    T = Symbol('T')
    x1 = Symbol('x1')
    y1 = Symbol('y1')
    y2 = Symbol('y2')

    expr = Disjunction((Q(x1),))
    res = transforms.minimize_component_disjunction(expr)
    assert res == expr

    expr = Disjunction((Q(x1, y1), R(x1, y2)))
    res = transforms.minimize_component_disjunction(expr)
    assert res == expr

    expr = Disjunction((
        ExistentialPredicate(y1, Q(x1, y1)),
        ExistentialPredicate(y2, Q(x1, y2))
    ))
    res = transforms.minimize_component_disjunction(expr)
    assert res in (
        Disjunction((q,)) for q in
        (
            ExistentialPredicate(y1, Q(x1, y1)),
            ExistentialPredicate(y2, Q(x1, y2))
        )
    )

    expr = Disjunction((
        ExistentialPredicate(y2, Q(x1, y2)),
        Disjunction((
            ExistentialPredicate(y1, Q(x1, y1)),
            T(x1)
        ))
    ))
    res = transforms.minimize_component_disjunction(expr)
    assert res == Disjunction((
        ExistentialPredicate(y1, Q(x1, y1)),
        T(x1)
        ))

    expr = Disjunction((
        Disjunction((
            ExistentialPredicate(y1, Q(x1, y1)),
            T(x1)
        )),
        ExistentialPredicate(y2, Q(x1, y2)),
    ))
    res = transforms.minimize_component_disjunction(expr)
    assert res == Disjunction((
        ExistentialPredicate(y1, Q(x1, y1)),
        T(x1)
    ))


def test_intractable_queries():
    """
    Queries that are "intractable", in that they are #P-hard, as defined in
    section 3.2 of [1]_.

    The lifted query processing algorithm should return a NonLiftable object
    containing the unprocessed query.

    [1] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    """
    R = Symbol('R')
    S = Symbol('S')
    S1 = Symbol('S1')
    S2 = Symbol('S2')
    S3 = Symbol('S3')
    T = Symbol('T')
    x = Symbol("x")
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y = Symbol("y")
    y0 = Symbol("y0")
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    y3 = Symbol('y3')
    h0 = ExistentialPredicate(
        x,
        Conjunction(
            (
                R(x),
                ExistentialPredicate(y, Conjunction((S(x, y), T(y)))),
            )
        ),
    )
    h1 = Disjunction(
        (
            ExistentialPredicate(
                x0, Conjunction((R(x0), ExistentialPredicate(y0, S(x0, y0))))
            ),
            ExistentialPredicate(
                y1, Conjunction((ExistentialPredicate(x1, S(x1, y1)), T(y1)))
            ),
        )
    )
    h2 = Disjunction(
        (
            ExistentialPredicate(
                x0,
                Conjunction((R(x0), S1(x0, y0))),
            ),
            ExistentialPredicate(
                x1,
                ExistentialPredicate(
                    y1,
                    Conjunction((S1(x1, y1), S2(x1, y1))),
                )
            ),
            ExistentialPredicate(
                y2,
                Conjunction((ExistentialPredicate(x2, S2(x2, y2)), T(y2))),
            ),
        )
    )
    h3 = Disjunction(
        (
            ExistentialPredicate(
                x0,
                Conjunction((R(x0), S1(x0, y0))),
            ),
            ExistentialPredicate(
                x1,
                ExistentialPredicate(
                    y1,
                    Conjunction((S1(x1, y1), S2(x1, y1))),
                )
            ),
            ExistentialPredicate(
                x2,
                ExistentialPredicate(
                    y2,
                    Conjunction((S2(x2, y2), S3(x2, y2))),
                )
            ),
            ExistentialPredicate(
                y3,
                Conjunction((ExistentialPredicate(x3, S3(x3, y3)), T(y3))),
            ),
        )
    )
    for h in (h0, h1, h2, h3):
        plan = dalvi_suciu_lift.dalvi_suciu_lift(h, {})
        assert not dalvi_suciu_lift.is_pure_lifted_plan(plan)


def test_example_4_6_a_really_simple_query():
    R = Symbol("R")
    S = Symbol("S")
    x = Symbol("x")
    y = Symbol("y")
    query = ExistentialPredicate(
        x, Conjunction((R(x), ExistentialPredicate(y, S(x, y))))
    )
    resulting_plan = dalvi_suciu_lift.dalvi_suciu_lift(query, {})
    assert dalvi_suciu_lift.is_pure_lifted_plan(resulting_plan)


def test_example_4_7_a_query_with_self_joins():
    R = Symbol("R")
    S = Symbol("S")
    T = Symbol("T")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    y1 = Symbol("y1")
    y2 = Symbol("y2")
    Q1 = ExistentialPredicate(
        x1,
        ExistentialPredicate(
            y1,
            Conjunction((R(x1), S(x1, y1))),
        ),
    )
    Q2 = ExistentialPredicate(
        x2,
        ExistentialPredicate(
            y2,
            Conjunction((T(x2), S(x2, y2))),
        ),
    )
    query = Conjunction((Q1, Q2))
    resulting_plan = dalvi_suciu_lift.dalvi_suciu_lift(query, {})
    assert dalvi_suciu_lift.is_pure_lifted_plan(resulting_plan)


@pytest.mark.skip
def test_example_4_8_tractable_query_intractable_subquery():
    """
    We test the query

        R(x1), S(x1, y1) ∨ S(x2, y2), T(y2) ∨ R(x3), T(y3)

    whose first two disjuncts correspond to the hard H1 query, but the third
    disjunct makes the query tractable by using distributivity and logical
    equivalence.

    """
    R = Symbol("R")
    S = Symbol("S")
    T = Symbol("T")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    x3 = Symbol("x3")
    y1 = Symbol("y1")
    y2 = Symbol("y2")
    y3 = Symbol("y3")
    query = Disjunction(
        (
            ExistentialPredicate(
                x1,
                Conjunction(
                    (
                        R(x1),
                        ExistentialPredicate(y1, S(x1, y1)),
                    )
                ),
            ),
            ExistentialPredicate(
                y2,
                Conjunction(
                    (
                        ExistentialPredicate(x2, S(x2, y2)),
                        T(y2),
                    )
                ),
            ),
            Conjunction(
                (
                    ExistentialPredicate(x3, R(x3)),
                    ExistentialPredicate(y3, T(y3)),
                )
            ),
        )
    )
    resulting_plan = dalvi_suciu_lift.dalvi_suciu_lift(query, {})
    assert dalvi_suciu_lift.is_pure_lifted_plan(resulting_plan)


def test_single_disjoint_project_one_variable():
    P = Symbol("P")
    x = Symbol("x")
    pchoice_relation = NamedRelationalAlgebraFrozenSet(
        iterable=[
            (0.2, "a"),
            (0.7, "b"),
            (0.1, "c"),
        ],
        columns=("_p_", "x"),
    )
    symbol_table = {
        P: ProbabilisticChoiceSet(
            Constant[AbstractSet](pchoice_relation),
            str2columnstr_constant("_p_"),
        ),
    }
    query = ExistentialPredicate(x, P(x))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert isinstance(plan, DisjointProjection)


def test_single_disjoint_project_two_variables():
    P = Symbol("P")
    x = Symbol("x")
    y = Symbol("y")
    pchoice_relation = NamedRelationalAlgebraFrozenSet(
        iterable=[
            (0.2, "a", "b"),
            (0.7, "b", "b"),
            (0.1, "b", "c"),
        ],
        columns=("_p_", "x", "y"),
    )
    symbol_table = {
        P: ProbabilisticChoiceSet(
            Constant[AbstractSet](pchoice_relation),
            str2columnstr_constant("_p_"),
        ),
    }
    query = ExistentialPredicate(y, P(x, y))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert isinstance(plan, DisjointProjection)


def test_simple_existential_query_plan():
    R = Symbol("R")
    x = Symbol("x")
    y = Symbol("y")
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b"),
                ("b", "a"),
                ("c", "a"),
            ],
            columns=("x", "y"),
        )
    )
    symbol_table = {R: DeterministicFactSet(relation)}
    query = ExistentialPredicate(x, R(x, y))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)


def test_extract_probabilistic_root_variables_no_probabilistic_atom():
    res = dalvi_suciu_lift.extract_probabilistic_root_variables(set(), dict())
    assert len(res) == 0

    P = Symbol("P")
    x = Symbol("x")
    y = Symbol("y")
    formulas = [
        P(x, y),
    ]
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b"),
                ("b", "a"),
                ("c", "a"),
            ],
            columns=("x", "y"),
        )
    )
    symbol_table = {
        P: DeterministicFactSet(relation)
    }
    res = dalvi_suciu_lift.extract_probabilistic_root_variables(
        formulas, symbol_table
    )
    assert len(res) == 0


def test_lifted_negation():
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    query = ExistentialPredicate(
        y,
        Conjunction((
            R(x, y),
            Negation(S(y))
        ))
    )

    symbol_table = {
        symbol: ProbabilisticFactSet(Symbol.fresh(), 'p')
        for symbol in (R, S)
    }
    res = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    expected = IndependentProjection(
        Difference(
            NameColumns(
                Projection(
                    R,
                    (Constant(ColumnInt(0)), Constant(ColumnInt(1))),
                ),
                (Constant(ColumnStr('x')), Constant(ColumnStr('y'))),
            ),
            NameColumns(
                Projection(
                    S,
                    (Constant(ColumnInt(0)),)
                ),
                (Constant(ColumnStr('y')),)
            ),
        ),
        (Constant(ColumnStr('x')),)
    )

    assert res == expected


def test_lifted_disjunction_with_negation():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    query = Disjunction((
        ExistentialPredicate(
            y,
            Conjunction((
                R(x, y),
                Negation(S(y))
            ))
        ),
        ExistentialPredicate(
            y,
            T(x, y)
        )
    ))

    symbol_table = {
        symbol: ProbabilisticFactSet(Symbol.fresh(), 'p')
        for symbol in (R, S, T)
    }
    res = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)

    assert dalvi_suciu_lift.is_pure_lifted_plan(res)


def test_lifted_disjunction_cross_product_with_negation():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    V = Symbol('V')
    x = Symbol('x')
    y = Symbol('y')

    query = Disjunction((
        ExistentialPredicate(
            y,
            Conjunction((
                R(x),
                Negation(S(y)),
                V(y)
            ))
        ),
        ExistentialPredicate(
            y,
            T(x, y)
        )
    ))

    symbol_table = {
        symbol: ProbabilisticFactSet(Symbol.fresh(), 'p')
        for symbol in (R, S, T, V)
    }
    res = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)

    assert dalvi_suciu_lift.is_pure_lifted_plan(res)


def test_lifted_conjunction_existential_negation():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    query = Conjunction((
        Negation(
            ExistentialPredicate(
                x,
                Conjunction((
                    R(x),
                    S(x, y),
                ))
            )
        ),
        T(y)
    ))

    symbol_table = {
        symbol: ProbabilisticFactSet(Symbol.fresh(), 'p')
        for symbol in (R, S, T)
    }
    res = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)

    assert dalvi_suciu_lift.is_pure_lifted_plan(res)


def test_lifted_conjunction_existential_negation_constant():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    Q = Symbol('Q')
    x = Symbol('x')
    y = Symbol('y')

    query = Conjunction((
        Negation(
            ExistentialPredicate(
                x,
                Conjunction((
                    R(x),
                    S(x, y, Constant('2'))
                ))
            )
        ),
        Q(x, y),
        T(y)
    ))

    symbol_table = {
        symbol: ProbabilisticFactSet(Symbol.fresh(), 'p')
        for symbol in (R, S, T)
    }
    symbol_table[Q] = DeterministicFactSet(Symbol.fresh())

    res = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)

    assert dalvi_suciu_lift.is_pure_lifted_plan(res)


def test_mixed_probabilistic_deterministic_safe():
    R = Symbol("R")
    S = Symbol("S")
    T = Symbol("T")
    x = Symbol("x")
    y = Symbol("y")
    R_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("x",))
    )
    S_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("_p_", "x", "y"))
    )
    T_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("_p_", "x",))
    )
    symbol_table = {
        R: DeterministicFactSet(R_relation),
        S: ProbabilisticFactSet(S_relation, Constant(ColumnStr("_p_"))),
        T: ProbabilisticFactSet(T_relation, Constant(ColumnStr("_p_"))),
    }
    query = ExistentialPredicate(
        x,
        ExistentialPredicate(
            y,
            Conjunction((R(x), S(x, y), T(y))),
        )
    )
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)


def test_mixed_probabilistic_deterministic_unsafe():
    """S is a function neither of x nor of y."""
    R = Symbol("R")
    S = Symbol("S")
    T = Symbol("T")
    x = Symbol("x")
    y = Symbol("y")
    R_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("x",))
    )
    S_relation = Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
        iterable=[(1, 1), (1, 2), (2, 1)], columns=("x", "y")
    ))
    T_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("_p_", "x",))
    )
    symbol_table = {
        R: ProbabilisticFactSet(R_relation, Constant(ColumnStr("_p_"))),
        S: DeterministicFactSet(S_relation),
        T: ProbabilisticFactSet(T_relation, Constant(ColumnStr("_p_"))),
    }
    query = ExistentialPredicate(
        x,
        ExistentialPredicate(
            y,
            Conjunction((R(x), S(x, y), T(y))),
        )
    )
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert not dalvi_suciu_lift.is_pure_lifted_plan(plan)


def test_interaction_deterministic_probfact_probchoice_safe():
    Det = Symbol("Det")
    Pchoice = Symbol("Pchoice")
    Pfact = Symbol("Pfact")
    x = Symbol("x")
    y = Symbol("y")
    Det_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("x",))
    )
    Pchoice_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("_p_", "y"))
    )
    Pfact_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[], columns=("_p_", "x",))
    )
    symbol_table = {
        Det: ProbabilisticFactSet(Det_relation, Constant(ColumnStr("_p_"))),
        Pchoice: DeterministicFactSet(Pchoice_relation),
        Pfact: ProbabilisticFactSet(
            Pfact_relation, Constant(ColumnStr("_p_"))
        ),
    }
    query = ExistentialPredicate(
        x,
        ExistentialPredicate(
            y,
            Conjunction((Det(x, y), Pchoice(y), Pfact(x))),
        )
    )
    plan = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)


@pytest.mark.parametrize("rules", (
    "S(x), S(y)",
    "S(x), S(y) | R(y, x), S(w)",
    "S(w), R(x, x) | S(y), S(x), S(w)",
), ids=(
    "self-join", "union implying the self-join",
    "three equivalent conjuncts",
))
def test_inclusion_exclusion_on_equivalent_formulas(rules):
    """∃x S(x) ∧ ∃y S(y) is ∃x S(x): its two conjuncts and their
    disjunction are equivalent, and the Möbius weights of their lattice
    must count them once. The union adds R(y, x) ∧ S(w), which implies
    ∃x S(x). The CNF of the last one has three conjuncts equivalent to
    ∃x S(x), of which minimisation must keep one."""
    R, S, Z, ans = (Symbol(n) for n in ("R", "S", "Z", "ans"))
    w, x, y = (Symbol(n) for n in ("w", "x", "y"))
    bodies = {
        "S(x), S(y)": (Conjunction((S(x), S(y))),),
        "S(x), S(y) | R(y, x), S(w)": (
            Conjunction((S(x), S(y))), Conjunction((R(y, x), S(w)))
        ),
        "S(w), R(x, x) | S(y), S(x), S(w)": (
            Conjunction((S(w), R(x, x))), Conjunction((S(y), S(x), S(w)))
        ),
    }[rules]
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        R, [(0.5, "a", "b"), (0.4, "b", "a"), (0.2, "a", "a")]
    )
    cpl_program.add_probabilistic_facts_from_tuples(
        S, [(0.3, "a"), (0.6, "b")]
    )
    cpl_program.walk(Union(tuple(Implication(Z(), b) for b in bodies)))
    result = dalvi_suciu_lift.solve_succ_query(
        Implication(ans(), Z()), cpl_program
    )
    expected = 1 - (1 - 0.3) * (1 - 0.6)
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


def test_symbols_of_relations_to_materialise_are_kept_in_plans():
    """Like constant sets, relations marked to be materialised are left
    behind their symbols, rather than inlined in the plan.
    """
    relation = Materialise(Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x",), [(1,)])
    ))
    symbol = Symbol.fresh()
    replace = dalvi_suciu_lift.ReplaceAllSymbolsButConstantSets(
        {symbol: relation}
    )

    assert replace.walk(symbol) == symbol


@pytest.mark.parametrize("rules", (
    "S(x), S(y)",
    "S(x), S(y) | R(y, x), S(w)",
    "S(w), R(x, x) | S(y), S(x), S(w)",
), ids=(
    "self-join", "union implying the self-join",
    "three equivalent conjuncts",
))
def test_inclusion_exclusion_on_equivalent_formulas(rules):
    """∃x S(x) ∧ ∃y S(y) is ∃x S(x): its two conjuncts and their
    disjunction are equivalent, and the Möbius weights of their lattice
    must count them once. The union adds R(y, x) ∧ S(w), which implies
    ∃x S(x). The CNF of the last one has three conjuncts equivalent to
    ∃x S(x), of which minimisation must keep one."""
    R, S, Z, ans = (Symbol(n) for n in ("R", "S", "Z", "ans"))
    w, x, y = (Symbol(n) for n in ("w", "x", "y"))
    bodies = {
        "S(x), S(y)": (Conjunction((S(x), S(y))),),
        "S(x), S(y) | R(y, x), S(w)": (
            Conjunction((S(x), S(y))), Conjunction((R(y, x), S(w)))
        ),
        "S(w), R(x, x) | S(y), S(x), S(w)": (
            Conjunction((S(w), R(x, x))), Conjunction((S(y), S(x), S(w)))
        ),
    }[rules]
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        R, [(0.5, "a", "b"), (0.4, "b", "a"), (0.2, "a", "a")]
    )
    cpl_program.add_probabilistic_facts_from_tuples(
        S, [(0.3, "a"), (0.6, "b")]
    )
    cpl_program.walk(Union(tuple(Implication(Z(), b) for b in bodies)))
    result = dalvi_suciu_lift.solve_succ_query(
        Implication(ans(), Z()), cpl_program
    )
    expected = 1 - (1 - 0.3) * (1 - 0.6)
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )

@pytest.mark.parametrize("answer", (False, True))
def test_choice_with_a_repeated_variable(answer):
    """C(x, x) on a probabilistic choice C is ranked into a selection of C,
    which is still a choice: its tuples are mutually exclusive."""
    C, Z, ans, x = (Symbol(n) for n in ("C", "Z", "ans", "x"))
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(
        C, [(0.5, "a", "b"), (0.3, "b", "b"), (0.2, "a", "a")]
    )
    if answer:
        cpl_program.walk(Union((Implication(Z(x), C(x, x)),)))
        query = Implication(ans(x), Z(x))
        expected = testing.make_prov_set(
            [(0.3, "b"), (0.2, "a")], ("_p_", "x")
        )
    else:
        cpl_program.walk(Union((Implication(Z(), C(x, x)),)))
        query = Implication(ans(), Z())
        expected = testing.make_prov_set([(0.5,)], ("_p_",))
    result = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(result, expected)


def _choice_table():
    s = Symbol("s")
    C, M = Symbol("C"), Symbol("M")
    return s, C, M, {
        C: ProbabilisticChoiceSet(
            Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
                columns=("p", "v"), iterable=[(0.5, 1), (0.5, 2)]
            )),
            Constant(ColumnStr("p")),
        ),
        M: Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("v", "w"), iterable=[(1, "a"), (2, "b")]
        )),
    }


def test_disjoint_separator_aligns_the_choice_atoms():
    s, C, M, symbol_table = _choice_table()
    f = Symbol("f")
    union = Disjunction((
        ExistentialPredicate(s, Conjunction((C(s), M(s, Constant("a"))))),
        ExistentialPredicate(f, Conjunction((C(f), M(f, Constant("b"))))),
    ))
    applies, plan = dalvi_suciu_lift.disjoint_separator(union, symbol_table)
    assert applies and isinstance(plan, DisjointProjection)


def test_disjoint_separator_declines():
    """No separator: a disjunct without the choice, a choice of two
    columns aligned on one only, or a disjunct without a free variable of
    the query."""
    s, C, M, symbol_table = _choice_table()
    f, u, t = Symbol("f"), Symbol("u"), Symbol("t")
    C2 = Symbol("C2")
    symbol_table[C2] = ProbabilisticChoiceSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("p", "v", "w"), iterable=[(0.5, 1, "a"), (0.5, 2, "b")]
        )),
        Constant(ColumnStr("p")),
    )
    queries = [
        Disjunction((
            ExistentialPredicate(s, Conjunction((C(s), M(s, t)))),
            ExistentialPredicate(f, M(f, t)),
        )),
        Disjunction((
            ExistentialPredicate(s, Conjunction((C2(s, t), M(s, t)))),
            ExistentialPredicate(f, ExistentialPredicate(u, Conjunction((
                C2(f, u), M(f, t)
            )))),
        )),
        Disjunction((
            ExistentialPredicate(s, Conjunction((C(s), M(s, t)))),
            ExistentialPredicate(f, Conjunction((C(f), M(f, Constant("a"))))),
        )),
    ]
    for query in queries:
        assert dalvi_suciu_lift.disjoint_separator(
            query, symbol_table
        ) == (False, None), query


def test_disjoint_separator_lifts_queries_with_a_negation():
    """
    The sum over the choice is sound whatever the sign of the other
    literals, and the plan of the deterministic rest has the negation as a
    difference of sets: the choice atoms in it are replaced by their
    support, so that every label is 0 or 1.
    """
    s, C, M, symbol_table = _choice_table()
    f, t = Symbol("f"), Symbol("t")
    for query in [
        ExistentialPredicate(s, Conjunction((
            C(s), Negation(M(s, Constant("a")))
        ))),
        Disjunction((
            ExistentialPredicate(s, Conjunction((
                C(s), M(s, t), Negation(M(s, Constant("a")))
            ))),
            ExistentialPredicate(f, Conjunction((C(f), M(f, t)))),
        )),
    ]:
        applies, plan = dalvi_suciu_lift.disjoint_separator(
            query, symbol_table
        )
        assert applies, query
        assert isinstance(plan, DisjointProjection)


def test_clauses_are_range_restricted():
    """
    A clause that is a disjunction with a negated atom whose variable no
    positive atom of the disjunct binds is not safe range; a negation
    alone, a clause of a conjunction, is not a disjunction.
    """
    f, t = Symbol("f"), Symbol("t")
    A, B = Symbol("A"), Symbol("B")
    unsafe = Conjunction((
        ExistentialPredicate(f, A(f, t)),
        Disjunction((ExistentialPredicate(f, A(f, t)), Negation(B(t)))),
    ))
    safe = Conjunction((
        ExistentialPredicate(f, A(f, t)),
        Disjunction((
            ExistentialPredicate(f, A(f, t)),
            Conjunction((B(t), A(f, t))),
        )),
    ))
    negation_alone = Conjunction((
        ExistentialPredicate(f, A(f, t)), Negation(B(t))
    ))
    assert not dalvi_suciu_lift._clauses_are_range_restricted(unsafe)
    assert dalvi_suciu_lift._clauses_are_range_restricted(safe)
    assert dalvi_suciu_lift._clauses_are_range_restricted(negation_alone)


def test_choice_terms_as_equalities():
    """C(s) ≡ ∃f' (C(f') ∧ f' = s) at a position where another disjunct
    has a quantified variable, and C(f) ∧ C(f') ≡ C(f) ∧ f = f', whose
    fresh f' is then substituted."""
    C, M = Symbol("C"), Symbol("M")
    s, f = Symbol("s"), Symbol("f")
    rewriter = dalvi_suciu_lift.ChoiceTermsAsEqualities({C}, {(C, 0)}, {f})
    rewritten = rewriter.walk(Conjunction((C(s), M(s))))
    fresh, = rewriter.fresh
    assert set(rewritten.formulas) == {C(fresh), M(s), EQ(fresh, s)}
    rewriter = dalvi_suciu_lift.ChoiceTermsAsEqualities({C}, {(C, 0)}, {f})
    rewritten = rewriter.walk(Conjunction((C(f), M(f), C(s))))
    assert rewriter.fresh == set()
    assert set(rewritten.formulas) == {C(f), M(f), EQ(f, s)}


@pytest.mark.parametrize("function", (True, False))
def test_separator_determined_by_a_functional_dependency(function):
    """∃x∃y R(x) ∧ F(x, y) ∧ T(y), with F deterministic: y separates the
    query when F is a function of x, each tuple of R then being in the
    subquery of one y only; the query is #P-hard otherwise."""
    R, F, T, Z, ans = (Symbol(n) for n in ("R", "F", "T", "Z", "ans"))
    x, y = Symbol("x"), Symbol("y")
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        R, [(0.5, 1), (0.4, 2), (0.3, 3)]
    )
    cpl_program.add_probabilistic_facts_from_tuples(
        T, [(0.6, "a"), (0.7, "b")]
    )
    cpl_program.add_extensional_predicate_from_tuples(
        F, [(1, "a"), (2, "a"), (3, "b")] + ([] if function else [(3, "a")])
    )
    cpl_program.walk(Union((
        Implication(Z(), Conjunction((R(x), F(x, y), T(y)))),
    )))
    query = Implication(ans(), Z())
    if not function:
        with pytest.raises(NonLiftableException):
            dalvi_suciu_lift.solve_succ_query(query, cpl_program)
        return
    result = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    expected = 1 - (1 - 0.6 * (1 - 0.5 * 0.6)) * (1 - 0.7 * 0.3)
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


def test_functional_dependency_in_one_disjunct_only():
    """A tuple of R in the subqueries of f(x) and of g(x): y is not a
    separator of ∃y (∃x R(x) ∧ F(x, y) ∧ T(y)) ∨ (∃x R(x) ∧ G(x, y))."""
    R, F, G, T = (Symbol(n) for n in ("R", "F", "G", "T"))
    x, y = Symbol("x"), Symbol("y")
    symbol_table = {
        R: ProbabilisticFactSet(Symbol("r"), Constant(0)),
        T: ProbabilisticFactSet(Symbol("t"), Constant(0)),
        F: DeterministicFactSet(Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(("a", "b"), [(1, 1), (2, 2)])
        )),
        G: DeterministicFactSet(Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(("a", "b"), [(1, 2), (2, 1)])
        )),
    }
    conjunction = Conjunction((R(x), F(x, y), T(y)))
    query = ExistentialPredicate(y, ExistentialPredicate(x, Disjunction((
        conjunction, Conjunction((R(x), G(x, y)))
    ))))
    assert dalvi_suciu_lift.find_separator_variables(
        query, symbol_table
    )[0] == set()
    assert y in dalvi_suciu_lift.find_separator_variables(
        ExistentialPredicate(y, ExistentialPredicate(x, conjunction)),
        symbol_table
    )[0]
