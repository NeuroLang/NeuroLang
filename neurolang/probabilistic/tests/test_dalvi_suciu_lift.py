from typing import AbstractSet

from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...expressions import Symbol, Constant
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication
)
from .. import transforms
from .. import dalvi_suciu_lift
from ...relational_algebra import (
    Projection, NameColumns, NaturalJoin, ColumnInt, ColumnStr, Union,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    WeightedNaturalJoin,
    DisjointProjection,
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..probabilistic_ra_utils import (
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
)

TNRA = TranslateToNamedRA()


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


def test_lifted_bcq_fig_4_():
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

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


def test_lifed_separator_variable():
    Q = Symbol('Q')
    R = Symbol('R')
    x1 = Symbol('x1')
    z = Symbol('z')

    cq = Implication(Q(z), R(x1, z))
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
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
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
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
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
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert not res


def test_lifted_bcq_fig_4_4():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    U = Symbol('U')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), U(x2, y2)
    )))

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res


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

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq, {})
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
    col_0 = Constant(ColumnInt(0))
    col_1 = Constant(ColumnInt(1))
    col_x = Constant(ColumnStr("x"))
    col_y = Constant(ColumnStr("y"))
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
    col_0 = Constant(ColumnInt(0))
    col_1 = Constant(ColumnInt(1))
    col_x1 = Constant(ColumnStr("x1"))
    col_y1 = Constant(ColumnStr("y1"))
    col_x2 = Constant(ColumnStr("x2"))
    col_y2 = Constant(ColumnStr("y2"))
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
