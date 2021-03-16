from neurolang.logic.transformations import convert_to_pnf_with_cnf_matrix
from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...expressions import Symbol
from ...logic import Conjunction, Disjunction, ExistentialPredicate, Implication, Negation
from ...relational_algebra_provenance import NaturalJoin, Projection, Union
from .. import lift_r

TNRA = TranslateToNamedRA()


def test_sort_independent_splits_trivial():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')

    for expression in (
        Disjunction((A(x), B(x), C(x))),
        Disjunction((Conjunction((A(x), B(x))), C(x)))
    ):
        res, splittable = \
            lift_r.compute_syntactically_independent_splits_if_possible(
                expression,
                Disjunction
            )
        assert splittable
        assert res == expression


def test_sort_independent_splits():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')

    expression = Disjunction((A(x), Conjunction((A(x), B(x))), C(x)))
    res, splittable = \
        lift_r.compute_syntactically_independent_splits_if_possible(
            expression,
            Disjunction
        )
    assert splittable
    assert (
        set(res.formulas) ==
        set((
            Disjunction((A(x), Conjunction((A(x), B(x))))),
            C(x)
        ))
    )


def test_has_separator_variable():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')
    y = Symbol('y')

    expression = Implication(
        A(x),
        Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))
    )

    sv, _ = lift_r.find_separator_variables(expression)
    assert sv == {y}

    expression = Implication(
        A(x),
        Disjunction((B(x, y), Conjunction((A(y), B(y, x))), C(y)))
    )

    sv, _ = lift_r.find_separator_variables(expression)
    assert sv == set()

    expression = Implication(
        A(x, y),
        Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))
    )

    sv, _ = lift_r.find_separator_variables(expression)
    assert sv == set()

    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')

    expression = Implication(
        Q(),
        Disjunction((
            Conjunction((R(x1), S(x1, y1))),
            Conjunction((T(x2), S(x2, y2)))
        ))
    )

    assert lift_r.has_separator_variables(expression)
    assert lift_r.find_separator_variables(expression)[0] & {x1, x2}


def test_lifted_bcq():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y0 = Symbol('y0')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    y3 = Symbol('y3')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    # bcq = Implication(Q(), Conjunction((R(x), S(x, y))))
    # plan = lift_r.LiftRAlgorithm().walk(bcq)

    # assert plan == Projection(
    #     NaturalJoin(
    #         TNRA.walk(R(x)),
    #         Projection(TNRA.walk(S(x, y)), (x,))
    #     ),
    #     tuple()
    # )

    # cq = Implication(Q(z), Disjunction((
    #     Conjunction((R(z, x1), S(x1, y1))),
    #     Conjunction((S(x2, y2), T(z, y2))),
    #     R(z, x3),
    #     T(z, y3)
    # )))
    # plan = lift_r.LiftRAlgorithm().walk(cq)

    cq = Implication(Q(z), Disjunction((
        Conjunction((R(z, x1), S(x1, y1))),
        Conjunction((S(x2, y2), T(z, y2))),
        Conjunction((R(z, x3), T(z, y3))),
    )))

    # cq = Implication(Q(z), Conjunction((
    #     Disjunction((R(z, x3), Conjunction((S(x2, y2), T(z, y2))))),
    #     Disjunction((Conjunction((R(z, x1), S(x1, y1))), T(z, y3)))
    # )))

    # S1 = Symbol('S1')
    # S2 = Symbol('S2')
    # S3 = Symbol('S3')
    # cq = Implication(Q(z), Conjunction((
    #     R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    # )))
    plan = lift_r.LiftRAlgorithm().walk(cq)
    res = lift_r.is_pure_lifted_plan(plan)
    assert res


def test_containment():
    Q = Symbol('Q')
    R = Symbol('R')
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    r1 = Implication(
        Q(x), Conjunction((R(x, y), R(y, z), R(z, w)))
    )
    r2 = Implication(
        Q(x), Conjunction((R(x, y), R(y, x)))
    )

    assert lift_r.is_contained(r2, r1)
    assert not lift_r.is_contained(r1, r2)
