from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...expressions import Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication
)
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

    sv, _ = lift_r.find_separator_variables(expression)
    assert sv == {y}

    expression = ExistentialPredicate(
        y,
        Disjunction((B(x, y), Conjunction((A(y), B(y, x))), C(y)))
    )

    sv, _ = lift_r.find_separator_variables(expression)
    assert sv == set()

    expression = Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))

    sv, _ = lift_r.find_separator_variables(expression)
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

    assert lift_r.has_separator_variables(expression)
    assert lift_r.find_separator_variables(expression)[0] & {x1, x2}


def test_lifted_bcq_fig_4_():
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

    cq = Implication(Q(z), Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    )))

    plan = lift_r.dalvi_suciu_lift(cq)
    res = lift_r.is_pure_lifted_plan(plan)
    assert res


def test_lifted_join():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    cq = Implication(Q(z), Conjunction((R(x1, z), S(x1, x2), T(x1, z))))
    plan = lift_r.dalvi_suciu_lift(cq)
    res = lift_r.is_pure_lifted_plan(plan)
    assert res


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

    plan = lift_r.dalvi_suciu_lift(cq)
    res = lift_r.is_pure_lifted_plan(plan)
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

    plan = lift_r.dalvi_suciu_lift(cq)
    res = lift_r.is_pure_lifted_plan(plan)
    assert res


def test_containment_query():
    R = Symbol('R')
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    r1 = ExistentialPredicate(y, ExistentialPredicate(w, ExistentialPredicate(
        z,
        Conjunction((R(x, y), R(y, z), R(z, w)))
    )))
    r2 = ExistentialPredicate(y, Conjunction((R(x, y), R(y, x))))

    assert lift_r.is_contained(r2, r1)
    assert not lift_r.is_contained(r1, r2)


def test_containment():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
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

    r3 = Implication(
        Q(x), R(x, y)
    )

    r4 = Implication(
        Q(x), Disjunction((R(x, y), S(x, y)))
    )

    r5 = Implication(
        Q(x), S(x, y)
    )

    assert lift_r.is_contained(r1, r1)
    assert lift_r.is_contained(r2, r1)
    assert not lift_r.is_contained(r1, r2)
    assert lift_r.is_contained(r1, r3)
    assert lift_r.is_contained(r2, r3)
    assert lift_r.is_contained(r4, r3)
    assert not lift_r.is_contained(r3, r5)
    assert not lift_r.is_contained(r5, r3)
