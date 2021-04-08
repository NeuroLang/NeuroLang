from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...expressions import Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication
)
from .. import dalvi_suciu_lift

TNRA = TranslateToNamedRA()


class SymbolFactory:
    def __getattribute__(self, name):
        return Symbol(name)


S = SymbolFactory()


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
            dalvi_suciu_lift.compute_syntactically_independent_splits_if_possible(
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
        dalvi_suciu_lift.compute_syntactically_independent_splits_if_possible(
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

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression)
    assert sv == {y}

    expression = ExistentialPredicate(
        y,
        Disjunction((B(x, y), Conjunction((A(y), B(y, x))), C(y)))
    )

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression)
    assert sv == set()

    expression = Disjunction((B(x, y), Conjunction((A(y), B(x, y))), C(y)))

    sv, _ = dalvi_suciu_lift.find_separator_variables(expression)
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

    assert dalvi_suciu_lift.has_separator_variables(expression)
    assert dalvi_suciu_lift.find_separator_variables(expression)[0] & {x1, x2}


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

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq)
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
    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq)
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
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

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq)
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

    plan = dalvi_suciu_lift.dalvi_suciu_lift(cq)
    res = dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert res

