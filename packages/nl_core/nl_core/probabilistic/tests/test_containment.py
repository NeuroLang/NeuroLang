from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation,
)
from ..containment import is_contained, is_contained_rule


def test_containment_conjunctive_query():
    R = Symbol('R')
    S = Symbol('S')
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')

    r1 = ExistentialPredicate(y, ExistentialPredicate(
        w,
        Conjunction((R(x, y), R(x, w)))
    ))

    r2 = ExistentialPredicate(y, R(x, y))

    r3 = ExistentialPredicate(y, S(x, y))

    r4 = ExistentialPredicate(y, ExistentialPredicate(
        w,
        Conjunction((R(x, y), S(x, w)))
    ))

    assert is_contained(r1, r1)
    assert is_contained(r2, r1)
    assert is_contained(r1, r2)
    assert not is_contained(r1, r3)
    assert not is_contained(r3, r1)
    assert is_contained(r1, r4)
    assert is_contained(r3, r4)
    assert not is_contained(r4, r1)
    assert not is_contained(r4, r3)


def test_containment_disjunctive_query():
    R = Symbol('R')
    S = Symbol('S')
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')

    r1 = ExistentialPredicate(y, ExistentialPredicate(
        w,
        Disjunction((R(x, y), R(x, w)))
    ))

    r2 = ExistentialPredicate(y, R(x, y))

    r3 = ExistentialPredicate(y, S(x, y))

    r4 = ExistentialPredicate(y, ExistentialPredicate(
        w,
        Disjunction((R(x, y), S(x, w)))
    ))

    assert is_contained(r1, r1)
    assert is_contained(r2, r1)
    assert is_contained(r1, r2)
    assert not is_contained(r1, r3)
    assert not is_contained(r3, r1)
    assert is_contained(r4, r1)
    assert is_contained(r4, r3)
    assert not is_contained(r1, r4)
    assert not is_contained(r3, r4)


def test_containment_query_cross_product():
    R = Symbol('R')
    x = Symbol('x')
    y = Symbol('y')

    r1 = R(x)
    r2 = R(y)

    assert not is_contained(r2, r1)
    assert not is_contained(r1, r2)


def test_containment_query():
    R = Symbol('R')
    x = Symbol('x')
    y = Symbol('y')

    r1 = ExistentialPredicate(
        y,
        Conjunction((R(x, y), R(y, x), R(x, x)))
    )
    r2 = ExistentialPredicate(y, Conjunction((R(x, y), R(y, x))))

    assert is_contained(r2, r1)
    assert not is_contained(r1, r2)


def test_containment_mixed():
    R = Symbol('R')
    S = Symbol('S')
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    r1 = ExistentialPredicate(y, ExistentialPredicate(
        w,
        Conjunction((R(x, y), R(y, x), R(w, w)))
    ))

    r2 = ExistentialPredicate(
        y,
        Conjunction((R(x, y), R(y, x)))
    )

    r3 = ExistentialPredicate(
        y,
        R(x, y)
    )

    r4 = ExistentialPredicate(
        y,
        Disjunction((R(x, y), S(x, y)))
    )

    r5 = ExistentialPredicate(
        y,
        S(x, y)
    )

    r6 = ExistentialPredicate(y, ExistentialPredicate(
        z,
        Conjunction((Disjunction((R(x, y), S(x, y))), S(x, z)))
    ))

    assert is_contained(r1, r1)
    assert is_contained(r2, r1)
    assert not is_contained(r1, r2)
    assert is_contained(r3, r1)
    assert is_contained(r3, r1)
    assert is_contained(r4, r3)
    assert is_contained(r5, r6)
    assert not is_contained(r3, r5)
    assert not is_contained(r5, r3)
    assert is_contained(r6, r5)


def test_containment_different_free_variables():
    R = Symbol('R')
    T = Symbol('T')
    t = Symbol('t')
    x = Symbol('x')
    y = Symbol('y')

    r_xy = R(x, y)
    r_x = ExistentialPredicate(t, R(x, t))
    r_x_or_t_xy = Disjunction((r_x, T(x, y)))

    # R(x, y) implies ∃t R(x, t) for every y
    assert is_contained(r_x, r_xy)
    # but ∃t R(x, t) does not imply R(x, y), which depends on y
    assert not is_contained(r_xy, r_x)
    # ∃t R(x, t) implies ∃t R(x, t) ∨ T(x, y) for every y
    assert is_contained(r_x_or_t_xy, r_x)
    assert not is_contained(r_x, r_x_or_t_xy)


def test_containment_independent_of_variable_names_order():
    R = Symbol('R')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    # The disjuncts have different free variables, the containment
    # test must not depend on the order in which they are collected.
    assert is_contained(Disjunction((T(y), R(x, y))), R(x, y))
    assert is_contained(Disjunction((R(x, y), T(y))), R(x, y))


def test_containment_of_closed_queries():
    """Queries without free variables give nullary heads, whose facts
    must be found in the chase solution."""
    S_ = Symbol("S_")
    T_ = Symbol("T_")
    x_ = Symbol("x_")
    y_ = Symbol("y_")
    q1 = ExistentialPredicate(x_, S_(x_))
    q2 = ExistentialPredicate(y_, S_(y_))
    assert is_contained(q1, q2)
    assert is_contained(q2, q1)
    assert not is_contained(ExistentialPredicate(x_, T_(x_)), q1)


def test_containment_of_rules_with_negations():
    """
    A negated atom is matched by the same negated atom of the other
    rule, not frozen as a fact (``M(s, a)`` was one of ``not M(s, a)``)
    nor tested by its absence from the facts.
    """
    M, C, Q, s = Symbol('M'), Symbol('C'), Symbol('Q'), Symbol('s')
    a = Constant('a')

    def rule(body):
        return Implication(Q(s), body)

    not_m = Negation(M(s, a))
    both = Conjunction((C(s), not_m))
    # (q1, q2, whether q2 implies q1)
    cases = [
        (rule(M(s, a)), rule(not_m), False),
        (rule(not_m), rule(M(s, a)), False),
        (rule(not_m), rule(both), True),
        (rule(C(s)), rule(both), True),
        (rule(both), rule(C(s)), False),
        (rule(both), rule(both), True),
    ]
    for q1, q2, expected in cases:
        assert is_contained_rule(q1, q2) == expected, (q1, q2)
