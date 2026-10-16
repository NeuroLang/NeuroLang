from ...expressions import Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate
)
from ..containment import is_contained


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
