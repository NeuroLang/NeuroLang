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
