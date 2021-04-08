from ...expressions import Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication
)
from ..containment import is_contained


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

    assert is_contained(r2, r1)
    assert not is_contained(r1, r2)


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

    assert is_contained(r1, r1)
    assert is_contained(r2, r1)
    assert not is_contained(r1, r2)
    assert is_contained(r1, r3)
    assert is_contained(r2, r3)
    assert is_contained(r4, r3)
    assert not is_contained(r3, r5)
    assert not is_contained(r5, r3)
