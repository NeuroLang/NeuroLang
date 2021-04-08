from ...expressions import Symbol
from ...logic import (
    Conjunction,
    ExistentialPredicate,
    Implication
)
from .. import dalvi_suciu_lift


def test_rule_e_query():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    rule = Implication(R(x, y), S(x, y))
    e_query = S(x, y)

    res = dalvi_suciu_lift.convert_rule_to_e_query(rule)
    assert e_query == res

    rule = Implication(R(x), S(x, y))
    e_query = ExistentialPredicate(y, S(x, y))

    res = dalvi_suciu_lift.convert_rule_to_e_query(rule)
    assert e_query == res

    rule = Implication(R(x), Conjunction((S(x, y), T(x))))
    e_query = Conjunction((ExistentialPredicate(y, S(x, y)), T(x)))

    res = dalvi_suciu_lift.convert_rule_to_e_query(rule)
    assert e_query == res

    rule = Implication(R(x), Conjunction((S(x, y), T(z, x))))
    e_query = Conjunction((
        ExistentialPredicate(z, T(z, x)),
        ExistentialPredicate(y, S(x, y)),
    ))

    res = dalvi_suciu_lift.convert_rule_to_e_query(rule)
    assert e_query == res
