from ...expressions import Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation
)
from .. import transforms


def test_rule_ucq():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    rule = Implication(R(x, y), S(x, y))
    ucq = S(x, y)

    res = transforms.convert_rule_to_ucq(rule)
    assert ucq == res

    rule = Implication(R(x), S(x, y))
    ucq = ExistentialPredicate(y, S(x, y))

    res = transforms.convert_rule_to_ucq(rule)
    assert ucq == res

    rule = Implication(R(x), Conjunction((S(x, y), T(x))))
    ucq = Conjunction((ExistentialPredicate(y, S(x, y)), T(x)))

    res = transforms.convert_rule_to_ucq(rule)
    assert ucq == res

    rule = Implication(R(x), Conjunction((S(x, y), T(z, x))))
    ucq = Conjunction((
        ExistentialPredicate(z, T(z, x)),
        ExistentialPredicate(y, S(x, y)),
    ))

    res = transforms.convert_rule_to_ucq(rule)
    assert ucq == res


def test_unify_existential_variables():
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    expression = Disjunction((
        ExistentialPredicate(x, T(x)),
        ExistentialPredicate(y, T(y))
    ))

    res = transforms.unify_existential_variables(expression)

    assert res == ExistentialPredicate(y, Disjunction((T(y), T(y))))

    expression = Disjunction((
        Conjunction((S(x, y), ExistentialPredicate(x, T(x)))),
        ExistentialPredicate(y, T(y))
    ))

    res = transforms.unify_existential_variables(expression)

    assert isinstance(res, ExistentialPredicate)
    e_var = res.head
    expected = ExistentialPredicate(
        e_var,
        Disjunction((
            Conjunction((S(x, y), T(e_var))),
            T(e_var)
        ))
    )
    assert res == expected


def test_rule_with_negation_to_ucq():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Implication(R(x), Conjunction((S(x, y), Negation(T(y)))))

    expected = ExistentialPredicate(y, Conjunction((S(x, y), Negation(T(y)))))
    res = transforms.convert_rule_to_ucq(rule)

    assert expected == res

    rule = Implication(R(x), Conjunction((S(x, y), S(x, x), Negation(T(y)))))
    expected = Conjunction((
        ExistentialPredicate(y, Conjunction((S(x, y), Negation(T(y))))),
        S(x, x)
    ))
    res = transforms.convert_rule_to_ucq(rule)

    assert expected == res
