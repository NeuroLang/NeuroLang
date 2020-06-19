from operator import add, eq, mul, pow, sub, truediv

from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....probabilistic.expressions import ProbabilisticPredicate
from ....expressions import Constant, Symbol
from .. import ExternalSymbol, parser


def test_facts():
    res = parser('A(3)')
    assert res == Union((Fact(Symbol('A')(Constant(3))),))

    res = parser('A("x")')
    assert res == Union((Fact(Symbol('A')(Constant('x'))),))

    res = parser("A('x', 3)")
    assert res == Union((Fact(Symbol('A')(Constant('x'), Constant(3))),))

    res = parser(
        'A("x", 3)\n'
        '`http://uri#test-fact`("x")\n'
        'ans():-A(x, y)'
    )
    assert res == Union((
        Fact(Symbol('A')(Constant('x'), Constant(3))),
        Fact(Symbol('http://uri#test-fact')(Constant('x'))),
        Implication(
            Symbol('ans')(),
            Conjunction((
                Symbol('A')(Symbol('x'), Symbol('y')),
            ))
        )
    ))


def test_rules():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    res = parser('A(x):-B(x, y), C(3, z)')
    assert res == Union((
        Implication(A(x), Conjunction((B(x, y), C(Constant(3), z)))),
    ))

    res = parser('A(x):-~B(x)')
    assert res == Union((
        Implication(A(x), Conjunction((Negation(B(x)),))),
    ))

    res = parser('A(x):-B(x, ...)')
    fresh_arg = res.formulas[0].antecedent.formulas[0].args[1]
    assert isinstance(fresh_arg, Symbol)
    assert res == Union((
        Implication(A(x), Conjunction((B(x, fresh_arg),))),
    ))

    res = parser('A(x):-B(x, y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(x, y), C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x + 5 * 2, y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(add)(
                        x,
                        Constant(mul)(Constant(5), Constant(2))),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x / 2, y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(truediv)(x, Constant(2)),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(f(x), y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(f(x), y),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x + (-5), "a")')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(add)(x, Constant(-5)),
                    Constant("a")
                ),
            ))
        ),
    ))

    res = parser('A(x):-B(x - 5 * 2, @y ** -2)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(sub)(
                        x,
                        Constant(mul)(Constant(5), Constant(2))
                    ),
                    Constant(pow)(ExternalSymbol('y'), Constant(-2))
                ),
            ))
        ),
    ))


def test_aggregation():
    A = Symbol('A')
    B = Symbol('B')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    res = parser('A(x, f(y)):-B(x, y)')
    assert res == Union((
        Implication(
            A(x, AggregationApplication(f, (y,))),
            Conjunction((B(x, y),))
        ),
    ))


def test_probabilistic_fact():
    A = Symbol('A')
    p = Symbol('p')
    res = parser('p::A(3)')
    assert res == Union((
        Implication(
            ProbabilisticPredicate(p, A(Constant(3.))),
            Constant(True)
        ),
    ))

    res = parser('0.8::A("a b", 3)')
    assert res == Union((
        Implication(
            ProbabilisticPredicate(
                Constant(0.8),
                A(Constant("a b"), Constant(3.))
            ),
            Constant(True)
        ),
    ))
