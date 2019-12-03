from operator import add, eq, mul, pow, sub, truediv

from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....expressions import Constant, Symbol
from .. import ExternalSymbol, parser


def test_facts():
    res = parser('A(3)')
    assert res == Union((Fact(Symbol('A')(Constant(3.))),))

    res = parser('A("x")')
    assert res == Union((Fact(Symbol('A')(Constant('x'))),))

    res = parser("A('x', 3)")
    assert res == Union((Fact(Symbol('A')(Constant('x'), Constant(3.))),))

    res = parser(
        'A("x", 3)\n'
        'ans():-A(x, y)'
    )
    assert res == Union((
        Fact(Symbol('A')(Constant('x'), Constant(3.))),
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

    res = parser('A(x):-B(x, y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(x, y), C(Constant(3), z), Constant(eq)(z, Constant(4.))
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
                        Constant(mul)(Constant(5.), Constant(2.))),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4.))
            ))
        ),
    ))

    res = parser('A(x):-B(x / 2, y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(truediv)(x, Constant(2.)),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4.))
            ))
        ),
    ))

    res = parser('A(x):-B(f(x), y), C(3, z), z == 4')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(f(x), y),
                C(Constant(3), z), Constant(eq)(z, Constant(4.))
            ))
        ),
    ))

    res = parser('A(x):-B(x + (-5), "a")')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(add)(x, Constant(-5.)),
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
                        Constant(mul)(Constant(5.), Constant(2.))
                    ),
                    Constant(pow)(ExternalSymbol('y'), Constant(-2.))
                ),
            ))
        ),
    ))
