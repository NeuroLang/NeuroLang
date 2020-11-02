from operator import add, eq, mul, pow, sub, truediv

from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....expressions import Constant, Symbol
from ..standard_syntax import ExternalSymbol
from ..natural_syntax import parser
from ....probabilistic.expressions import ProbabilisticPredicate


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

    res = parser('"john" is cat')
    assert res == Union((Fact(Symbol('cat')(Constant("john"))),))

    res = parser('"john" is `http://owl#fact`')
    assert res == Union((Fact(Symbol('http://owl#fact')(Constant("john"))),))

    res = parser('"john" is "perceval"\'s mascot')
    assert res == Union((
        Fact(Symbol('mascot')(Constant("john"), Constant("perceval"))),
    ))

    res = parser('"john" has 4 legs')
    assert res == Union((
        Fact(Symbol('legs')(Constant("john"), Constant(4.))),
    ))

    res = parser('"john" has 4 legs')
    assert res == Union((
        Fact(Symbol('legs')(Constant("john"), Constant(4.))),
    ))

    res = parser('"john" is below the "table"')
    assert res == Union((
        Fact(Symbol('below')(Constant("john"), Constant("table"))),
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


def test_nl_rules():
    cat = Symbol('cat')
    bird = Symbol('bird')
    feline = Symbol('feline')
    legs = Symbol('legs')
    small = Symbol('small')
    goodluck_cat = Symbol('goodluck_cat')
    black = Symbol('black')

    x = Symbol('x')
    y = Symbol('y')

    res = parser('x is cat if x is feline, x has 4 legs, x is small')
    assert res == Union((
        Implication(
            cat(x), Conjunction((feline(x), legs(x, Constant(4.)), small(x)))
        ),
    ))

    res = parser('x is goodluck_cat if x is cat, not x is black')
    assert res == Union((
        Implication(
            goodluck_cat(x), Conjunction((cat(x), Negation(black(x))))
        ),
    ))

    res = parser('''
        x has y legs if x is a cat & y == 4.0
        or x has y legs if x is a bird and y == 2
    ''')
    assert res == Union((
        Implication(
            legs(x, y), Conjunction((cat(x), Constant(eq)(y, Constant(4.))))
        ),
        Implication(
            legs(x, y), Conjunction((bird(x), Constant(eq)(y, Constant(2))))
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


def test_aggregation_nsd():
    A = Symbol('A')
    B = Symbol('B')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    res = parser('x has f(y) A if B(x, y)')
    assert res == Union((
        Implication(
            A(x, AggregationApplication(f, (y,))),
            Conjunction((B(x, y),))
        ),
    ))


def test_uri_nsd():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(name='http://sig.biostr.washington.edu/fma3.0#regional_part_of')
    x = Symbol('x')

    res = parser(f'x is `{str(label.name)}` if x is `{str(regional_part.name)}`')
    expected_result = Union((
        Implication(
            label(x),
            Conjunction((regional_part(x),))
        ),
    ))

    assert res == expected_result


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


def test_probabilistic_fact_nsd():
    cat = Symbol('cat')
    p = Symbol('p')
    res = parser('with probability p "john" is cat')
    assert res == Union((
        Implication(
            ProbabilisticPredicate(p, cat(Constant("john"))),
            Constant(True)
        ),
    ))

    res = parser("with probability p 'john' is 'george''s cat")
    assert res == Union((
        Implication(
            ProbabilisticPredicate(p, cat(Constant("john"), Constant("george"))),
            Constant(True)
        ),
    ))
