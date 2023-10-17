from operator import add, eq, lt, mul, pow, sub, truediv

import pytest


from ....logic import ExistentialPredicate

from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....expressions import (
    Command,
    Constant,
    FunctionApplication,
    Lambda,
    Query,
    Statement,
    Symbol
)
from ....exceptions import UnexpectedTokenError
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticFact
)
from ..standard_syntax import ExternalSymbol, parser


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
        Query(
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
    print("_________________________________")
    print("A(x):-B(x, y), C(3, z)")
    res = parser('A(x):-B(x, y), C(3, z)')
    assert res == Union((
        Implication(A(x), Conjunction((B(x, y), C(Constant(3), z)))),
    ))

    print("_________________________________")
    print("A(x):-~B(x)")
    res = parser('A(x):-~B(x)')
    assert res == Union((
        Implication(A(x), Conjunction((Negation(B(x)),))),
    ))

    print("_________________________________")
    print("A(x):-B(x, ...)")
    res = parser('A(x):-B(x, ...)')
    fresh_arg = res.formulas[0].antecedent.formulas[0].args[1]
    assert isinstance(fresh_arg, Symbol)
    assert res == Union((
        Implication(A(x), Conjunction((B(x, fresh_arg),))),
    ))

    print("_________________________________")
    print("A(x):-B(x, y), C(3, z), (z == 4)")
    res = parser('A(x):-B(x, y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(x, y), C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    print("_________________________________")
    print("A(x):-B(x + 5 * 2, y), C(3, z), (z == 4)")
    res = parser('A(x):-B(x + 5 * 2, y), C(3, z), (z == 4)')
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

    print("_________________________________")
    print("A(x):-B(x / 2, y), C(3, z), (z == 4)")
    res = parser('A(x):-B(x / 2, y), C(3, z), (z == 4)')
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

    print("_________________________________")
    print("A(x):-B(f(x), y), C(3, z), (z == 4)")
    res = parser('A(x):-B(f(x), y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(f(x), y),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    print("_________________________________")
    print('A(x):-B(x + (-5), "a")')
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

    print("_________________________________")
    print("A(x):-B(x - 5 * 2, @y ** -2)")
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
    expected_result = Union((
        Implication(
            A(x, FunctionApplication(f, (y,))),
            Conjunction((B(x, y),))
        ),
    ))
    assert res == expected_result


def test_uri():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )
    x = Symbol('x')
    y = Symbol('y')

    res = parser(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)')
    expected_result = Union((
        Implication(
            label(x),
            Conjunction((regional_part(x, y),))
        ),
    ))

    assert res == expected_result


def test_probabilistic_fact():
    A = Symbol('A')
    p = Symbol('p')
    res = parser('p::A(3)')
    assert res == Union((
        Implication(
            ProbabilisticFact(p, A(Constant(3.))),
            Constant(True)
        ),
    ))

    res = parser('0.8::A("a b", 3)')
    assert res == Union((
        Implication(
            ProbabilisticFact(
                Constant(0.8),
                A(Constant("a b"), Constant(3.))
            ),
            Constant(True)
        ),
    ))

    exp = Symbol("exp")
    d = Symbol("d")
    x = Symbol("x")
    B = Symbol("B")
    res = parser("B(x) :: exp(-d / 5.0) :- A(x, d) & (d < 0.8)")
    expected = Union(
        (
            Implication(
                ProbabilisticFact(
                    FunctionApplication(
                        exp,
                        (
                            Constant(truediv)(
                                Constant(mul)(Constant(-1), d), Constant(5.0)
                            ),
                        ),
                    ),
                    B(x),
                ),
                Conjunction((A(x, d), Constant(lt)(d, Constant(0.8)))),
            ),
        )
    )
    assert res == expected


def test_condition():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')
    res = parser('C(x) :- A(x) // B(x)')

    expected = Union((
        Implication(
            C(x),
            Condition(A(x), B(x))
        ),
    ))

    assert res == expected

    res = parser('C(x) :- (A(x), B(x)) // B(x)')

    expected = Union((
        Implication(
            C(x),
            Condition(Conjunction((A(x), B(x))), B(x))
        ),
    ))

    assert res == expected

    res = parser('C(x) :- A(x) // (A(x), B(x))')

    expected = Union((
        Implication(
            C(x),
            Condition(A(x), Conjunction((A(x), B(x))))
        ),
    ))

    assert res == expected


def test_existential():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    s1 = Symbol("s1")
    s2 = Symbol("s2")

    res = parser("C(x) :- B(x), exists(s1; A(s1))")
    expected = Union(
        (
            Implication(
                C(x), Conjunction((B(x), ExistentialPredicate(s1, A(s1))))
            ),
        )
    )
    assert res == expected

    res = parser("C(x) :- B(x), ∃(s1 st A(s1))")
    assert res == expected

    res = parser("C(x) :- B(x), exists(s1, s2; A(s1), A(s2))")

    expected = Union(
        (
            Implication(
                C(x),
                Conjunction(
                    (
                        B(x),
                        ExistentialPredicate(
                            s2,
                            ExistentialPredicate(
                                s1, Conjunction((A(s1), A(s2)))
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    assert res == expected

    # try :
    #     res = parser("C(x) :- B(x), exists(s1; )")
    # except UnexpectedToken as e:
    #     raise UnexpectedTokenError from e
    with pytest.raises(UnexpectedTokenError):
        res = parser("C(x) :- B(x), exists(s1; )")

def test_query():
    ans = Symbol("ans")
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("ans(x) :- B(x, y), C(3, y)")
    assert res == Union(
        (Query(ans(x), Conjunction((B(x, y), C(Constant(3), y)))),)
    )


def test_prob_implicit():
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("B(x, PROB, y) :- C(x, y)")
    assert res == Union(
        (Implication(B(x, PROB(x, y), y), Conjunction((C(x, y),))),)
    )


def test_prob_explicit():
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("B(x, PROB(x, y), y) :- C(x, y)")
    assert res == Union(
        (Implication(B(x, PROB(x, y), y), Conjunction((C(x, y),))),)
    )


def test_lambda_definition():
    c = Symbol("c")
    x = Symbol("x")

    res = parser("c := lambda x: x + 1")
    expression = Lambda((x,), FunctionApplication(
        Constant(add), (x, Constant[int](1))
    ))
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_lambda_definition_statement():
    c = Symbol("c")
    x = Symbol("x")
    y = Symbol("y")

    res = parser("c(x, y) := x + y")
    expression = Lambda((x, y), FunctionApplication(
        Constant(add), (x, y)
    ))
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_lambda_application():
    c = Symbol("c")
    x = Symbol("x")

    res = parser("c := (lambda x: x + 1)(2)")
    expression = FunctionApplication(
        Lambda((x,), FunctionApplication(
            Constant(add), (x, Constant[int](1))
        )),
        (Constant[int](2),)
    )
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_command_syntax():
    res = parser('.load_csv(A, "http://myweb/file.csv", B)')
    expected = Union(
        (
            Command(
                Symbol("load_csv"),
                (Symbol("A"), Constant("http://myweb/file.csv"), Symbol("B")),
                (),
            ),
        )
    )
    assert res == expected

    res = parser('.load_csv("http://myweb/file.csv")')
    expected = Union(
        (
            Command(
                Symbol("load_csv"), (Constant("http://myweb/file.csv"),), ()
            ),
        )
    )
    assert res == expected

    res = parser(".load_csv()")
    expected = Union((Command(Symbol("load_csv"), (), ()),))
    assert res == expected

    res = parser('.load_csv(sep=",")')
    expected = Union(
        (Command("load_csv", (), ((Symbol("sep"), Constant(",")),)),)
    )
    assert res == expected

    res = parser('.load_csv(sep=",", header=None, index_col=0)')
    expected = Union(
        (
            Command(
                "load_csv",
                (),
                (
                    (Symbol("sep"), Constant(",")),
                    (Symbol("header"), Symbol("None")),
                    (Symbol("index_col"), Constant(0)),
                ),
            ),
        )
    )
    assert res == expected

    res = parser('.load_csv(A, "http://myweb/file.csv", sep=",", header=None)')
    expected = Union(
        (
            Command(
                "load_csv",
                (Symbol("A"), "http://myweb/file.csv"),
                (
                    (Symbol("sep"), Constant(",")),
                    (Symbol("header"), Symbol("None")),
                ),
            ),
        )
    )
    assert res == expected

def test_autocompletion():
    print("")
    print("input : ''")
    res = parser('', interactive=True)
    expected = {'DOT', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'NEG_UNICODE', 'LAMBDA', 'INT', 'TRUE', 'TEXT', 'FLOAT', 'AT', 'LPAR', 'ANS', 'FALSE', 'EXISTS', 'MINUS', 'TILDE'}
    assert res == expected
    # assert res == 0
    print("res :")
    print(res)

    print("")
    print("input : .load_csv")
    res = parser('.load_csv', interactive=True)
    expected = {'LPAR'}
    assert res == expected
    # assert res == 0
    print("res :")
    print(res)
