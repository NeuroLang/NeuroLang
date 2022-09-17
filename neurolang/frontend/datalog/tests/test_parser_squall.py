from inspect import signature
from itertools import product
from operator import add, eq, mul, pow, sub, truediv

import pytest

from .... import config
from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....expression_pattern_matching import add_match
from ....expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ....expressions import Constant, Query, Symbol
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate
)
from ....probabilistic.expressions import ProbabilisticPredicate
from ..squall_syntax import LogicSimplifier, parser
from ..standard_syntax import ExternalSymbol

config.disable_expression_type_printing()


EQ = Constant(eq)


class LogicWeakEquivalence(ExpressionWalker):
    @add_match(EQ(UniversalPredicate, UniversalPredicate))
    def eq_universal_predicates(self, expression):
        left, right = expression.args
        if left.head != right.head:
            new_head = Symbol.fresh()
            rew = ReplaceExpressionWalker(
                {left.head: new_head, right.head: new_head}
            )
            left = rew.walk(left)
            right = rew.walk(right)

        return self.walk(EQ(left.body, right.body))

    @add_match(EQ(ExistentialPredicate, ExistentialPredicate))
    def eq_existential_predicates(self, expression):
        return self.eq_universal_predicates(expression)

    @add_match(
        EQ(NaryLogicOperator, NaryLogicOperator),
        lambda exp: (
            len(exp.args[0].formulas) == len(exp.args[1].formulas) and
            type(exp.args[0]) == type(exp.args[1])
        )
    )
    def n_ary_sort(self, expression):
        left, right = expression.args

        return all(
            self.walk(EQ(a1, a2))
            for a1, a2 in zip(
                sorted(left.formulas, key=repr),
                sorted(right.formulas, key=repr)
            )
        )

    @add_match(EQ(LogicOperator, LogicOperator))
    def eq_logic_operator(self, expression):
        left, right = expression.args
        for l, r in zip(left.unapply(), right.unapply()):
            if isinstance(l, tuple) and isinstance(r, tuple):
                return (
                    (len(l) == len(r)) and
                    all(
                        self.walk(EQ(ll, rr))
                        for ll, rr in zip(l, r)
                    )
                )
            else:
                return self.walk(EQ(l, r))

    @add_match(EQ(..., ...))
    def eq_expression(self, expression):
        return expression.args[0] == expression.args[1]


def weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return LogicWeakEquivalence().walk(EQ(left, right))


@pytest.fixture(scope="module")
def nouns():
    return [
        ("'marseillaise'", lambda x: x(Constant('marseillaise'))),
        ("?x", lambda x:x(Symbol('x')))
    ]


@pytest.fixture(scope="module")
def verb1():
    return [
        ("plays", lambda x: Symbol("plays")(x)),
    ]


@pytest.fixture(scope="module")
def verb2():
    return [
        ("~sings", lambda x, y: Symbol("sings")(x, y))
    ]


@pytest.fixture(scope="module")
def verbs(verb1, verb2):
    return verb1 + verb2


def op_application(np):
    return lambda d: np(lambda y: d(y))


@pytest.fixture(scope="module")
def op(noun_phrases):
    return [
        (np[0], op_application(np[1]))
        for np in noun_phrases
    ]


def vp_op_application(op, v2):
    return lambda x: op(lambda y: v2(x, y))


@pytest.fixture(scope="module")
def verb_phrases_do_op(verbs, op):
    return [
        (f"{v2[0]} {op[0]}", vp_op_application(op[1], v2[1]))
        for v2, op in product(verbs, op)
        if is_transitive(v2[1])
    ]


@pytest.fixture(scope="module")
def nouns_1():
    return [
        ("person", Symbol("person")),
        ("country", Symbol("country"))
    ]


def lambda_simple(arg):
    return lambda x: arg(x)


def lambda_conjunction(*args):
    return lambda x: Conjunction(tuple(a(x) for a in args))


@pytest.fixture(scope="module")
def noun_groups_1(nouns_1):
    return [
        (f"{n1[0]}", lambda_conjunction(n1[1]))
        for n1 in nouns_1
    ]


@pytest.fixture(scope="module")
def det():
    res = []

    fresh = lambda: Symbol.fresh()
    res.append((
        "every",
        lambda d1: lambda d2: (
            (
                lambda x: UniversalPredicate(x, Implication(d2(x), d1(x))))
                (fresh()
            )
        )
    ))

    for det in ("a", "an", "some"):
        res.append((
            det,
            lambda d1: lambda d2: (
                (lambda x: ExistentialPredicate(x, Conjunction((d1(x), d2(x)))))(fresh())
            )
        ))

    res.append((
        "no",
        lambda d1: lambda d2: (
            (lambda x: Negation(ExistentialPredicate(x, Conjunction((d1(x), d2(x))))))(fresh())
        )
    ))
    return res


@pytest.fixture(scope="module")
def noun_phrase_quantified_1(det, noun_groups_1):
    res = []
    for det_, ng1 in product(det, noun_groups_1):
        exp = det_[1](ng1[1])
        res.append((
            f"{det_[0]} {ng1[0]}", lambda_simple(exp)
        ))
    return res


@pytest.fixture(scope="module")
def verb_phrases(verbs, verb_phrases_do_op):
    return [
        v for v in verbs if not is_transitive(v[1])
    ] + verb_phrases_do_op


@pytest.fixture(scope="module")
def noun_phrases(nouns, noun_phrase_quantified_1):
    return nouns + noun_phrase_quantified_1


@pytest.fixture(scope="module")
def s_np_vp(noun_phrases, verb_phrases):
    return [
        (f"{np[0]} {vp[0]}", np[1](vp[1]))
        for np, vp in product(noun_phrases, verb_phrases)
    ]


@pytest.fixture(scope="module")
def s_for(s_np_vp, noun_phrases):
    res = list(s_np_vp)
    for _ in range(1):
        res += [
            (f"for {np[0]}, {s[0]}", np[1](lambda x: s[1]))
            for np, s in product(noun_phrases, res)
        ]
    res = res[len(s_np_vp):]
    return res


@pytest.fixture(scope="module")
def s(s_np_vp, s_for):
    return s_np_vp + s_for


def test_squall_s(s):
    for query, expected in s:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def is_transitive(verb):
    return len(signature(verb).parameters) > 1


def test_squall_simple_np_nv(noun_phrases, verb_phrases):
    query_result_pairs = []
    for np, vp in product(noun_phrases, verb_phrases):
        query = f"{np[0]} {vp[0]}"
        result = np[1](vp[1])
        query_result_pairs.append((query, result))

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_quantified_np_nv(noun_phrase_quantified_1, verb_phrases):
    query_result_pairs = []
    for np, vp in product(noun_phrase_quantified_1, verb_phrases):
       query_result_pairs.append(
            (f"{np[0]} {vp[0]}", np[1](vp[1]))
        )

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_quantified_np_nv2(
    noun_phrase_quantified_1, verb2, noun_phrases
):
    query_result_pairs = []
    for npq, vp, np in product(
        noun_phrase_quantified_1, verb2, noun_phrases
    ):
        vp_ = lambda x: np[1](lambda y: vp[1](x, y))
        query_result_pairs.append(
            (f"{npq[0]} {vp[0]} {np[0]}", npq[1](vp_))
        )

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_voxel_activation():
    query = "every voxel (?x; ?y; ?z) that a study ?s ~reports activates"
    res = parser(f"squall {query}")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    s = Symbol("s")
    voxel = Symbol("voxel")
    study = Symbol("study")
    reports = Symbol("reports")
    activates = Symbol("activates")
    expected = UniversalPredicate(
        z,
        UniversalPredicate(
            y,
            UniversalPredicate(
                x,
                Implication(
                    activates(x, y, z),
                    Conjunction((
                        voxel(x, y, z),
                        ExistentialPredicate(
                            s,
                            Conjunction((
                                reports(s, x, y, z),
                                study(s)
                            ))
                        )

                    ))
                )
            )
        )
    )

    assert res == expected


def test_squall():
    # res = parser("squall ?t1 runs")
    # res = parser("squall every woman sees a man")
    # res = parser("squall a woman that eats sees a man that sleeps")
    res = parser("squall ?s reports")
    parser("squall every voxel that a study reports is an activation")
    parser("squall every voxel that 'a' reports activates")
    parser("squall every voxel that a study that mentions a word that is 'pepe' reports activates")
    parser("squall every voxel that a study that reports no region reports activates")
    parser("squall every voxel that a study -that reports no region- reports activates")

    assert res


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
        Query(
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
