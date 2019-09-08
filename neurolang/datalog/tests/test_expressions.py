from ...expression_walker import PatternWalker, add_match
from ...expressions import ExpressionBlock, Symbol
from ..expressions import (Conjunction, Disjunction, Implication, Negation,
                           TranslateToLogic)


class WalkAll(PatternWalker):
    @add_match(...)
    def _(self, expression):
        return expression


class Translator(TranslateToLogic, WalkAll):
    pass


def test_translation_negation():
    Q = Symbol('Q')
    x = Symbol('x')

    ttl = Translator()

    exp = ~Q(x)

    res = ttl.walk(exp)

    assert isinstance(res, Negation)
    assert res.literal == exp.args[0]


def test_translation_conjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    ttl = Translator()

    conj1 = Q(x) & R(x)
    res = ttl.walk(conj1)
    assert isinstance(res, Conjunction)
    assert res.literals == conj1.args

    conj2 = conj1 & S(x)
    res = ttl.walk(conj2)
    assert isinstance(res, Conjunction)
    assert res.literals == conj1.args + (conj2.args[-1],)

    conj3 = S(x) & conj1
    res = ttl.walk(conj3)
    assert isinstance(res, Conjunction)
    assert res.literals == (conj3.args[0],) + conj1.args


def test_translation_disjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    ttl = Translator()

    disj1 = Q(x) | R(x)
    res = ttl.walk(disj1)
    assert isinstance(res, Disjunction)
    assert res.literals == disj1.args

    disj2 = disj1 | S(x)
    res = ttl.walk(disj2)
    assert isinstance(res, Disjunction)
    assert res.literals == disj1.args + (disj2.args[-1],)

    disj3 = S(x) | disj1
    res = ttl.walk(disj3)
    assert isinstance(res, Disjunction)
    assert res.literals == (disj3.args[0],) + disj1.args

    conj1 = Implication(Q(x), Q(x) & R(x))
    conj2 = Implication(Q(x), R(x))
    eb = ExpressionBlock((conj1, conj2))
    res = ttl.walk(eb)
    assert isinstance(res, Conjunction)
    assert all(
        literal == ttl.walk(expression)
        for literal, expression in zip(res.literals, eb.expressions)
    )
