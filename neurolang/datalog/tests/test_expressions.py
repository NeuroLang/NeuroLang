from ...expression_walker import IdentityWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from ..expressions import (Conjunction, Disjunction, Fact, Implication,
                           Negation, TranslateToLogic)


class Translator(TranslateToLogic, IdentityWalker):
    pass


def test_translation_negation():
    Q = Symbol('Q')
    x = Symbol('x')

    ttl = Translator()

    exp = ~Q(x)

    res = ttl.walk(exp)

    assert isinstance(res, Negation)
    assert res.formula == exp.args[0]


def test_translation_conjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    ttl = Translator()

    conj1 = Q(x) & R(x)
    res = ttl.walk(conj1)
    assert isinstance(res, Conjunction)
    assert res.formulas == conj1.args

    conj2 = conj1 & S(x)
    res = ttl.walk(conj2)
    assert isinstance(res, Conjunction)
    assert res.formulas == conj1.args + (conj2.args[-1],)

    conj3 = S(x) & conj1
    res = ttl.walk(conj3)
    assert isinstance(res, Conjunction)
    assert res.formulas == (conj3.args[0],) + conj1.args


def test_translation_disjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    ttl = Translator()

    disj1 = Q(x) | R(x)
    res = ttl.walk(disj1)
    assert isinstance(res, Disjunction)
    assert res.formulas == disj1.args

    disj2 = disj1 | S(x)
    res = ttl.walk(disj2)
    assert isinstance(res, Disjunction)
    assert res.formulas == disj1.args + (disj2.args[-1],)

    disj3 = S(x) | disj1
    res = ttl.walk(disj3)
    assert isinstance(res, Disjunction)
    assert res.formulas == (disj3.args[0],) + disj1.args

    conj1 = Implication(Q(x), Q(x) & R(x))
    conj2 = Implication(Q(x), R(x))
    eb = ExpressionBlock((conj1, conj2))
    res = ttl.walk(eb)
    assert isinstance(res, Disjunction)
    assert all(
        formula == ttl.walk(expression)
        for formula, expression in zip(res.formulas, eb.expressions)
    )


def test_translation_to_facts():
    Q = Symbol('Q')
    x = Symbol('x')

    ttl = Translator()

    imp = Implication(Q(x), Constant(True))
    res = ttl.walk(imp)
    assert isinstance(res, Fact)
    assert res.consequent == imp.consequent
