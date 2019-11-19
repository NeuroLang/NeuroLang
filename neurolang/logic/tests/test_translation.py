from ...expression_walker import IdentityWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from .. import Conjunction, Disjunction, Implication, Negation, Predicate
from ..expression_processing import TranslateToLogic


class Translator(TranslateToLogic, IdentityWalker):
    pass


def test_translation_negation():
    Q = Symbol('Q')
    x = Symbol('x')

    ttl = Translator()

    exp = ~Q(x)
    exp_pred = Predicate(Q, (x,))
    res = ttl.walk(exp)

    assert isinstance(res, Negation)
    assert res.formula == exp_pred


def test_translation_conjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    Qx = Predicate(Q, (x,))
    Rx = Predicate(R, (x,))
    Sx = Predicate(S, (x,))

    ttl = Translator()

    conj1 = Q(x) & R(x)
    res = ttl.walk(conj1)
    assert isinstance(res, Conjunction)
    assert res.formulas == (Qx, Rx)

    conj2 = conj1 & S(x)
    res = ttl.walk(conj2)
    assert isinstance(res, Conjunction)
    assert res.formulas == (Qx, Rx, Sx)

    conj3 = S(x) & conj1
    res = ttl.walk(conj3)
    assert isinstance(res, Conjunction)
    assert res.formulas == (Sx, Qx, Rx)


def test_translation_disjunctions():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')

    Qx = Predicate(Q, (x,))
    Rx = Predicate(R, (x,))
    Sx = Predicate(S, (x,))

    ttl = Translator()

    disj1 = Q(x) | R(x)
    res = ttl.walk(disj1)
    assert isinstance(res, Disjunction)
    assert res.formulas == (Qx, Rx)

    disj2 = disj1 | S(x)
    res = ttl.walk(disj2)
    assert isinstance(res, Disjunction)
    assert res.formulas == (Qx, Rx, Sx)

    disj3 = S(x) | disj1
    res = ttl.walk(disj3)
    assert isinstance(res, Disjunction)
    assert res.formulas == (Sx, Qx, Rx)

    conj1 = Implication(Q(x), Q(x) & R(x))
    conj2 = Implication(Q(x), R(x))
    eb = ExpressionBlock((conj1, conj2))
    res = ttl.walk(eb)
    assert isinstance(res, Disjunction)
    assert res == Disjunction(
        (
            Implication(Qx, Conjunction((Qx, Rx))),
            Implication(Qx, Rx)
        )
    )
