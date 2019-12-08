from ...expression_walker import IdentityWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from ..expressions import (Union, Fact, Implication,
                           TranslateToLogic)


class Translator(TranslateToLogic, IdentityWalker):
    pass


def test_translation_to_facts():
    Q = Symbol('Q')
    x = Symbol('x')

    ttl = Translator()

    imp = Implication(Q(x), Constant(True))
    res = ttl.walk(imp)
    assert isinstance(res, Fact)
    assert res.consequent == imp.consequent


def test_translation_to_union():
    Q = Symbol('Q')
    R = Symbol('R')
    x = Symbol('x')

    ttl = Translator()

    imp = ExpressionBlock((
        Implication(Q(x), R(x, x)),
        Implication(Q(x), Q(x))
    ))
    res = ttl.walk(imp)
    assert isinstance(res, Union)
    assert res.formulas == imp.expressions
