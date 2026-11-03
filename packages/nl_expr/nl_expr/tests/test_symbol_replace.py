import operator as op
from typing import Callable

from .. import expressions
from ..expression_walker import (ExpressionBasicEvaluator, ReplaceSymbolWalker,
                                 TypedSymbolTableEvaluator)

C_ = expressions.Constant
S_ = expressions.Symbol


class Evaluator(
    TypedSymbolTableEvaluator,
    ExpressionBasicEvaluator
):
    pass


def test_replace_in_walker():
    value = C_[int](2)
    symbol_to_replace = S_('x')

    rsw = ReplaceSymbolWalker({symbol_to_replace: value})
    result = rsw.walk(symbol_to_replace)

    assert result == value


def test_replace_variable_in_expression():
    symbol_to_replace = S_('a')
    value = C_[int](2)

    add_constant = C_[Callable[[int, int], int]](op.add)
    add_op = add_constant(symbol_to_replace, C_[int](3))

    rsw = ReplaceSymbolWalker({symbol_to_replace: value})
    add_replacement = rsw.walk(add_op)

    ebe = Evaluator()
    add_result = ebe.walk(add_replacement)
    assert add_result == 5


def test_replace_variables_in_expression():
    symbols_to_replace = {
        S_('a'): C_[int](2),
        S_('b'): C_[int](3)
    }

    add_constant = C_[Callable[[int, int], int]](op.add)
    add_op = add_constant(*symbols_to_replace)

    rsw = ReplaceSymbolWalker(symbols_to_replace)
    add_replacement = rsw.walk(add_op)

    ebe = Evaluator()
    add_result = ebe.walk(add_replacement)
    assert add_result == 5
