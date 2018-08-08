from typing import Callable, AbstractSet
import operator as op

from ..symbols_and_types import TypedSymbolTable
from ..region_solver import SetBasedSolver
from .. import expressions
from ..expressions import Predicate, ExistentialPredicate
from ..expression_walker import ReplaceSymbolWalker, ExpressionBasicEvaluator

C_ = expressions.Constant
S_ = expressions.Symbol


def test_replace_in_walker():
    value = C_[int](2)
    symbol_to_replace = S_('x')

    rsw = ReplaceSymbolWalker(symbol_to_replace, value)
    result = rsw.walk(symbol_to_replace)

    assert result == value


def test_replace_variable_in_expression():
    symbol_to_replace = S_('a')
    value = C_[int](2)

    add_constant = C_[Callable[[int, int], int]](op.add)
    add_op = add_constant(symbol_to_replace, C_[int](3))

    rsw = ReplaceSymbolWalker(symbol_to_replace, value)
    add_replacement = rsw.walk(add_op)

    ebe = ExpressionBasicEvaluator()
    add_result = ebe.walk(add_replacement)
    assert add_result == 5
