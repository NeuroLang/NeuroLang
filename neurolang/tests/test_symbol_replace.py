from ..symbols_and_types import TypedSymbolTable
from ..region_solver import SetBasedSolver
from ..expressions import (
    Symbol, Constant, Predicate, ExistentialPredicate
)
from typing import Callable, AbstractSet
import operator as op
from ..expression_walker import ReplaceSymbolWalker, ExpressionBasicEvaluator


def test_replace_in_walker():
    value = Constant[int](2)
    symbol_to_replace = Symbol('x')

    rsw = ReplaceSymbolWalker(symbol_to_replace, value)
    result = rsw.walk(symbol_to_replace)

    assert result == value


def test_replace_variable_in_expression():
    symbol_to_replace = Symbol('a')
    value = Constant[int](2)

    add_constant = Constant[Callable[[int, int], int]](op.add)
    add_op = add_constant(symbol_to_replace, Constant[int](3))

    rsw = ReplaceSymbolWalker(symbol_to_replace, value)
    add_replacement = rsw.walk(add_op)

    ebe = ExpressionBasicEvaluator()
    add_result = ebe.walk(add_replacement)
    assert add_result == 5