from pytest import raises

from .. import expression_walker
from .. import symbols_and_types
from .. import neurolang as nl

from typing import Callable, AbstractSet, Tuple

S_ = nl.Symbol
C_ = nl.Constant
F_ = nl.FunctionApplication


def test_symbol_table_as_parameter():
    symbol_table = symbols_and_types.TypedSymbolTable()
    solver = expression_walker.SymbolTableEvaluator(symbol_table)
    s = S_('S1')
    c = C_[str]('S')
    symbol_table[s] = c
    assert solver.symbol_table[s] is c


def test_evaluating_embedded_functions():
    class Walker(
        expression_walker.ExpressionBasicEvaluator,
        expression_walker.SymbolTableEvaluator
    ):
        def function_add_set(self, a: AbstractSet[int]) -> int:
            return sum(a)

        def function_add_tuple(self, a: Tuple[int, int]) -> int:
            return sum(a)

    a = C_[AbstractSet[int]](frozenset(
        C_[int](i) for i in range(3)
    ))

    w = Walker()
    f = S_[Callable[[AbstractSet[int]], int]]('add_set')

    assert f in w.symbol_table
    assert isinstance(w.symbol_table[f], nl.Constant)
    assert w.symbol_table[f].value == w.function_add_set

    res = w.walk(F_[int](f, (a,)))
    assert res.value == 3

    f = S_[Callable[[AbstractSet[int]], int]]('add_tuple')
    a = C_[Tuple[int, int, int]](tuple(
        C_[int](i) for i in range(3)
    ))

    res = w.walk(F_[int](f, (a,)))
    assert res.value == 3


def test_pattern_walker_wrong_args():
    with raises(TypeError):
        class PM(expression_walker.PatternWalker):
            @expression_walker.add_match(
                F_(...)
            )
            def __(self, expression):
                return expression
