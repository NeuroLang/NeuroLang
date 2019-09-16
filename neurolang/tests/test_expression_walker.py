from pytest import raises

from .. import expression_walker
from .. import expressions
from .. import neurolang as nl

from typing import Callable, AbstractSet, Tuple

S_ = nl.Symbol
C_ = nl.Constant
F_ = nl.FunctionApplication


def test_symbol_table_as_parameter():
    symbol_table = expressions.TypedSymbolTable()
    solver = expression_walker.TypedSymbolTableEvaluator(symbol_table)
    s = S_('S1')
    c = C_[str]('S')
    symbol_table[s] = c
    assert solver.symbol_table[s] is c


def test_evaluating_embedded_functions():
    class Walker(
        expression_walker.ExpressionBasicEvaluator,
        expression_walker.TypedSymbolTableEvaluator
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


def test_symbol_table_scopes():
    symbol_table = expressions.TypedSymbolTable()
    solver = expression_walker.TypedSymbolTableEvaluator(symbol_table)
    s = S_('S1')
    c = C_[str]('S')
    symbol_table[s] = c
    assert solver.symbol_table[s] is c
    solver.push_scope()
    assert len(solver.symbol_table) == 0
    assert solver.symbol_table[s] == c
    solver.pop_scope()
    assert solver.symbol_table[s] is c
    with raises(expressions.NeuroLangException):
        solver.pop_scope()
        solver.pop_scope()


def test_entry_point_walker():
    class PM(expression_walker.EntryPointPatternWalker):
        @expression_walker.add_entry_point_match(F_(S_('start'), ...))
        def entry_point(self, expression):
            return expression.functor(*(
                self.walk(arg) for arg in expression.args
            ))

        @expression_walker.add_match(F_)
        def __(self, expression):
            return expression

    pm = PM()

    exp_correct = S_('start')(S_('a')())
    exp_wrong = S_('end')(S_('a')())

    res = pm.walk(exp_correct)
    assert res == exp_correct

    with raises(expressions.NeuroLangException):
        pm.walk(exp_wrong)

    class PM2(PM):
        @expression_walker.add_entry_point_match(
            F_,
            lambda x: x.functor == S_('pre_start')
        )
        def entry_point(self, expression):
            return expression.functor(*(
                self.walk(arg) for arg in expression.args
            ))

    pm2 = PM2()

    exp_pre_correct = S_('pre_start')(S_('a')())
    res = pm2.walk(exp_pre_correct)

    assert res == exp_pre_correct

    with raises(expressions.NeuroLangException):
        pm2.walk(exp_correct)

    with raises(
        expressions.NeuroLangException, message="Entry point not declared"
    ):
        class PM3(expression_walker.EntryPointPatternWalker):
            @expression_walker.add_match(F_)
            def __(self, expression):
                return expression

        PM3()
