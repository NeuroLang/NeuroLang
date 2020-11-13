from pytest import raises

from .. import expression_walker
from .. import expressions

from typing import Callable, AbstractSet, Tuple

S_ = expressions.Symbol
C_ = expressions.Constant
F_ = expressions.FunctionApplication


def test_symbol_table_as_parameter():
    symbol_table = expressions.TypedSymbolTable()
    solver = expression_walker.TypedSymbolTableEvaluator(symbol_table)
    s = S_('S1')
    c = C_[str]('S')
    symbol_table[s] = c
    assert solver.symbol_table[s] is c


def test_expression_evaluator():
    ebe = expression_walker.ExpressionBasicEvaluator()
    a = C_(1)
    b = C_(2)

    c = ebe.walk(a + b)
    assert isinstance(c, C_[int])
    assert c.value == 3


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
    assert isinstance(w.symbol_table[f], expressions.Constant)
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
        expressions.NeuroLangException, match="Entry point not declared"
    ):

        class PM3(expression_walker.EntryPointPatternWalker):
            @expression_walker.add_match(F_)
            def __(self, expression):
                return expression

        PM3()


def test_chained_walker():
    class WalkerA(expression_walker.ExpressionWalker):
        @expression_walker.add_match(S_("A"))
        def match_a(self, expression):
            return S_("B")

    class WalkerB(expression_walker.ExpressionWalker):
        @expression_walker.add_match(S_("A"))
        def match_a(self, expression):
            return S_("Q")

        @expression_walker.add_match(S_("B"))
        def match_a(self, expression):
            return S_("C")

    walker = expression_walker.ChainedWalker(WalkerA, WalkerB)
    exp = S_("A")
    res = walker.walk(exp)
    assert res == S_("C")


def test_convert_to_lambda():
    add = C_(lambda x, y: x + y)
    func = add(S_('x'), C_(1))

    fa2pl = expression_walker.FunctionApplicationToPythonLambda()

    l0, args = fa2pl.walk(func)
    assert l0(x=3) == 4
    assert args == {'x'}

    func = add(S_('x'), S_('y'))
    l1, args = fa2pl.walk(func)
    assert l1(x=3, y=1) == 4
    assert args == {'x', 'y'}

    sub = C_(lambda x, y: x - y)
    func = sub(add(S_('x'), C_(1)), C_(2))
    l2, args = fa2pl.walk(func)
    assert l2(x=3) == 2
    assert args == {'x'}

    one = C_(lambda: 1)
    func = sub(add(S_('x'), one()), C_(2))
    l3, args = fa2pl.walk(func)
    assert l3(x=3) == 2
    assert args == {'x'}


def test_expression_walker_keep_fresh():
    s = S_.fresh()
    assert s.is_fresh
    walker = expression_walker.ExpressionWalker()
    result = walker.walk(s)
    assert result is s


def test_replace_expression_maintains_symbol_fresh():
    s = S_.fresh()
    assert s.is_fresh
    replacer = expression_walker.ReplaceExpressionWalker({})
    result = replacer.walk(s)
    assert result is s


def test_replace_expression_with_tuple():
    s1 = S_.fresh()
    s2 = S_.fresh()
    s3 = S_("s3")
    replacer = expression_walker.ReplaceExpressionWalker({s1: s3})
    result = replacer.walk((s2, (s1, s2)))
    assert isinstance(result, Tuple)
    assert result[0] is s2 and result[0].is_fresh
    assert isinstance(result[1], Tuple)
    assert result[1][0] == s3
    assert result[1][1] is s2 and result[1][1].is_fresh
