import operator

from .. import lazy_code
from ..expression_walker import ExpressionBasicEvaluator, ExpressionWalker
from ..expressions import (
    Constant,
    ExpressionBlock,
    Statement,
    Symbol,
    TypedSymbolTableMixin
)


class LazyCode(
    lazy_code.LazyCodeEvaluationMixin,
    TypedSymbolTableMixin,
    ExpressionWalker
):
    pass


class LazyCodeBasicExpressionEvaluator(
    lazy_code.LazyCodeEvaluationMixin,
    TypedSymbolTableMixin,
    ExpressionBasicEvaluator
):
    pass


A = Symbol('A')
B = Symbol('B')
C = Symbol('C')
D = Symbol('D')
ADD = Constant(operator.add)


def test_statement():
    code = ExpressionBlock((Statement(A, Constant(1)), Statement(B, A)))

    lce = LazyCode()
    lce.walk(code)

    assert lce.symbol_table[A] == Constant(1)
    assert lce.symbol_table[B] == Symbol(A)

    lce.walk(lazy_code.Evaluate(B))
    assert lce.symbol_table[B] == Constant(1)


def test_evaluations():
    code = ExpressionBlock((
        Statement(A, Constant(1)), Statement(B, A),
        Statement(C, ADD(Constant(2), A)), Statement(D, C)
    ))

    lce = LazyCodeBasicExpressionEvaluator()
    lce.walk(code)

    assert lce.symbol_table[A] == Constant(1)
    assert lce.symbol_table[B] == Symbol(A)
    assert lce.symbol_table[C] == ADD(Constant(2), A)

    lce.walk(lazy_code.Evaluate(B))
    assert lce.symbol_table[B] == Constant(1)
    assert lce.symbol_table[C] == ADD(Constant(2), A)

    lce = LazyCodeBasicExpressionEvaluator()
    lce.walk(code)

    assert lce.walk(lazy_code.Evaluate(C)) == Constant(3)
    assert lce.symbol_table[A] == Constant(1)
    assert lce.symbol_table[B] == A
    assert lce.symbol_table[C] == Constant(3)
