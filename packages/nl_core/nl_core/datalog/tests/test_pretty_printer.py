"""Tests for the Datalog pretty-printer."""

import operator

import pytest

from nl_expr.expressions import Constant, Symbol
from nl_core.datalog.expressions import Implication
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from nl_core.probabilistic.expressions import ProbabilisticFact

a, b, c = Symbol("a"), Symbol("b"), Symbol("c")


def _op(function):
    return lambda left, right: Constant(function)(left, right)


ADD, SUB, MUL, DIV, POW = map(
    _op, (operator.add, operator.sub, operator.mul, operator.truediv,
          operator.pow)
)


@pytest.mark.parametrize("expression,text", [
    (DIV(a, MUL(b, c)), "a ÷ (b × c)"),
    (MUL(DIV(a, b), c), "a ÷ b × c"),
    (SUB(a, SUB(b, c)), "a − (b − c)"),
    (MUL(ADD(a, b), c), "(a + b) × c"),
    (ADD(a, MUL(b, c)), "a + b × c"),
    (POW(POW(a, b), c), "(a ** b) ** c"),
    (MUL(a, POW(b, Constant(2))), "a × b ** 2"),
])
def test_operands_binding_less_tightly_are_in_parentheses(expression, text):
    assert DatalogPrettyPrinter().walk(expression) == text


def test_probabilistic_rule_prints_its_probability():
    p, q, v = Symbol("p"), Symbol("q"), Symbol("v")
    rule = Implication(ProbabilisticFact(v, p(a)), q(a, v))
    assert DatalogPrettyPrinter().walk(rule) == "p(a) :: v :-\n    q(a, v)"
