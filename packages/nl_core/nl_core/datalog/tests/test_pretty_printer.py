"""Tests of the printing of rules in logic notation."""

import operator

from nl_expr.expressions import Constant, Query, Symbol
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Negation,
)

from ...probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticChoice,
    ProbabilisticFact,
)
from ..expressions import Implication
from ..pretty_printer import DatalogPrettyPrinter, LogicPrettyPrinter

P, Q, R, S, T = map(Symbol, "PQRST")
x, y = Symbol("x"), Symbol("y")
NE = Constant(operator.ne)
LE = Constant(operator.le)


def logic(expression):
    return LogicPrettyPrinter().walk(expression)


def test_conjuncts_are_aligned_under_the_first():
    rule = Implication(
        Symbol("close")(x, y),
        Conjunction((Symbol("similar_study")(x, y), Symbol("selected")(y))),
    )
    assert logic(rule) == (
        "close(x, y) ← similar_study(x, y)\n"
        "            ∧ selected(y)"
    )


def test_negation_comparison_and_disjunction():
    rule = Implication(Q(x), Conjunction((
        R(x, y), Negation(S(y)), NE(y, Constant(3)), LE(y, Constant(9)),
        Disjunction((S(x), T(x))),
    )))
    assert logic(rule) == (
        "Q(x) ← R(x, y)\n"
        "     ∧ ¬(S(y))\n"
        "     ∧ y ≠ 3\n"
        "     ∧ y ≤ 9\n"
        "     ∧ (S(x) ∨ T(x))"
    )


def test_aggregation_existential_and_query():
    count = Symbol("count")
    assert logic(Implication(Q(x, count(y)), R(x, y))) == (
        "Q(x, count(y)) ← R(x, y)"
    )
    query = Query(x, ExistentialPredicate(y, R(x, y)))
    assert logic(query) == "x ← ∃y. (R(x, y))"


def test_probabilities_read_as_p_of_the_body():
    conditional = Implication(
        Q(x, PROB(x)), Condition(R(x, y), Conjunction((S(y),)))
    )
    assert logic(conditional) == (
        "Q(x, P[R(x, y)\n"
        "      | S(y)])"
    )
    marginal = Implication(
        Q(x, PROB(x), y), Conjunction((R(x, y), S(y), T(x)))
    )
    assert logic(marginal) == (
        "Q(x, P[R(x, y)\n"
        "      ∧ S(y)\n"
        "      ∧ T(x)], y)"
    )
    assert logic(Implication(Q(x, PROB(x)), R(x))) == "Q(x, P[R(x)])"


def test_the_datalog_notation_is_unchanged():
    rule = Implication(Q(x), Conjunction((R(x, y), S(y))))
    assert DatalogPrettyPrinter().walk(rule) == (
        "Q(x) :-\n    R(x, y),\n    S(y)"
    )


def test_probabilistic_facts_and_choices():
    fact = Implication(ProbabilisticFact(Constant(0.3), P(x)), Q(x))
    assert logic(fact) == "P(x) :: 0.3 ← Q(x)"
    m = Symbol("m")
    choice = Implication(
        ProbabilisticChoice(Constant(operator.truediv)(Constant(1), m), P(x)),
        Conjunction((Q(x), R(m))),
    )
    assert DatalogPrettyPrinter().walk(choice) == (
        "P(x) ^ (1 ÷ m) :-\n    Q(x),\n    R(m)"
    )
