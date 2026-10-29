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
from ..pretty_printer import (
    DatalogPrettyPrinter,
    LogicPrettyPrinter,
    fold_probabilities,
)

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
        "     ∧ ¬S(y)\n"
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
    assert logic(conditional) == "Q(x, P[R(x, y) | S(y)])"
    marginal = Implication(
        Q(x, PROB(x), y), Conjunction((R(x, y), S(y), T(x)))
    )
    assert logic(marginal) == "Q(x, P[R(x, y) ∧ S(y) ∧ T(x)], y)"
    assert logic(Implication(Q(x, PROB(x)), R(x))) == "Q(x, P[R(x)])"


LONG = Symbol("a_rather_long_predicate_name")


def test_long_probabilities_wrap_at_the_width():
    conditioned = Conjunction((LONG(x, y), S(y), LONG(y, x)))
    conditional = Implication(
        Q(x, PROB(x)), Condition(conditioned, Conjunction((S(y), T(y))))
    )
    assert logic(conditional) == (
        "Q(x, P[a_rather_long_predicate_name(x, y) ∧ S(y)\n"
        "       ∧ a_rather_long_predicate_name(y, x)\n"
        "      | S(y) ∧ T(y)])"
    )
    assert all(
        len(line) <= LogicPrettyPrinter.width
        for line in logic(conditional).split("\n")
    )


def test_disjunctions_break_when_long_or_of_conjunctions():
    rule = Implication(Q(x), Conjunction((
        R(x, y), Disjunction((S(x), Conjunction((T(x), R(x, x))))),
        Negation(Conjunction((S(y), T(y)))),
        Disjunction((LONG(x, y), LONG(y, x), LONG(x, x))),
    )))
    assert logic(rule) == (
        "Q(x) ← R(x, y)\n"
        "     ∧ ( S(x)\n"
        "       ∨ T(x) ∧ R(x, x) )\n"
        "     ∧ ¬(S(y) ∧ T(y))\n"
        "     ∧ ( a_rather_long_predicate_name(x, y)\n"
        "       ∨ a_rather_long_predicate_name(y, x)\n"
        "       ∨ a_rather_long_predicate_name(x, x) )"
    )


def _probability_query():
    """Return ``ans(t, p) :- PROB[R(t, s) // S(s)] = p, (p > 0.3)``."""
    t, s, p = Symbol("t"), Symbol("s"), Symbol("p")
    helper = Symbol.fresh()
    return [
        Implication(helper(t, PROB(t)), Condition(R(t, s), S(s))),
        Query(
            Symbol("ans")(t, p),
            Conjunction((
                Constant(operator.gt)(p, Constant(0.3)), helper(t, p)
            )),
        ),
    ]


def test_helper_rules_of_probabilities_are_folded():
    formulas = fold_probabilities(_probability_query())
    assert [logic(f) for f in formulas] == [
        "ans(t, p) ← p = P[R(t, s) | S(s)]\n"
        "          ∧ p > 0.3"
    ]
    assert [DatalogPrettyPrinter().walk(f) for f in formulas] == [
        "ans(t, p) :-\n    PROB[R(t, s) | S(s)] = p,\n    p > 0.3"
    ]


def test_rules_of_user_predicates_are_not_folded():
    t, s, p = Symbol("t"), Symbol("s"), Symbol("p")
    rules = [
        Implication(Q(t, PROB(t)), Condition(R(t, s), S(s))),
        Query(Symbol("ans")(t, p), Q(t, p)),
    ]
    assert fold_probabilities(rules) == rules
    helper, query = _probability_query()
    used = query.body.formulas[1]
    other = used.functor(used.args[0], Symbol("q"))
    twice = Query(query.head, Conjunction((used, other)))
    assert fold_probabilities([helper, twice]) == [helper, twice]


def test_the_datalog_notation_is_unchanged():
    rule = Implication(Q(x), Conjunction((R(x, y), S(y))))
    assert DatalogPrettyPrinter().walk(rule) == (
        "Q(x) :-\n    R(x, y),\n    S(y)"
    )
    negations = Conjunction((Negation(S(y)), Negation(NE(y, Constant(3)))))
    assert DatalogPrettyPrinter().walk(negations) == "¬S(y) ∧ ¬(y ≠ 3)"


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
