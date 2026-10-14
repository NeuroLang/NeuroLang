"""Lifted inference for conjunctive queries with negated existential
subqueries, Q1(x, y) ∧ ¬∃t Q2(x, t), checked against the probabilities
computed by enumerating every possible world.
"""
import itertools

import pytest

from ...datalog.expression_processing import flatten_query
from ...exceptions import NonLiftableException
from ...expression_walker import expression_iterator
from ...expressions import Symbol
from ...logic import Conjunction, Implication, Negation, Union
from ...relational_algebra_provenance import Difference, Subtraction
from .. import dalvi_suciu_lift
from ..cplogic.program import CPLogicProgram

ans = Symbol("ans")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
D = Symbol("D")
M = Symbol("M")
N = Symbol("N")
Q = Symbol("Q")
R = Symbol("R")
S = Symbol("S")
T = Symbol("T")
W = Symbol("W")
t = Symbol("t")
u = Symbol("u")
v = Symbol("v")
x = Symbol("x")
y = Symbol("y")

R2 = {(0.5, "a", "b"), (0.4, "a", "c"), (0.3, "b", "b"), (0.7, "c", "a")}
S2 = {(0.6, "a", "b"), (0.2, "a", "c"), (0.9, "b", "a")}
R1 = {(0.3, "a"), (0.6, "b"), (0.9, "c")}
T1 = {(0.5, "a"), (0.25, "b"), (0.8, "c")}
Q1 = {(0.4, "a"), (0.7, "c")}


def possible_worlds_probabilities(
    probabilistic_facts, deterministic_facts, query
):
    """Probability of each answer of `query`, a function from a database,
    mapping symbols to sets of tuples, to a set of answers.
    """
    facts = [
        (symbol, row[1:], row[0])
        for symbol, rows in probabilistic_facts.items()
        for row in rows
    ]
    probabilities = {}
    for world in itertools.product((False, True), repeat=len(facts)):
        weight = 1.
        database = {
            symbol: set(rows) for symbol, rows in deterministic_facts.items()
        }
        for symbol in probabilistic_facts:
            database.setdefault(symbol, set())
        for present, (symbol, row, probability) in zip(world, facts):
            if present:
                weight *= probability
                database[symbol].add(row)
            else:
                weight *= 1 - probability
        for answer in query(database):
            probabilities[answer] = probabilities.get(answer, 0) + weight
    return probabilities


def solve(probabilistic_facts, deterministic_facts, rules, query):
    cpl = CPLogicProgram()
    for symbol, rows in probabilistic_facts.items():
        cpl.add_probabilistic_facts_from_tuples(symbol, rows)
    for symbol, rows in deterministic_facts.items():
        cpl.add_extensional_predicate_from_tuples(symbol, rows)
    if rules:
        cpl.walk(Union(tuple(rules)))
    result = dalvi_suciu_lift.solve_succ_query(query, cpl)
    columns = tuple(arg.name for arg in query.consequent.args)
    provenance_column = result.provenance_column.value
    return {
        tuple(row[column] for column in columns): row[provenance_column]
        for _, row in (
            result.relation.value.as_pandas_dataframe().iterrows()
        )
    }


def assert_probabilities_equal(result, expected):
    result = {k: p for k, p in result.items() if p > 1e-12}
    expected = {k: p for k, p in expected.items() if p > 1e-12}
    assert result.keys() == expected.keys()
    for answer, probability in expected.items():
        assert result[answer] == pytest.approx(probability)


def check(probabilistic_facts, deterministic_facts, rules, query, oracle):
    result = solve(probabilistic_facts, deterministic_facts, rules, query)
    expected = possible_worlds_probabilities(
        probabilistic_facts, deterministic_facts, oracle
    )
    assert_probabilities_equal(result, expected)


def exists(rows, condition):
    return any(condition(*row) for row in rows)


def test_independent_negation():
    """R(x, y) ∧ ¬∃t S(x, t): tuples of R whose x is not in S keep
    their probability."""
    check(
        {R: R2, S: S2}, {},
        [Implication(N(x), S(x, t))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x))))),
        lambda db: {
            (a, b) for a, b in db[R]
            if not exists(db[S], lambda c, _: c == a)
        },
    )


def test_independent_negations():
    check(
        {R: R2, S: S2, T: S2}, {},
        [Implication(N(x), S(x, t)), Implication(M(y), T(y, t))],
        Implication(ans(x, y), Conjunction((
            R(x, y), Negation(N(x)), Negation(M(y))
        ))),
        lambda db: {
            (a, b) for a, b in db[R]
            if not exists(db[S], lambda c, _: c == a)
            and not exists(db[T], lambda c, _: c == b)
        },
    )


def test_negation_sharing_a_deterministic_symbol():
    """D is deterministic, so the negated subquery is independent
    of the rest of the query."""
    check(
        {R: R2, S: S2}, {D: {("a",), ("b",)}},
        [Implication(N(x), Conjunction((S(x, t), D(t))))],
        Implication(
            ans(x, y), Conjunction((R(x, y), D(x), Negation(N(x))))
        ),
        lambda db: {
            (a, b) for a, b in db[R]
            if (a,) in db[D]
            and not exists(db[S], lambda c, d: c == a and (d,) in db[D])
        },
    )


def test_negation_sharing_a_probabilistic_symbol_unsatisfiable():
    """R(x, y) ∧ ¬∃t R(x, t) is unsatisfiable."""
    check(
        {R: R2}, {},
        [Implication(N(x), R(x, t))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x))))),
        lambda db: set(),
    )


def test_negation_sharing_a_probabilistic_symbol():
    """W(x) ∧ ∃y S(x, y) ∧ ¬∃t (S(x, t) ∧ T(t)), lifted as
    P[W(x) ∧ ∃y S(x, y)] - P[W(x) ∧ ∃t (S(x, t) ∧ T(t))]."""
    check(
        {W: R1, S: S2, T: T1}, {},
        [Implication(N(x), Conjunction((S(x, t), T(t))))],
        Implication(
            ans(x), Conjunction((W(x), S(x, y), Negation(N(x))))
        ),
        lambda db: {
            (a,) for (a,) in db[W]
            if exists(db[S], lambda c, _: c == a)
            and not exists(db[S], lambda c, d: c == a and (d,) in db[T])
        },
    )


def test_negation_sharing_a_probabilistic_symbol_inclusion_exclusion():
    """The positive query W(x) ∧ ∃y (S(x, y) ∧ Q(y)) ∧ ∃t (S(x, t) ∧ T(t))
    of the subtraction is lifted by inclusion-exclusion."""
    check(
        {W: R1, S: S2, T: T1, Q: Q1}, {},
        [Implication(N(x), Conjunction((S(x, t), T(t))))],
        Implication(
            ans(x), Conjunction((W(x), S(x, y), Q(y), Negation(N(x))))
        ),
        lambda db: {
            (a,) for (a,) in db[W]
            if exists(db[S], lambda c, d: c == a and (d,) in db[Q])
            and not exists(db[S], lambda c, d: c == a and (d,) in db[T])
        },
    )


def test_hard_negated_subquery_is_not_liftable():
    """W(x) ∧ ¬∃u, v (A(u) ∧ B(u, v) ∧ C(v)) is #P-hard, as the negated
    subquery is H0 and is independent of W(x)."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(W, T1)
    cpl.add_probabilistic_facts_from_tuples(A, T1)
    cpl.add_probabilistic_facts_from_tuples(B, S2)
    cpl.add_probabilistic_facts_from_tuples(C, {(0.3, "a"), (0.6, "b")})
    cpl.walk(Implication(N(), Conjunction((A(u), B(u, v), C(v)))))
    query = Implication(ans(x), Conjunction((W(x), Negation(N()))))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


def test_negation_sharing_a_symbol_through_a_free_variable_not_liftable():
    """R(x) ∧ ¬∃t (R(t) ∧ S(x, t)) is not lifted: R(x) and R(t) refer to
    the same tuple when t = x, which would require shattering on the
    free variable x."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R1)
    cpl.add_probabilistic_facts_from_tuples(S, S2)
    cpl.walk(Implication(N(x), Conjunction((R(t), S(x, t)))))
    query = Implication(ans(x), Conjunction((R(x), Negation(N(x)))))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


def test_positive_conjunct_implied_by_another():
    """R(x, y) ∧ ∃t R(x, t) is minimised to R(x, y)."""
    check(
        {R: R2, T: T1}, {},
        [Implication(N(x), R(x, t))],
        Implication(ans(x, y), Conjunction((R(x, y), T(y), N(x)))),
        lambda db: {(a, b) for a, b in db[R] if (b,) in db[T]},
    )


def test_positive_conjuncts_with_different_free_variables():
    """Inclusion-exclusion does not apply when the conjuncts have
    different free variables: R(x, y) ∧ ∃t (R(x, t) ∧ T(t)) used to
    produce probabilities larger than 1."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, {(0.7, "c", "a")})
    cpl.add_probabilistic_facts_from_tuples(T, {(0.5, "a")})
    cpl.walk(Implication(N(x), Conjunction((R(x, t), T(t)))))
    query = Implication(ans(x, y), Conjunction((R(x, y), N(x))))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


def test_plan_uses_difference_for_independent_negation():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R2)
    cpl.add_probabilistic_facts_from_tuples(S, S2)
    cpl.walk(Implication(N(x), S(x, t)))
    query = Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x)))))
    plan = _lifted_plan(query, cpl)
    assert _contains(plan, Difference)
    assert not _contains(plan, Subtraction)


def test_plan_uses_subtraction_for_dependent_negation():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(W, R1)
    cpl.add_probabilistic_facts_from_tuples(S, S2)
    cpl.add_probabilistic_facts_from_tuples(T, T1)
    cpl.walk(Implication(N(x), Conjunction((S(x, t), T(t)))))
    query = Implication(
        ans(x), Conjunction((W(x), S(x, y), Negation(N(x))))
    )
    plan = _lifted_plan(query, cpl)
    assert _contains(plan, Subtraction)


def _lifted_plan(query, cpl):
    flat_query = Implication(
        query.consequent, flatten_query(query.antecedent, cpl)
    )
    shattered_query, symbol_table = (
        dalvi_suciu_lift._prepare_and_optimise_query(flat_query, cpl)
    )
    plan = dalvi_suciu_lift.lift_with_negation_subtraction(
        shattered_query, symbol_table
    )
    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)
    return plan


def _contains(expression, operation):
    return any(
        isinstance(sub_expression, operation)
        for _, sub_expression in expression_iterator(expression)
    )
