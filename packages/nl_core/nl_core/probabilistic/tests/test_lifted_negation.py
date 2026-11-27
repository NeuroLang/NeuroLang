"""Lifted inference for conjunctive queries with existential and negated
existential subqueries, Q1(x, y) ∧ ¬∃t Q2(x, t), sharing symbols with the
rest of the query or with comparisons, checked against the probabilities
computed by enumerating every possible world.
"""
import itertools
import operator

import pytest

from ...datalog.expression_processing import flatten_query
from nl_expr.exceptions import NonLiftableException, UnsupportedSolverError
from nl_expr.expression_walker import expression_iterator
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import Conjunction, Implication, Negation, Union
from ...relational_algebra import Selection
from ...relational_algebra_provenance import Difference, Subtraction
from .. import dalvi_suciu_lift
from ..cplogic.program import CPLogicProgram

ans = Symbol("ans")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
D = Symbol("D")
E = Symbol("E")
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
w = Symbol("w")
y = Symbol("y")
z = Symbol("z")

NE = Constant(operator.ne)
LT = Constant(operator.lt)
LE = Constant(operator.le)

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
    return answer_probabilities(
        dalvi_suciu_lift.solve_succ_query(query, cpl), query
    )


def answer_probabilities(result, query):
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


SL = {
    (0.6, "a", "b"), (0.5, "a", "a"), (0.9, "b", "a"),
    (0.4, "c", "c"), (0.3, "c", "a"),
}


def test_negation_sharing_a_symbol_through_a_free_variable():
    """R(x) ∧ ¬∃t (R(t) ∧ S(x, t)): R(x) and R(t) are the same tuple
    when t = x, it is lifted as R(x) ∧ ¬S(x, x) ∧ ¬∃t≠x (R(t) ∧ S(x, t)).
    """
    check(
        {R: R1, S: SL}, {},
        [Implication(N(x), Conjunction((R(t), S(x, t))))],
        Implication(ans(x), Conjunction((R(x), Negation(N(x))))),
        lambda db: {
            (a,) for (a,) in db[R]
            if not exists(db[S], lambda c, d: c == a and (d,) in db[R])
        },
    )


def test_negation_sharing_symbols_through_a_free_variable():
    """R(x) ∧ T(x) ∧ ¬∃t (R(t) ∧ E(t, t) ∧ S(t, x) ∧ T(t))."""
    check(
        {R: R1, S: SL, E: SL, T: T1}, {},
        [Implication(N(x), Conjunction((R(t), E(t, t), S(t, x), T(t))))],
        Implication(ans(x), Conjunction((R(x), T(x), Negation(N(x))))),
        lambda db: {
            (a,) for (a,) in db[R]
            if (a,) in db[T] and not exists(
                db[S],
                lambda c, d: d == a and (c,) in db[R] and (c,) in db[T]
                and (c, c) in db[E]
            )
        },
    )


def test_negation_sharing_a_symbol_through_free_variables():
    """R(x, y) ∧ ¬∃t (R(t, y) ∧ S(x, t))."""
    check(
        {R: R2, S: SL}, {},
        [Implication(N(x, y), Conjunction((R(t, y), S(x, t))))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x, y))))),
        lambda db: {
            (a, b) for (a, b) in db[R]
            if not exists(db[S], lambda c, d: c == a and (d, b) in db[R])
        },
    )


def test_negation_sharing_a_choice_relation():
    """R(x) ∧ ¬∃t (R(t) ∧ S(x, t)) with R a probabilistic choice: where
    R(x) holds, R(t) holds only for t = x, and the query is
    R(x) ∧ ¬S(x, x)."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(
        R, {(0.3, "a"), (0.6, "b"), (0.1, "c")}
    )
    cpl.add_probabilistic_facts_from_tuples(S, SL)
    cpl.walk(Implication(N(x), Conjunction((R(t), S(x, t)))))
    query = Implication(ans(x), Conjunction((R(x), Negation(N(x)))))
    assert_probabilities_equal(
        answer_probabilities(
            dalvi_suciu_lift.solve_succ_query(query, cpl), query
        ),
        {
            ("a",): 0.3 * (1 - 0.5),
            ("b",): 0.6,
            ("c",): 0.1 * (1 - 0.4),
        }
    )


def test_negation_sharing_a_symbol_through_an_existential_not_liftable():
    """R(x) ∧ S(x, y) ∧ ¬∃t (R(t) ∧ S(x, t)) is not lifted: S(x, y) and
    S(x, t) are the same tuple when t = y, an existential variable."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R1)
    cpl.add_probabilistic_facts_from_tuples(S, SL)
    cpl.walk(Implication(N(x), Conjunction((R(t), S(x, t)))))
    query = Implication(
        ans(x), Conjunction((R(x), S(x, y), Negation(N(x))))
    )
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


R2L = {
    (0.5, "a", "b"), (0.4, "a", "a"), (0.3, "b", "b"),
    (0.7, "c", "a"), (0.2, "b", "c"),
}


def test_negation_sharing_a_symbol_through_a_free_variable_of_the_rest():
    """R(x, y) ∧ ¬∃t (R(x, t) ∧ T(t)): t takes the place of y, which is
    not a free variable of the negated subquery."""
    check(
        {R: R2L, T: T1}, {},
        [Implication(N(x), Conjunction((R(x, t), T(t))))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x))))),
        lambda db: {
            (a, b) for a, b in db[R]
            if not exists(db[R], lambda c, d: c == a and (d,) in db[T])
        },
    )


def test_boolean_negation_sharing_a_symbol_through_a_free_variable():
    """R(x) ∧ ¬∃t, u (R(t) ∧ S(t, u))."""
    check(
        {R: R1, S: SL}, {},
        [Implication(N(), Conjunction((R(t), S(t, u))))],
        Implication(ans(x), Conjunction((R(x), Negation(N())))),
        lambda db: {
            (a,) for (a,) in db[R]
            if not exists(db[S], lambda c, _: (c,) in db[R])
        },
    )


def test_positive_conjuncts_sharing_a_symbol_through_a_free_variable():
    """R(x, y) ∧ ∃t (R(x, t) ∧ T(t)), lifted as
    P[R(x, y)] - P[R(x, y) ∧ ¬∃t (R(x, t) ∧ T(t))]. Inclusion-exclusion
    does not apply, as the conjuncts have different free variables:
    it produced probabilities larger than 1."""
    check(
        {R: R2L, T: T1}, {},
        [Implication(N(x), Conjunction((R(x, t), T(t))))],
        Implication(ans(x, y), Conjunction((R(x, y), N(x)))),
        lambda db: {
            (a, b) for a, b in db[R]
            if exists(db[R], lambda c, d: c == a and (d,) in db[T])
        },
    )


def test_boolean_positive_conjunct_sharing_a_symbol():
    """R(x) ∧ ∃t, u (R(t) ∧ S(t, u))."""
    check(
        {R: R1, S: SL}, {},
        [Implication(N(), Conjunction((R(t), S(t, u))))],
        Implication(ans(x), Conjunction((R(x), N()))),
        lambda db: {
            (a,) for (a,) in db[R]
            if exists(db[S], lambda c, _: (c,) in db[R])
        },
    )


def test_positive_conjunct_without_separator_not_liftable():
    """R(x) ∧ ∃t, u (R(t) ∧ S(t, u) ∧ R(u)) is not lifted: t is not a
    separator variable of the existential subquery."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R1)
    cpl.add_probabilistic_facts_from_tuples(S, SL)
    cpl.walk(Implication(N(), Conjunction((R(t), S(t, u), R(u)))))
    query = Implication(ans(x), Conjunction((R(x), N())))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


PW = {
    (0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d"),
    (0.4, "c", "b"), (0.3, "b", "b"),
}


def test_negated_inequality():
    """Segregation: P(x, y) ∧ ¬∃w (P(w, y) ∧ w ≠ x)."""
    check(
        {R: PW}, {},
        [Implication(A(x, y), Conjunction((R(w, y), NE(w, x))))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(A(x, y))))),
        lambda db: {
            (a, b) for a, b in db[R]
            if not exists(db[R], lambda c, d: d == b and c != a)
        },
    )


def test_inequality():
    """P(x, y) ∧ ∃w (P(w, y) ∧ w ≠ x)."""
    check(
        {R: PW}, {},
        [Implication(A(x, y), Conjunction((R(w, y), NE(w, x))))],
        Implication(ans(x, y), Conjunction((R(x, y), A(x, y)))),
        lambda db: {
            (a, b) for a, b in db[R]
            if exists(db[R], lambda c, d: d == b and c != a)
        },
    )


def test_negated_order_comparison():
    """P(x, y) ∧ ¬∃w (P(w, y) ∧ T(w) ∧ w < x)."""
    check(
        {R: PW, T: T1}, {},
        [Implication(A(x, y), Conjunction((R(w, y), T(w), LT(w, x))))],
        Implication(ans(x, y), Conjunction((R(x, y), Negation(A(x, y))))),
        lambda db: {
            (a, b) for a, b in db[R]
            if not exists(
                db[R], lambda c, d: d == b and c < a and (c,) in db[T]
            )
        },
    )


def test_inequalities_with_constant():
    """P(x, y) ∧ ∃w (T(w) ∧ w ≠ "a" ∧ w ≠ x)."""
    check(
        {R: PW, T: T1}, {},
        [Implication(
            A(x), Conjunction((T(w), NE(w, Constant("a")), NE(w, x)))
        )],
        Implication(ans(x, y), Conjunction((R(x, y), A(x)))),
        lambda db: {
            (a, b) for a, b in db[R]
            if exists(db[T], lambda c: c not in ("a", a))
        },
    )


def test_non_strict_comparison_sharing_a_symbol_not_liftable():
    """P(x, y) ∧ ¬∃w (P(w, y) ∧ w ≤ x) is not lifted: w ≤ x does not make
    P(w, y) and P(x, y) different tuples."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, PW)
    cpl.walk(Implication(A(x, y), Conjunction((R(w, y), LE(w, x)))))
    query = Implication(
        ans(x, y), Conjunction((R(x, y), Negation(A(x, y))))
    )
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


R4 = R1 | {(0.5, "d")}
S3 = {
    (0.5, "a", "b"), (0.4, "a", "c"), (0.3, "a", "d"), (0.7, "b", "a"),
    (0.6, "b", "b"), (0.2, "b", "c"), (0.9, "c", "a"),
}


def test_at_least_three():
    """∃x, y, z (R(x) ∧ R(y) ∧ R(z) ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z), also with
    comparisons x < y < z, x ≠ z: R has at least three tuples."""
    for comparisons in (
        (NE(x, y), NE(y, z), NE(x, z)), (LT(x, y), LT(y, z), NE(x, z))
    ):
        check(
            {R: R4}, {}, [],
            Implication(ans(), Conjunction((R(x), R(y), R(z)) + comparisons)),
            lambda db: {()} if len(db[R]) >= 3 else set(),
        )


def test_at_least_three_per_answer():
    """∃x, y, z (S(t, x) ∧ S(t, y) ∧ S(t, z) ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z)."""
    check(
        {S: S3}, {}, [],
        Implication(ans(t), Conjunction((
            S(t, x), S(t, y), S(t, z), NE(x, y), NE(y, z), NE(x, z)
        ))),
        lambda db: {
            (a,) for a, _ in db[S]
            if sum(1 for c, _ in db[S] if c == a) >= 3
        },
    )


def test_at_least_four():
    """Four values of R, with pairwise ≠."""
    variables = (x, y, z, u)
    check(
        {R: R4}, {}, [],
        Implication(ans(), Conjunction(
            tuple(R(v) for v in variables) +
            tuple(NE(*p) for p in itertools.combinations(variables, 2))
        )),
        lambda db: {()} if len(db[R]) >= 4 else set(),
    )


@pytest.mark.parametrize("variables, comparisons", [
    ((x, y, z), (NE(x, y), NE(y, z))),
    ((x, y, z), (LT(x, y), LT(y, z))),
    ((x, y, z), (LT(x, y), LT(y, z), LT(z, x))),
])
def test_at_least_k_not_lifted(variables, comparisons):
    """Without a strict comparison of each pair, satisfied by an order
    of the variables, the query is not at least k values of R."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R4)
    query = Implication(
        ans(), Conjunction(tuple(R(v) for v in variables) + comparisons)
    )
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


R7 = {(0.9, 1), (0.8, 2), (0.7, 3), (0.6, 4), (0.5, 5), (0.4, 6), (0.3, 7)}
S7 = {(0.9, "a", 1), (0.8, "a", 2), (0.7, "a", 3), (0.6, "a", 4),
      (0.5, "a", 5), (0.4, "b", 1), (0.3, "b", 2), (0.5, "b", 3)}


def pairwise_distinct(variables):
    return tuple(NE(*p) for p in itertools.combinations(variables, 2))


@pytest.mark.parametrize("k", [5, 6, 7, 8])
def test_at_least_more_than_four_values(k):
    """The Poisson-binomial tail, for k beyond the former limit of 4, and
    for k greater than the number of tuples, where it is 0."""
    variables = tuple(Symbol(f"x{i}") for i in range(k))
    check(
        {R: R7}, {}, [],
        Implication(ans(), Conjunction(
            tuple(R(v) for v in variables) + pairwise_distinct(variables)
        )),
        lambda db: {()} if len(db[R]) >= k else set(),
    )


@pytest.mark.parametrize("k", [2, 3, 5])
def test_at_least_k_per_answer_with_a_deterministic_atom(k):
    """Every copy has a deterministic atom, the answer variable is free."""
    variables = tuple(Symbol(f"x{i}") for i in range(k))
    even = {(a, c) for _, a, c in S7 if c % 2 == 1 or a == "b"}
    check(
        {S: S7}, {D: even}, [],
        Implication(ans(t), Conjunction(
            tuple(S(t, v) for v in variables) +
            tuple(D(t, v) for v in variables) + pairwise_distinct(variables)
        )),
        lambda db: {
            (a,) for a, _ in db[S]
            if len({c for b, c in db[S] if b == a and (b, c) in db[D]}) >= k
        },
    )


def test_at_least_k_with_an_independent_conjunct():
    """P[T(t) ∧ at least three values] = P[T(t)] P[at least three]."""
    variables = (x, y, z)
    check(
        {S: S7, T: {(0.5, "a"), (0.25, "b")}}, {}, [],
        Implication(ans(t), Conjunction(
            (T(t),) + tuple(S(t, v) for v in variables) +
            pairwise_distinct(variables)
        )),
        lambda db: {
            (a,) for a, in db[T]
            if len({c for b, c in db[S] if b == a}) >= 3
        },
    )


def test_at_least_k_with_comparisons_in_the_copies():
    variables = (x, y, z)
    check(
        {R: R7}, {}, [],
        Implication(ans(), Conjunction(
            tuple(R(v) for v in variables) +
            tuple(Constant(operator.gt)(v, Constant(2)) for v in variables) +
            (LT(x, y), LT(y, z), NE(x, z))
        )),
        lambda db: {()} if len({c for c, in db[R] if c > 2}) >= 3 else set(),
    )


def test_at_least_k_of_dependent_events_not_lifted():
    """The events of two values of x share T(y): the tail of the
    Poisson-binomial distribution would be wrong."""
    variables, ys = (x, y), (u, v)
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R1)
    cpl.add_probabilistic_facts_from_tuples(S, {(0.5, "a", "a")})
    cpl.add_probabilistic_facts_from_tuples(T, T1)
    query = Implication(ans(), Conjunction(
        tuple(R(a) for a in variables) +
        tuple(S(a, b) for a, b in zip(variables, ys)) +
        tuple(T(b) for b in ys) + (NE(x, y),)
    ))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


EVIDENCE = {
    (0.5, "brain1", "G", "D"), (0.4, "brain2", "G", "D"),
    (0.3, "blood1", "G", "D"), (0.6, "blood2", "G", "D"),
    (0.7, "other1", "G", "D"), (0.2, "blood1", "H", "D"),
    (0.9, "brain1", "G", "E"),
}
ORGANS = {
    ("brain1", "brain"), ("brain2", "brain"), ("blood1", "blood"),
    ("blood2", "blood"), ("other1", "liver"),
}


def _evidence_in_an_organ_or_at_least(k, organ, organs=ORGANS):
    """q() :- E(d, G, D), O(d, organ) and q() :- at least k values of d
    with E(d, G, D), against the possible worlds."""
    G, D = Constant("G"), Constant("D")
    ds = Symbol.fresh(), Symbol.fresh(), Symbol.fresh()
    filtered = E(x, G, D) if organ is None else Conjunction(
        (E(x, G, D), M(x, Constant(organ)))
    )
    check(
        {E: EVIDENCE}, {M: organs},
        [
            Implication(A(), filtered),
            Implication(B(), Conjunction(
                tuple(E(d, G, D) for d in ds[:k]) +
                tuple(NE(*p) for p in itertools.combinations(ds[:k], 2))
            )),
            Implication(Q(), A()),
            Implication(Q(), B()),
        ],
        Implication(ans(), Q()),
        lambda db: {()} if (
            any((d, organ) in db[M] or organ is None
                for d, g, s in db[E] if (g, s) == ("G", "D")) or
            sum((g, s) == ("G", "D") for _, g, s in db[E]) >= k
        ) else set(),
    )


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("organ", ["brain", "blood", "kidney", None])
def test_evidence_in_an_organ_or_at_least_k(k, organ):
    """In a brain dataset, or in at least k datasets: A ∨ Q_k ≡ A ∨
    Q_k on the datasets outside the brain, an organ of no dataset, or
    without the organ, Q_k implying A."""
    _evidence_in_an_organ_or_at_least(k, organ)


def test_evidence_in_an_organ_of_every_dataset_or_at_least_two():
    """The organ of every dataset: Q_k on no tuple."""
    _evidence_in_an_organ_or_at_least(
        2, "brain", {(d, "brain") for d, _ in ORGANS}
    )


def test_evidence_in_a_validated_blood_dataset_or_at_least_two():
    """A filter of two atoms, M(x, blood) ∧ W(x)."""
    G, D = Constant("G"), Constant("D")
    validated = {("blood1",), ("brain1",)}
    check(
        {E: EVIDENCE}, {M: ORGANS, W: validated},
        [
            Implication(A(), Conjunction(
                (E(x, G, D), M(x, Constant("blood")), W(x))
            )),
            Implication(B(), Conjunction((E(y, G, D), E(z, G, D), NE(y, z)))),
            Implication(Q(), A()),
            Implication(Q(), B()),
        ],
        Implication(ans(), Q()),
        lambda db: {()} if (
            any((d, "blood") in db[M] and (d,) in db[W]
                for d, g, s in db[E] if (g, s) == ("G", "D")) or
            sum((g, s) == ("G", "D") for _, g, s in db[E]) >= 2
        ) else set(),
    )


def test_evidence_in_the_brain_or_at_least_two_per_gene():
    """The same per gene g, with y < z."""
    g, D = Symbol("g"), Constant("D")
    check(
        {E: EVIDENCE}, {M: ORGANS},
        [
            Implication(A(g), Conjunction(
                (E(x, g, D), M(x, Constant("brain")))
            )),
            Implication(B(g), Conjunction((E(y, g, D), E(z, g, D), LT(y, z)))),
            Implication(Q(g), A(g)),
            Implication(Q(g), B(g)),
        ],
        Implication(ans(g), Q(g)),
        lambda db: {
            (a,) for _, a, s in db[E] if s == "D" and (
                any((d, "brain") in db[M]
                    for d, b, t in db[E] if (b, t) == (a, "D")) or
                sum((b, t) == (a, "D") for _, b, t in db[E]) >= 2
            )
        },
    )


def test_at_least_with_another_atom_of_the_relation_not_rewritten():
    """A third disjunct on E: the tuples outside the organ are not
    independent of it, the disjunction is not rewritten."""
    G, D = Constant("G"), Constant("D")
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(E, EVIDENCE)
    cpl.add_extensional_predicate_from_tuples(M, ORGANS)
    cpl.walk(Union((
        Implication(Q(), Conjunction((E(x, G, D), M(x, Constant("brain"))))),
        Implication(Q(), Conjunction((E(y, G, D), E(z, G, D), NE(y, z)))),
        Implication(Q(), E(Constant("blood1"), G, D)),
    )))
    with pytest.raises((NonLiftableException, UnsupportedSolverError)):
        dalvi_suciu_lift.solve_succ_query(Implication(ans(), Q()), cpl)


def test_plan_uses_difference_for_independent_negation():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R2)
    cpl.add_probabilistic_facts_from_tuples(S, S2)
    cpl.walk(Implication(N(x), S(x, t)))
    query = Implication(ans(x, y), Conjunction((R(x, y), Negation(N(x)))))
    plan = _lifted_plan(query, cpl)
    assert _contains(plan, Difference)
    assert not _contains(plan, Subtraction)


def test_plan_uses_free_variable_shattering():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(R, R1)
    cpl.add_probabilistic_facts_from_tuples(S, SL)
    cpl.walk(Implication(N(x), Conjunction((R(t), S(x, t)))))
    query = Implication(ans(x), Conjunction((R(x), Negation(N(x)))))
    plan = _lifted_plan(query, cpl)
    assert _contains(plan, Selection)
    assert _contains(plan, Difference)


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
    plan = dalvi_suciu_lift.lift_with_dependent_negations(
        shattered_query, symbol_table
    )
    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)
    return plan


def _contains(expression, operation):
    return any(
        isinstance(sub_expression, operation)
        for _, sub_expression in expression_iterator(expression)
    )
