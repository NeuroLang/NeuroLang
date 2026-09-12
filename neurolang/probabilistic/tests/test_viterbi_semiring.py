"""Tests for the max-product semiring and Viterbi-style queries."""

from typing import AbstractSet

import pytest

from ...datalog import Fact
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ...relational_algebra import NamedRelationalAlgebraFrozenSet
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..semiring import MaxProductSemiring, ProbabilitySemiring

ans = Symbol("ans")
P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
S = Symbol("S")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

c1 = Constant(1)
c2 = Constant(2)


# ---------------------------------------------------------------------------
# Semiring sanity checks
# ---------------------------------------------------------------------------


def test_probability_semiring_regression():
    """The ProbabilitySemiring must preserve existing behaviour exactly."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, {(0.7, 1, 2), (0.9, 1, 3), (0.88, 2, 4)}
    )
    query = Implication(ans(x), P(x, y))
    result = dalvi_suciu_lift.solve_succ_query(
        query, cpl, semiring=ProbabilitySemiring()
    )
    expected = testing.make_prov_set(
        [
            ((1 - (1 - 0.7) * (1 - 0.9)), 1),
            (0.88, 2),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_probability_semiring_default():
    """Default (semiring=None) must behave identically to ProbabilitySemiring."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, {(0.7, 1, 2), (0.9, 1, 3), (0.88, 2, 4)}
    )
    query = Implication(ans(x), P(x, y))

    with_semiring = dalvi_suciu_lift.solve_succ_query(
        query, cpl, semiring=ProbabilitySemiring()
    )
    default = dalvi_suciu_lift.solve_succ_query(query, cpl)
    assert testing.eq_prov_relations(with_semiring, default)


# ---------------------------------------------------------------------------
# Max-product semiring: existential quantification
# ---------------------------------------------------------------------------


def test_maxproduct_existential():
    """MaxProductSemiring on an existential query yields max, not 1-∏(1-p).

    Query:  ans(x) :- P(x, y)
    Meaning: find the max-probability y for each x (MAP over existential).
    """
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, {(0.3, 1, "a"), (0.7, 1, "b"), (0.5, 1, "c"), (0.88, 2, "d")}
    )
    query = Implication(ans(x), P(x, y))
    result = dalvi_suciu_lift.solve_succ_query(
        query, cpl, semiring=MaxProductSemiring()
    )
    expected = testing.make_prov_set(
        [
            (0.7, 1),  # max(0.3, 0.7, 0.5) = 0.7
            (0.88, 2),  # only (2, d) exists
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


# ---------------------------------------------------------------------------
# Max-product semiring: conjunction (mul is same for both semirings)
# ---------------------------------------------------------------------------


def test_maxproduct_conjunction():
    """Conjunction probabilities are multiplied — same in both semirings."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(P, {(0.5, "a"), (0.9, "b")})
    cpl.add_probabilistic_facts_from_tuples(Q, {(0.4, "a"), (0.8, "b")})

    code = Union((Implication(R(x), Conjunction((P(x), Q(x)))),))
    cpl.walk(code)

    query = Implication(ans(x), R(x))
    result_prob = dalvi_suciu_lift.solve_succ_query(query, cpl)
    result_map = dalvi_suciu_lift.solve_succ_query(
        query, cpl, semiring=MaxProductSemiring()
    )

    expected = testing.make_prov_set(
        [
            (0.5 * 0.4, "a"),
            (0.9 * 0.8, "b"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result_prob, expected)
    assert testing.eq_prov_relations(result_map, expected)


# ---------------------------------------------------------------------------
# Viterbi forward pass pattern (2-step chain)
# ---------------------------------------------------------------------------


def test_viterbi_forward_2step():
    """Simulate a 2-step HMM Viterbi forward pass.

    The query structure mimics:

        viterbi(1, s)  :- init_prob(s, pi), emission(s, o1, em)
        viterbi(2, s)  :- viterbi(1, s'), transition(s', s, tr),
                           emission(s, o2, em)

    Here we test the non-recursive building blocks:

        ans1(s)   :- init(s), emit1(s)
        ans2(s)   :- ans1_prev(t_prev), transition(t_prev, s), emit2(s)

    The max-product semiring makes the existential projections over the
    previous state use max instead of sum, implementing the Viterbi
    max over paths.
    """
    cpl = CPLogicProgram()

    # -- step 1 facts --
    init = Symbol("init")
    trans = Symbol("trans")
    emit1 = Symbol("emit1")
    emit2 = Symbol("emit2")
    v1 = Symbol("viterbi_1")
    ans2_sym = Symbol("ans_2")

    # init_prob: π(state)  (these are deterministic facts for simplicity)
    cpl.add_probabilistic_facts_from_tuples(
        init, {(0.6, "s1"), (0.4, "s2")}
    )
    # emission at t=1
    cpl.add_probabilistic_facts_from_tuples(
        emit1, {(0.7, "s1"), (0.2, "s2")}
    )
    # emission at t=2
    cpl.add_probabilistic_facts_from_tuples(
        emit2, {(0.1, "s1"), (0.8, "s2")}
    )
    # transition(s', s, prob)
    cpl.add_probabilistic_facts_from_tuples(
        trans,
        {
            (0.8, "s1", "s1"),
            (0.2, "s1", "s2"),
            (0.3, "s2", "s1"),
            (0.7, "s2", "s2"),
        },
    )

    # Step 1: viterbi_1(s) = init(s) * emit1(s)
    code = Union((
        Implication(v1(x), Conjunction((init(x), emit1(x)))),
    ))
    cpl.walk(code)

    query1 = Implication(ans(x), v1(x))
    result1 = dalvi_suciu_lift.solve_succ_query(
        query1, cpl, semiring=ProbabilitySemiring()
    )
    # Expected: independent product
    expected1 = testing.make_prov_set(
        [
            (0.6 * 0.7, "s1"),
            (0.4 * 0.2, "s2"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result1, expected1)

    # Step 2: ans2(s) = max_{s'} [viterbi_1(s') * trans(s', s)] * emit2(s)
    # This is the key Viterbi step: max over previous states.
    #
    # The query: ans2(z) :- v1(y), trans(y, z), emit2(z)
    # The existential y is projected away -> MaxProductSemiring uses max.
    code2 = Union((
        Implication(ans2_sym(z), Conjunction((v1(y), trans(y, z), emit2(z)))),
    ))
    cpl.walk(code2)

    query2 = Implication(ans(x), ans2_sym(x))
    result2 = dalvi_suciu_lift.solve_succ_query(
        query2, cpl, semiring=MaxProductSemiring()
    )

    # Manually compute Viterbi step 2:
    # For s1: max(score1(s1)*trans(s1,s1), score1(s2)*trans(s2,s1)) * emit2(s1)
    v1_s1 = 0.6 * 0.7  # 0.42
    v1_s2 = 0.4 * 0.2  # 0.08
    score_s1 = max(v1_s1 * 0.8, v1_s2 * 0.3) * 0.1  # max(0.336, 0.024) * 0.1 = 0.0336
    score_s2 = max(v1_s1 * 0.2, v1_s2 * 0.7) * 0.8  # max(0.084, 0.056) * 0.8 = 0.0672
    expected2 = testing.make_prov_set(
        [
            (score_s1, "s1"),
            (score_s2, "s2"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result2, expected2)

    # Sanity: with the *probability* semiring this same query produces a
    # different result (1 - ∏(1-p) for the existential over y).
    result2_prob = dalvi_suciu_lift.solve_succ_query(
        query2, cpl, semiring=ProbabilitySemiring()
    )
    assert not testing.eq_prov_relations(result2, result2_prob), (
        "Max-product and probability semirings must differ on existential "
        "projection"
    )


# ---------------------------------------------------------------------------
# Max-product preserves multiplication commutativity
# ---------------------------------------------------------------------------


def test_maxproduct_mul_commutative():
    """Multiplication in max-product semiring (same as probability)."""
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(P, {(0.3, "a"), (0.5, "b")})
    cpl.add_probabilistic_facts_from_tuples(Q, {(0.4, "a"), (0.6, "b")})
    code = Union((Implication(R(x), Conjunction((P(x), Q(x)))),))
    cpl.walk(code)
    query = Implication(ans(x), R(x))

    result = dalvi_suciu_lift.solve_succ_query(
        query, cpl, semiring=MaxProductSemiring()
    )
    expected = testing.make_prov_set(
        [
            (0.3 * 0.4, "a"),
            (0.5 * 0.6, "b"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)
