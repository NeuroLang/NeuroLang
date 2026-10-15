"""
Lifted inference for non-repeating queries with negation, following
Fink & Olteanu (PODS 2014): a query in which no probabilistic relation
occurs twice can be computed in polynomial time if it is hierarchical.

The expected probabilities are closed forms, checked against an
enumeration of the possible worlds.
"""
import pytest

from ...exceptions import UnsupportedSolverError
from ...expressions import Symbol
from ...logic import Conjunction, Implication, Negation, Union
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram

ans = Symbol("ans")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
D = Symbol("D")
F = Symbol("F")
M = Symbol("M")
R = Symbol("R")
S = Symbol("S")
T = Symbol("T")
U = Symbol("U")
Z = Symbol("Z")
s = Symbol("s")
t = Symbol("t")
x = Symbol("x")
y = Symbol("y")

R_FACTS = {(0.2, "a"), (0.7, "b"), (0.5, "c")}
S_FACTS = {
    (0.3, "a", "a"), (0.6, "a", "b"), (0.8, "b", "c"), (0.45, "c", "c")
}


def _program(rules, pfacts=None, pchoices=None, deterministic=None):
    cpl_program = CPLogicProgram()
    for symbol, facts in (pfacts or {}).items():
        cpl_program.add_probabilistic_facts_from_tuples(symbol, facts)
    for symbol, facts in (pchoices or {}).items():
        cpl_program.add_probabilistic_choice_from_tuples(symbol, facts)
    for symbol, tuples in (deterministic or {}).items():
        cpl_program.add_extensional_predicate_from_tuples(symbol, tuples)
    cpl_program.walk(Union(rules))
    return cpl_program


def _solve(rules, query, **relations):
    return dalvi_suciu_lift.solve_succ_query(
        query, _program(rules, **relations)
    )


def test_existential_over_a_negated_existential():
    """∃x R(x) ∧ ¬∃y S(x, y), whose quantifiers are mixed once the
    negations are moved to the atoms."""
    rules = (
        Implication(A(x), S(x, y)),
        Implication(Z(), Conjunction((R(x), Negation(A(x))))),
    )
    result = _solve(
        rules, Implication(ans(), Z()), pfacts={R: R_FACTS, S: S_FACTS}
    )
    expected = 1 - (
        (1 - 0.2 * 0.7 * 0.4) * (1 - 0.7 * 0.2) * (1 - 0.5 * 0.55)
    )
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


@pytest.mark.parametrize("boolean", (False, True))
def test_nested_negation(boolean):
    """R(x) ∧ ¬∃y (S(x, y) ∧ ¬U(x, y))"""
    rules = (
        Implication(B(x, y), Conjunction((S(x, y), Negation(U(x, y))))),
        Implication(A(x), B(x, y)),
        Implication(Z(x), Conjunction((R(x), Negation(A(x))))),
    )
    per_x = {
        "a": 0.2 * (1 - 0.3 * 0.65) * (1 - 0.6),
        "b": 0.7 * (1 - 0.8 * 0.85),
        "c": 0.5 * (1 - 0.45),
    }
    pfacts = {
        R: R_FACTS, S: S_FACTS, U: {(0.35, "a", "a"), (0.15, "b", "c")}
    }
    if boolean:
        query = Implication(ans(), Z(x))
        p = 1
        for value in per_x.values():
            p *= 1 - value
        expected = testing.make_prov_set([(1 - p,)], ("_p_",))
    else:
        query = Implication(ans(x), Z(x))
        expected = testing.make_prov_set(
            [(p, value) for value, p in per_x.items()], ("_p_", "x")
        )
    result = _solve(rules, query, pfacts=pfacts)
    assert testing.eq_prov_relations(result, expected)


def test_deterministic_atom_in_a_negated_existential():
    """∃x R(x) ∧ ¬∃y (S(x, y) ∧ D(y)), with D deterministic."""
    rules = (
        Implication(A(x), Conjunction((S(x, y), D(y)))),
        Implication(Z(), Conjunction((R(x), Negation(A(x))))),
    )
    result = _solve(
        rules, Implication(ans(), Z()),
        pfacts={R: R_FACTS, S: S_FACTS},
        deterministic={D: {("a",), ("c",)}},
    )
    expected = 1 - (1 - 0.2 * 0.7) * (1 - 0.7 * 0.2) * (1 - 0.5 * 0.55)
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


def test_choice_with_a_negated_existential():
    """∃s C(s) ∧ ¬∃t M(s, t), C being a probabilistic choice: the events
    for different s are mutually exclusive."""
    rules = (
        Implication(A(s), M(s, t)),
        Implication(Z(), Conjunction((C(s), Negation(A(s))))),
    )
    result = _solve(
        rules, Implication(ans(), Z()),
        pchoices={C: {(0.3, "s1"), (0.5, "s2"), (0.2, "s3")}},
        pfacts={M: {(0.5, "s1", "t1"), (0.5, "s1", "t2"), (0.1, "s2", "t1")}},
    )
    expected = 0.3 * 0.25 + 0.5 * 0.9 + 0.2
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


def test_choice_with_a_negated_fact():
    """∃s C(s) ∧ ¬F(s)"""
    rules = (Implication(Z(), Conjunction((C(s), Negation(F(s))))),)
    result = _solve(
        rules, Implication(ans(), Z()),
        pchoices={C: {(0.3, "s1"), (0.5, "s2"), (0.2, "s3")}},
        pfacts={F: {(0.4, "s1"), (0.9, "s3")}},
    )
    expected = 0.3 * 0.6 + 0.5 + 0.2 * 0.1
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


def test_non_hierarchical_query_is_not_lifted():
    """∃x R(x) ∧ ¬∃y (S(x, y) ∧ T(y)) is #P-hard."""
    rules = (
        Implication(A(x), Conjunction((S(x, y), T(y)))),
        Implication(Z(), Conjunction((R(x), Negation(A(x))))),
    )
    with pytest.raises(UnsupportedSolverError):
        _solve(
            rules, Implication(ans(), Z()),
            pfacts={
                R: R_FACTS, S: S_FACTS,
                T: {(0.4, "a"), (0.9, "b"), (0.25, "c")},
            },
        )
