"""The Möbius weights of inclusion-exclusion, computed with a containment
test that is not complete, see doc/inclusion_exclusion.rst.
"""

from fractions import Fraction
from random import Random

import pytest
from nl_expr.exceptions import NonLiftableException
from nl_expr.expressions import Symbol
from nl_expr.logic import Conjunction, Disjunction, Implication, Negation

from .. import dalvi_suciu_lift
from .test_lifted_negation import Q1, S2, T1, check, exists

A, B, D, Q, S, T, U, V = (Symbol(n) for n in "ABDQSTUV")
ans = Symbol("ans")
t, u, w, x, y, z = (Symbol(n) for n in "tuwxyz")

U1 = {(0.35, "a"), (0.45, "b")}


def disjunctions(clauses):
    """The disjunction of each non-empty set of `clauses`, with the set of
    their indices."""
    return [
        (frozenset(subset), Disjunction(tuple(clauses[i] for i in subset)))
        for subset in dalvi_suciu_lift.powerset(range(len(clauses)))
        if subset
    ]


def assert_is_a_transitive_order(order):
    for subset, below in order.items():
        assert subset not in below
        for other in below:
            assert order[other] <= below


def test_weights_of_equivalent_disjunctions():
    """K3 = D(x) ∧ (A(x) ∨ B(x)) implies K1 ∨ K2 = A(x) ∨ B(x), so
    K1 ∨ K2 ≡ K1 ∨ K2 ∨ K3: the two are one term, whose inclusion-exclusion
    coefficient is -1 + 1 = 0."""
    clauses = (
        A(x),
        B(x),
        Disjunction((Conjunction((D(x), A(x))), Conjunction((D(x), B(x))))),
    )
    weights, order = dalvi_suciu_lift._formulas_weights(disjunctions(clauses))

    assert set(order) == {
        frozenset(s) for s in ({0}, {1}, {2}, {0, 2}, {1, 2}, {0, 1, 2})
    }
    assert_is_a_transitive_order(order)
    for subset, below in order.items():
        for other in order:
            if other < subset:
                assert other in below
    assert weights == {
        frozenset({0}): 1,
        frozenset({1}): 1,
        frozenset({2}): 1,
        frozenset({0, 2}): -1,
        frozenset({1, 2}): -1,
        frozenset({0, 1, 2}): 0,
    }


def test_containments_found_later_are_closed_transitively(monkeypatch):
    """Z ≼ Y ≼ X, where the containment test finds Z ≼ Y, at the pair of
    indices (1, 2), but not Z ≼ X: the order must still have Z ≼ X,
    whatever the order in which the pairs are visited."""
    X, Y, Z = (Symbol(n) for n in "XYZ")
    implied_by = {(X, Y), (Y, Z)}
    monkeypatch.setattr(
        dalvi_suciu_lift,
        "is_contained",
        lambda q1, q2: q1 == q2 or (q1, q2) in implied_by,
    )
    sets = {X: frozenset({0}), Y: frozenset({1}), Z: frozenset({2})}

    weights, order = dalvi_suciu_lift._formulas_weights(
        [(sets[f], f) for f in (X, Y, Z)]
    )

    assert order[sets[X]] == {sets[Y], sets[Z]}
    assert_is_a_transitive_order(order)
    assert weights == {sets[Z]: 1, sets[Y]: 0, sets[X]: 0}


@pytest.mark.parametrize("clauses", (2, 3, 4))
def test_weights_with_any_sound_containment_test(monkeypatch, clauses):
    """For clauses that are random events, sets of worlds, and a sound
    containment test that finds a random part of the implications, not
    transitive and not even the inclusion of the subsets, the weighted sum
    is the probability of the conjunction (the lemma of
    doc/inclusion_exclusion.rst)."""
    random = Random(clauses)
    for _ in range(100):
        worlds = range(6)
        probability = {
            world: Fraction(random.randint(1, 9)) for world in worlds
        }
        events = [
            frozenset(world for world in worlds if random.random() < 0.6)
            for _ in range(clauses)
        ]
        powerset = [
            frozenset(subset)
            for subset in dalvi_suciu_lift.powerset(range(clauses))
            if subset
        ]
        event = {
            subset: frozenset().union(*(events[i] for i in subset))
            for subset in powerset
        }
        found = {
            (q1, q2): event[q2] <= event[q1] and random.random() < 0.5
            for q1 in powerset
            for q2 in powerset
        }
        monkeypatch.setattr(
            dalvi_suciu_lift, "is_contained", lambda q1, q2: found[q1, q2]
        )

        weights, order = dalvi_suciu_lift._formulas_weights(
            [(subset, subset) for subset in powerset]
        )

        def p(world_set):
            return sum(probability[world] for world in world_set) / sum(
                probability.values()
            )

        assert_is_a_transitive_order(order)
        for subset, below in order.items():
            assert all(other in below for other in order if other < subset)
        assert sum(
            weight * p(event[subset]) for subset, weight in weights.items()
        ) == p(frozenset.intersection(*events))


def test_inclusion_exclusion_of_three_clauses(monkeypatch):
    """∃y (S(x, y) ∧ Q(y)) ∧ ∃t (S(x, t) ∧ T(t)) ∧ ∃u (S(x, u) ∧ U(u)) has
    no separator variable: it is lifted by inclusion-exclusion on its
    three clauses, with seven terms."""
    lifted = []
    inclusion_exclusion = dalvi_suciu_lift.inclusion_exclusion_conjunction

    def spy(expression, symbol_table):
        lifted.append(expression)
        return inclusion_exclusion(expression, symbol_table)

    monkeypatch.setattr(
        dalvi_suciu_lift, "inclusion_exclusion_conjunction", spy
    )

    def has(database, a, relation):
        return exists(
            database[S], lambda c, d: c == a and (d,) in database[relation]
        )

    check(
        {S: S2, Q: Q1, T: T1, U: U1},
        {},
        [],
        Implication(
            ans(x),
            Conjunction((S(x, y), Q(y), S(x, t), T(t), S(x, u), U(u))),
        ),
        lambda db: {
            (a,)
            for a, _ in db[S]
            if has(db, a, Q) and has(db, a, T) and has(db, a, U)
        },
    )
    assert len(lifted) == 1
    assert len(lifted[0].formulas) == 3


V2 = {(0.6, "a", "b"), (0.2, "a", "c"), (0.9, "b", "a"), (0.4, "c", "c")}
U2 = {(0.5, "b", "a"), (0.3, "c", "b"), (0.8, "a", "a"), (0.6, "c", "c")}
Q2 = {(0.4, "b", "a"), (0.7, "c", "c")}


@pytest.mark.xfail(
    raises=NonLiftableException,
    strict=True,
    reason=(
        "∃w (V(x, z) ∧ U(z, w) ∧ ¬Q(z, w)), z free, is declined while its "
        "positive form is lifted: a term of inclusion-exclusion with a "
        "negated atom is not lifted"
    ),
)
def test_inclusion_exclusion_with_negated_atoms():
    """∃z (V(x, z) ∧ ∃w (U(z, w) ∧ ¬Q(z, w))) ∧ ∃z2 (V(x, z2) ∧ T(z2))
    is polynomial: its two clauses, and their disjunction, have the
    separator variable z."""
    z2 = Symbol("z2")

    def first(database, a):
        return exists(
            database[V],
            lambda c, d: c == a
            and exists(
                database[U],
                lambda e, f: e == d and (e, f) not in database[Q],
            ),
        )

    def second(database, a):
        return exists(database[V], lambda c, d: c == a and (d,) in database[T])

    check(
        {V: V2, U: U2, Q: Q2, T: T1},
        {},
        [],
        Implication(
            ans(x),
            Conjunction(
                (V(x, z), U(z, w), Negation(Q(z, w)), V(x, z2), T(z2))
            ),
        ),
        lambda db: {(a,) for a, _ in db[V] if first(db, a) and second(db, a)},
    )
