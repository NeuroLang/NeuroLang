"""
Queries with independent components, one of which is not liftable:

    P[Q_1 ∧ ... ∧ Q_m] = Π_i P[Q_i]

for components without a probabilistic relation in common. The
non-liftable query ∃x, y R(x) ∧ S(x, y) ∧ T(y) with n independent atoms
∃z U_i(z) took a time exponential in n to be declined: the lifted solvers
declined it after enumerating the subsets of its clauses. The expected
probabilities are computed by enumerating the possible worlds.
"""
import itertools
import random
import time

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from nl_expr.exceptions import NonLiftableException, UnsupportedSolverError

from ...frontend import NeurolangPDL

R_FACTS = {(1,): 0.5, (2,): 0.5}
S_FACTS = {(a, b): 0.4 for a in (1, 2) for b in (1, 2)}
T_FACTS = {(1,): 0.6, (2,): 0.6}
U_FACTS = {(1,): 0.3, (2,): 0.2}
TIME_LIMIT = 60


def _engine(lifted_only, atoms=0, cores=1):
    if lifted_only:
        nl = NeurolangPDL(
            probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
            probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
        )
    else:
        nl = NeurolangPDL()
    for k in range(cores):
        for name, facts in (("r", R_FACTS), ("s", S_FACTS), ("t", T_FACTS)):
            nl.add_probabilistic_facts_from_tuples(
                [(p,) + t for t, p in facts.items()], name=f"{name}{k}"
            )
    for i in range(atoms):
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + t for t, p in U_FACTS.items()], name=f"u{i}"
        )
    return nl


def _worlds(relations):
    """(probability, facts present) of the worlds of independent facts."""
    facts = [(n, t, p) for n, fs in relations.items() for t, p in fs.items()]
    for present in itertools.product((False, True), repeat=len(facts)):
        p = 1.
        for (_, _, q), is_present in zip(facts, present):
            p *= q if is_present else 1 - q
        yield p, {(n, t) for (n, t, _), ok in zip(facts, present) if ok}


def _core_holds(world, k=0):
    return any(
        (f"r{k}", (x,)) in world and (f"s{k}", (x, y)) in world
        and (f"t{k}", (y,)) in world
        for x in (1, 2) for y in (1, 2)
    )


def _core_probability(k=0):
    relations = {f"r{k}": R_FACTS, f"s{k}": S_FACTS, f"t{k}": T_FACTS}
    return sum(p for p, w in _worlds(relations) if _core_holds(w, k))


def _any_u(i):
    return 1 - (1 - U_FACTS[(1,)]) * (1 - U_FACTS[(2,)])


def _core(k):
    return f"r{k}(x{k}), s{k}(x{k}, y{k}), t{k}(y{k})"


def _solve(nl, program):
    [(probability,)] = nl.execute_datalog_program(program)
    return probability


def test_the_core_probability_is_that_of_the_enumeration():
    nl = _engine(False)
    assert _solve(nl, f"ans(p) :- PROB[{_core(0)}] = p") == pytest.approx(
        _core_probability()
    )


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize("atoms", (0, 4, 10))
def test_the_default_solvers_multiply_the_independent_atoms(atoms):
    nl = _engine(False, atoms)
    extra = "".join(f", u{i}(z{i})" for i in range(atoms))
    start = time.monotonic()
    found = _solve(nl, f"ans(p) :- PROB[{_core(0)}{extra}] = p")
    assert time.monotonic() - start < TIME_LIMIT
    assert found == pytest.approx(_core_probability() * _any_u(0) ** atoms)


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize("atoms", (0, 4, 10))
def test_the_lifted_solvers_decline_quickly(atoms):
    nl = _engine(True, atoms)
    extra = "".join(f", u{i}(z{i})" for i in range(atoms))
    start = time.monotonic()
    with pytest.raises((NonLiftableException, UnsupportedSolverError)):
        nl.execute_datalog_program(f"ans(p) :- PROB[{_core(0)}{extra}] = p")
    assert time.monotonic() - start < TIME_LIMIT


def test_independent_atoms_over_constants():
    nl = _engine(False, 3)
    found = _solve(
        nl, f"ans(p) :- PROB[{_core(0)}, u0(1), u1(1), u2(2)] = p"
    )
    expected = _core_probability() * 0.3 * 0.3 * 0.2
    assert found == pytest.approx(expected)


@pytest.mark.timeout(TIME_LIMIT)
def test_several_non_liftable_cores():
    nl = _engine(False, 3, cores=3)
    cores = ", ".join(_core(k) for k in range(3))
    extra = "".join(f", u{i}(z{i})" for i in range(3))
    start = time.monotonic()
    found = _solve(nl, f"ans(p) :- PROB[{cores}{extra}] = p")
    assert time.monotonic() - start < TIME_LIMIT
    assert found == pytest.approx(_core_probability() ** 3 * _any_u(0) ** 3)


@pytest.mark.timeout(TIME_LIMIT)
def test_several_non_liftable_cores_are_declined_quickly():
    nl = _engine(True, 3, cores=4)
    cores = ", ".join(_core(k) for k in range(4))
    extra = "".join(f", u{i}(z{i})" for i in range(3))
    start = time.monotonic()
    with pytest.raises((NonLiftableException, UnsupportedSolverError)):
        nl.execute_datalog_program(f"ans(p) :- PROB[{cores}{extra}] = p")
    # 0.3 s; the subsets of the clauses took 26 s
    assert time.monotonic() - start < 10


def test_a_non_liftable_core_with_an_independent_negation():
    nl = _engine(False, 1)
    found = _solve(nl, f"ans(p) :- PROB[{_core(0)}, ~u0(1)] = p")
    assert found == pytest.approx(_core_probability() * (1 - 0.3))


def test_a_liftable_query_with_an_independent_negated_existential():
    """R(x) ∧ S(x, y) ∧ ¬∃z U(z) is lifted: both components are."""
    relations = {"r0": R_FACTS, "s0": S_FACTS}
    expected = sum(
        p for p, w in _worlds(relations)
        if any(("r0", (x,)) in w and ("s0", (x, y)) in w
               for x in (1, 2) for y in (1, 2))
    ) * (1 - _any_u(0))
    nl = _engine(True, 1)
    found = _solve(
        nl, "ans(p) :- PROB[r0(x), s0(x, y), ~exists(z; u0(z))] = p"
    )
    assert found == pytest.approx(expected)


def test_atoms_with_an_answer_variable_in_common_are_not_apart():
    """Components that share a free variable, here x, are lifted
    together: the answers are those of the enumeration."""
    nl = _engine(True, 2)
    result = nl.execute_datalog_program(
        "ans(x, p) :- PROB[r0(x), s0(x, y), t0(y), u0(x), u1(x)] = p"
    )
    relations = {
        "r0": R_FACTS, "s0": S_FACTS, "t0": T_FACTS, "u0": U_FACTS,
        "u1": U_FACTS,
    }
    expected = {
        x: sum(
            p for p, w in _worlds(relations)
            if ("r0", (x,)) in w and ("u0", (x,)) in w and ("u1", (x,)) in w
            and any(("s0", (x, y)) in w and ("t0", (y,)) in w
                    for y in (1, 2))
        )
        for x in (1, 2)
    }
    assert dict(result) == pytest.approx(expected)


def _choice_engine(c_facts, r_facts, t_facts, extra=0):
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_probabilistic_choice_from_tuples(
        [(p,) + t for t, p in c_facts.items()], name="C"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p,) + t for t, p in r_facts.items()], name="R"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p,) + t for t, p in t_facts.items()], name="T"
    )
    for i in range(extra):
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + t for t, p in U_FACTS.items()], name=f"u{i}"
        )
    return nl


def _random_choice_facts(seed):
    rng = random.Random(seed)
    pairs = list(itertools.product("ab", repeat=2))
    weights = [rng.random() + .1 for _ in pairs]
    c_facts = {t: w / sum(weights) for t, w in zip(pairs, weights)}
    r_facts = {(v,): rng.uniform(.1, .9) for v in "ab"}
    t_facts = {(v,): rng.uniform(.1, .9) for v in "ab"}
    return c_facts, r_facts, t_facts


@pytest.mark.parametrize("seed", range(4))
def test_independent_components_with_a_free_variable(seed):
    """h(z) ← C(x, y) ∧ R(x) ∧ T(z) and h(z) ← C(y, x) ∧ T(x) ∧ T(z), with
    C a probabilistic choice: the component of the answer variable z is
    apart from the union on the choice, which is lifted, as the answers
    of the enumeration. The solvers declined this query."""
    c_facts, r_facts, t_facts = _random_choice_facts(seed)
    nl = _choice_engine(c_facts, r_facts, t_facts)
    result = nl.execute_datalog_program(
        "h(z) :- C(x, y), R(x), T(z)\n"
        "h(z) :- C(y, x), T(x), T(z)\n"
        "ans(z, p) :- PROB[h(z)] = p"
    )
    expected = {z: 0. for z in "ab"}
    for (c1, c2), p_c in c_facts.items():
        for r_present in itertools.product((False, True), repeat=2):
            r = {v for v, ok in zip("ab", r_present) if ok}
            p_r = 1.
            for v, ok in zip("ab", r_present):
                p_r *= r_facts[(v,)] if ok else 1 - r_facts[(v,)]
            for t_present in itertools.product((False, True), repeat=2):
                t = {v for v, ok in zip("ab", t_present) if ok}
                p_t = 1.
                for v, ok in zip("ab", t_present):
                    p_t *= t_facts[(v,)] if ok else 1 - t_facts[(v,)]
                for z in t:
                    if c1 in r or c2 in t:
                        expected[z] += p_c * p_r * p_t
    assert dict(result) == pytest.approx(expected)


def test_a_lifted_union_on_a_choice_with_an_independent_atom():
    """P[h ∧ ∃z U(z)] = P[h] P[∃z U(z)] for the union of the rules
    C(x, y) ∧ R(x) and C(y, x) ∧ T(x) on a choice, lifted apart from the
    atom, and still lifted."""
    c_facts = {("a", "b"): .5, ("b", "a"): .3, ("a", "a"): .2}
    r_facts = {("a",): .4, ("b",): .7}
    t_facts = {("a",): .6}
    nl = _choice_engine(c_facts, r_facts, t_facts, extra=1)
    [(found,)] = nl.execute_datalog_program(
        "h() :- C(x, y), R(x)\nh() :- C(y, x), T(x)\n"
        "ans(PROB) :- h(), u0(z)"
    )
    base = .5 * .4 + .3 * (1 - .3 * .4) + .2 * (1 - .6 * .4)
    assert found == pytest.approx(base * _any_u(0))


W_VALUES = (1, 2)


def _answer_engine(lifted_only, atoms):
    """The relations of the core with an answer variable w in all of them."""
    if lifted_only:
        nl = NeurolangPDL(
            probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
            probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
        )
    else:
        nl = NeurolangPDL()
    for name, facts in (("r", R_FACTS), ("s", S_FACTS), ("t", T_FACTS)):
        nl.add_probabilistic_facts_from_tuples(
            [(p, w) + t for t, p in facts.items() for w in W_VALUES],
            name=name,
        )
    for i in range(atoms):
        nl.add_probabilistic_facts_from_tuples(
            [(p, w) + t for t, p in U_FACTS.items() for w in W_VALUES],
            name=f"u{i}",
        )
    return nl


def _answer_core(atoms):
    return "r(w, x), s(w, x, y), t(w, y)" + "".join(
        f", u{i}(w, z{i})" for i in range(atoms)
    )


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize("atoms", (0, 4, 8))
def test_atoms_with_the_answer_variable_of_a_non_liftable_core(atoms):
    """The answers w of a core that is not liftable for each w, with atoms
    of w: the atoms are lifted apart from the core, as they have the same
    columns, and the query is declined quickly. It took a time exponential
    in the number of atoms."""
    nl = _answer_engine(True, atoms)
    start = time.monotonic()
    with pytest.raises((NonLiftableException, UnsupportedSolverError)):
        nl.execute_datalog_program(
            f"ans(w, p) :- PROB[{_answer_core(atoms)}] = p"
        )
    assert time.monotonic() - start < TIME_LIMIT


@pytest.mark.timeout(TIME_LIMIT)
def test_the_default_solvers_multiply_the_atoms_of_the_answer_variable():
    atoms = 6
    nl = _answer_engine(False, atoms)
    result = nl.execute_datalog_program(
        f"ans(w, p) :- PROB[{_answer_core(atoms)}] = p"
    )
    expected = _core_probability() * _any_u(0) ** atoms
    assert dict(result) == pytest.approx({w: expected for w in W_VALUES})
