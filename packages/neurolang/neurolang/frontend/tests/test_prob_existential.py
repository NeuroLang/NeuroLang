"""
PROB and MARG over an existential, ``PROB[exists(y; s(x, y), t(y))]``: its
body may be a conjunction or a disjunction, with several variables, and
conditioned with ``//``. The expected probabilities are computed by
enumerating the possible worlds, over engines that have only the lifted
solvers.
"""
import itertools

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from ...frontend import NeurolangPDL

S = {(1, "a"): 0.5, (1, "b"): 0.4, (2, "a"): 0.3, (3, "c"): 0.8}
T = {("a",): 0.6, ("b",): 0.7, ("c",): 0.2}
U = {(1,): 0.9, (2,): 0.5, (3,): 0.4}
R = {(1, "c"): 0.35, (2, "b"): 0.6}
FACTS = {"s": S, "t": T, "u": U, "r": R}
XS = (1, 2, 3)


def _engine():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    for name, facts in FACTS.items():
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + f for f, p in facts.items()], name=name
        )
    return nl


def _solve(program):
    nl = _engine()
    nl.execute_datalog_program(program)
    df = nl.solve_all()["q"].as_pandas_dataframe()
    return {
        tuple(row[:-1]): row[-1] for row in df.itertuples(index=False)
        if row[-1] > 0
    }


def _enumerated(holds, knowing=None, answers=((),)):
    """P[holds(world, answer)], conditioned on `knowing(world, answer)`,
    by enumerating the worlds of the independent facts."""
    facts = [(n, f) for n, fs in FACTS.items() for f in fs]
    num = {a: 0. for a in answers}
    den = {a: 0. for a in answers}
    for present in itertools.product((False, True), repeat=len(facts)):
        p = 1.
        for (n, f), is_present in zip(facts, present):
            p *= FACTS[n][f] if is_present else 1 - FACTS[n][f]
        world = {f for f, is_present in zip(facts, present) if is_present}
        for a in answers:
            if knowing is None or knowing(world, a):
                den[a] += p
                num[a] += p * holds(world, a)
    return {a: num[a] / den[a] for a in answers if den[a] > 0 and num[a] > 0}


def _assert_close(found, expected):
    assert found.keys() == expected.keys()
    for k, v in expected.items():
        assert found[k] == pytest.approx(v)


def _has(world, name, fact):
    return (name, fact) in world


def _linked(world, x):
    return any(
        _has(world, "s", (x, y)) and _has(world, "t", (y,))
        for y in "abc"
    )


@pytest.mark.parametrize("keyword", ["PROB", "MARG"])
def test_existential_over_a_conjunction_without_group(keyword):
    found = _solve(f"q(p) :- {keyword}[exists(y; s(x, y), t(y))] = p")
    _assert_close(
        found,
        _enumerated(lambda w, a: any(_linked(w, x) for x in XS)),
    )


@pytest.mark.parametrize("keyword", ["PROB", "MARG"])
def test_existential_over_a_conjunction_by_group(keyword):
    found = _solve(f"q(x, p) :- {keyword}[exists(y; s(x, y), t(y))] = p")
    _assert_close(
        found,
        _enumerated(
            lambda w, a: _linked(w, a[0]), answers=[(x,) for x in XS]
        ),
    )


def test_several_existential_variables():
    found = _solve("q(p) :- PROB[exists(x, y; s(x, y), t(y))] = p")
    _assert_close(
        found,
        _enumerated(lambda w, a: any(_linked(w, x) for x in XS)),
    )


def test_existential_over_a_disjunction():
    found = _solve("q(x, p) :- PROB[exists(y; (s(x, y) | r(x, y)))] = p")
    _assert_close(
        found,
        _enumerated(
            lambda w, a: any(
                _has(w, "s", (a[0], y)) or _has(w, "r", (a[0], y))
                for y in "abc"
            ),
            answers=[(x,) for x in XS],
        ),
    )


def test_existential_in_the_event_conditioned():
    found = _solve(
        "q(x, p) :- PROB[exists(y; s(x, y), t(y)) // u(x)] = p"
    )
    _assert_close(
        found,
        _enumerated(
            lambda w, a: _linked(w, a[0]),
            knowing=lambda w, a: _has(w, "u", (a[0],)),
            answers=[(x,) for x in XS],
        ),
    )


def test_existential_in_what_is_known():
    found = _solve(
        "q(x, p) :- PROB[u(x) // exists(y; s(x, y), t(y))] = p"
    )
    _assert_close(
        found,
        _enumerated(
            lambda w, a: _has(w, "u", (a[0],)),
            knowing=lambda w, a: _linked(w, a[0]),
            answers=[(x,) for x in XS],
        ),
    )


@pytest.mark.parametrize("by_group", [False, True])
def test_marg_of_a_single_atom(by_group):
    """MARG[s(x, y)], with or without a group, is the PROB of the atom."""
    head = "q(x, p)" if by_group else "q(p)"
    found = _solve(f"{head} :- MARG[s(x, y)] = p")
    if by_group:
        expected = _enumerated(
            lambda w, a: any(_has(w, "s", (a[0], y)) for y in "abc"),
            answers=[(x,) for x in XS],
        )
    else:
        expected = _enumerated(lambda w, a: any(
            _has(w, "s", (x, y)) for x in XS for y in "abc"
        ))
    _assert_close(found, expected)
