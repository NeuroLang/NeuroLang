"""
Ranking of queries in which a variable precedes another in an atom and
follows it in another, Dalvi & Suciu (J. ACM 2012), Proposition 4.2.

The expected probabilities are computed by enumerating the possible
worlds of the probabilistic facts.
"""
import itertools

import pytest

from ...exceptions import NotRankedException, UnsupportedSolverError
from ...frontend import NeurolangPDL
from ...probabilistic import dalvi_suciu_lift

R_FACTS = {
    ("a", "b"): 0.5, ("b", "a"): 0.4, ("a", "a"): 0.3,
    ("b", "c"): 0.6, ("c", "b"): 0.2,
}


def _worlds(relations):
    """Possible worlds of tuple-independent `relations`, a mapping from
    names to mappings from tuples to probabilities, with their
    probabilities."""
    facts = [
        (name, t, p) for name, tuples in relations.items()
        for t, p in tuples.items()
    ]
    for present in itertools.product((False, True), repeat=len(facts)):
        world = {name: set() for name in relations}
        probability = 1.
        for (name, t, p), is_present in zip(facts, present):
            probability *= p if is_present else 1 - p
            if is_present:
                world[name].add(t)
        yield world, probability


def _enumerated(relations, holds, answers=((),)):
    """Probability of each answer for which `holds(world, answer)`."""
    result = {}
    for world, probability in _worlds(relations):
        for answer in answers:
            if holds(world, answer):
                result[answer] = result.get(answer, 0) + probability
    return result


def _engine(relations, solvers=None, choices=None):
    if solvers is None:
        nl = NeurolangPDL()
    else:
        nl = NeurolangPDL(
            probabilistic_solvers=tuple(
                s.solve_succ_query for s in solvers
            ),
            probabilistic_marg_solvers=tuple(
                s.solve_marg_query for s in solvers
            ),
        )
    for name, tuples in relations.items():
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + t for t, p in tuples.items()], name=name
        )
    for name, tuples in (choices or {}).items():
        nl.add_probabilistic_choice_from_tuples(
            [(p,) + t for t, p in tuples.items()], name=name
        )
    return nl


def _solve(nl, program):
    nl.execute_datalog_program(program)
    df = nl.solve_all()["q"].as_pandas_dataframe()
    return {
        tuple(row[1:]): row[0] for row in df[
            [df.columns[-1]] + list(df.columns[:-1])
        ].itertuples(index=False)
        if row[0] > 0
    }


def _assert_close(result, expected):
    assert result.keys() == expected.keys()
    for answer, probability in expected.items():
        assert result[answer] == pytest.approx(probability)


def test_symmetric_pair_is_lifted():
    """∃x, y R(x, y) ∧ R(y, x), the example of Dalvi & Suciu, Sect. 4.1:
    x is not a separator variable, as it is first in R(x, y) and second
    in R(y, x), but the query is tractable once ranked."""
    expected = _enumerated(
        {"R": R_FACTS},
        lambda w, _: any((x, y) in w["R"] and (y, x) in w["R"]
                         for x, y in R_FACTS)
    )
    nl = _engine({"R": R_FACTS}, solvers=(dalvi_suciu_lift,))
    _assert_close(_solve(nl, "q(PROB) :- R(x, y), R(y, x)"), expected)


def test_answer_variable_to_rank_is_left_to_the_next_solver():
    """Dalvi & Suciu rank Boolean queries, the answer variables being
    constants: q(x) :- R(x, y), R(y, x) would need ranking x."""
    values = ("a", "b", "c")
    program = "q(x, PROB) :- R(x, y), R(y, x)"
    nl = _engine({"R": R_FACTS}, solvers=(dalvi_suciu_lift,))
    with pytest.raises(NotRankedException):
        _solve(nl, program)
    expected = _enumerated(
        {"R": R_FACTS},
        lambda w, a: any((a[0], y) in w["R"] and (y, a[0]) in w["R"]
                         for y in values),
        answers=tuple((v,) for v in values)
    )
    _assert_close(_solve(_engine({"R": R_FACTS}), program), expected)


def test_symmetric_pair_with_another_relation_is_answered():
    """∃x, y R(x, y) ∧ R(y, x) ∧ S(x): ranked, the disjuncts for x < y and
    y < x have S on different variables, and the union is not lifted.
    The next solver answers it."""
    s_facts = {("a",): 0.7, ("b",): 0.1}
    relations = {"R": R_FACTS, "S": s_facts}
    program = "q(PROB) :- R(x, y), R(y, x), S(x)"
    expected = _enumerated(
        relations,
        lambda w, _: any(
            (x, y) in w["R"] and (y, x) in w["R"] and (x,) in w["S"]
            for x, y in R_FACTS
        )
    )
    nl = _engine(relations, solvers=(dalvi_suciu_lift,))
    with pytest.raises(UnsupportedSolverError):
        _solve(nl, program)
    _assert_close(_solve(_engine(relations), program), expected)


@pytest.mark.parametrize("program, relations, holds, choices", (
    pytest.param(
        "q(PROB) :- C(x, y), R(y, x)",
        {"R": {(1., 2.): 0.6, (2., 1.): 0.4, (1., 1.): 0.7}},
        None,
        {"C": {(1., 2.): 0.5, (2., 1.): 0.3, (1., 1.): 0.2}},
        id="probabilistic choice",
    ),
    pytest.param(
        "q(PROB) :- R(x, y), R(y, x)",
        {"R": {
            (frozenset({1}), frozenset({1, 2})): 0.5,
            (frozenset({1, 2}), frozenset({1})): 0.4,
        }},
        lambda w, _: any((x, y) in w["R"] and (y, x) in w["R"]
                         for x, y in w["R"]),
        None,
        id="sets, not totally ordered",
    ),
    pytest.param(
        "q(PROB) :- R(x, y), R(y, x), R(z, w), R(w, z), R(u, z)",
        {"R": R_FACTS},
        lambda w, _: any(
            (x, y) in w["R"] and (y, x) in w["R"] for x, y in w["R"]
        ) and any(
            (z, v) in w["R"] and (v, z) in w["R"] and
            any(t[1] == z for t in w["R"])
            for z, v in w["R"]
        ),
        None,
        id="too many variables to rank",
    ),
))
def test_queries_not_ranked_are_left_to_the_next_solver(
    program, relations, holds, choices
):
    nl = _engine(relations, solvers=(dalvi_suciu_lift,), choices=choices)
    with pytest.raises(NotRankedException):
        _solve(nl, program)
    if holds is None:
        # A single tuple c of the choice C is true, and the query holds if
        # R has its reverse: the sum of P[c] P[R(c[1], c[0])].
        expected = {(): 0.5 * 0.4 + 0.3 * 0.6 + 0.2 * 0.7}
    else:
        expected = _enumerated(relations, holds)
    _assert_close(_solve(_engine(relations, choices=choices), program),
                  expected)


def test_negated_atom_to_rank_is_not_ranked():
    """R(x, y) ∧ ¬R(y, x): splitting the negated atom by the order of x
    and y would need the order to hold, so the query is not ranked."""
    nl = _engine({"R": R_FACTS}, solvers=(dalvi_suciu_lift,))
    with pytest.raises(UnsupportedSolverError):
        _solve(nl, "q(PROB) :- R(x, y), ~R(y, x)")


def test_two_answer_variables_to_rank_are_left_to_the_next_solver():
    program = "q(x, y, PROB) :- R(x, y), R(y, x)"
    nl = _engine({"R": R_FACTS}, solvers=(dalvi_suciu_lift,))
    with pytest.raises(NotRankedException):
        _solve(nl, program)
    pairs = tuple(R_FACTS)
    expected = _enumerated(
        {"R": R_FACTS},
        lambda w, a: a in w["R"] and a[::-1] in w["R"],
        answers=pairs
    )
    _assert_close(_solve(_engine({"R": R_FACTS}), program), expected)
