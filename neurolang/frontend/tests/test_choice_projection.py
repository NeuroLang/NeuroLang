"""
Disjoint projection on a probabilistic choice, its body being lifted with
every rule:

    P[∃X (C(ū) ∧ φ)] = Σ_a P[C(ū)[a/X]] P[φ[a/X]]

for a choice C whose atom has the variables X and occurs nowhere else.
The expected probabilities are computed by enumerating the possible
worlds.
"""
import itertools

import pytest

from ...exceptions import UnsupportedSolverError
from ...frontend import NeurolangPDL
from ...probabilistic import dalvi_suciu_lift

MENTIONS = {
    ("s1", "pain"): 0.4, ("s2", "pain"): 0.7, ("s1", "memory"): 0.2,
    ("s3", "memory"): 0.5, ("s2", "fear"): 0.3,
}
STUDIES = ("s1", "s2", "s3")
TERMS = ("pain", "memory", "fear")
SEGREGATION = (
    "seg(t, s) :- mentions(s, t), ~exists(s2; mentions(s2, t), (s2 != s))"
)


def _engine(lifted_only, choices):
    if lifted_only:
        nl = NeurolangPDL(
            probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
            probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
        )
    else:
        nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [(p,) + f for f, p in MENTIONS.items()], name="mentions"
    )
    for name, values in choices.items():
        nl.add_uniform_probabilistic_choice_over_set(
            [(v,) for v in values], name=name
        )
    return nl


def _solve(nl, program):
    nl.execute_datalog_program(program)
    df = nl.solve_all()["q"].as_pandas_dataframe()
    probability = df.columns[-1]
    return {
        tuple(row[:-1]): row[-1]
        for row in df[list(df.columns[:-1]) + [probability]].itertuples(
            index=False
        )
        if row[-1] > 0
    }


def _enumerated(choices, holds, answers=((),)):
    """Probability of each answer for which `holds(world, chosen,
    answer)`, `chosen` mapping each choice to its value."""
    facts = list(MENTIONS)
    result = {}
    names = list(choices)
    for chosen_values in itertools.product(*(choices[n] for n in names)):
        chosen = dict(zip(names, chosen_values))
        p_chosen = 1.
        for name in names:
            p_chosen /= len(choices[name])
        for present in itertools.product((False, True), repeat=len(facts)):
            world = {f for f, is_present in zip(facts, present) if is_present}
            p = p_chosen
            for f, is_present in zip(facts, present):
                p *= MENTIONS[f] if is_present else 1 - MENTIONS[f]
            for answer in answers:
                if holds(world, chosen, answer):
                    result[answer] = result.get(answer, 0) + p
    return result


def _segregated(world, s, t):
    return (s, t) in world and not any(
        (s2, t) in world for s2 in STUDIES if s2 != s
    )


def _assert_close(result, expected):
    assert result.keys() == expected.keys()
    for answer, p in expected.items():
        assert result[answer] == pytest.approx(p)


def test_segregation_on_the_selected_study_is_lifted():
    """P[selected study mentions t and no other study does]."""
    choices = {"selected_study": STUDIES}
    program = SEGREGATION + "\nq(t, PROB) :- seg(t, s), selected_study(s)"
    expected = _enumerated(
        choices,
        lambda w, c, a: _segregated(w, c["selected_study"], a[0]),
        answers=tuple((t,) for t in TERMS)
    )
    nl = _engine(True, choices)
    _assert_close(_solve(nl, program), expected)


def test_segregation_on_the_selected_study_without_answer_variables():
    choices = {"selected_study": STUDIES}
    program = SEGREGATION + "\nq(PROB) :- seg(t, s), selected_study(s)"
    expected = _enumerated(
        choices,
        lambda w, c, _: any(
            _segregated(w, c["selected_study"], t) for t in TERMS
        )
    )
    nl = _engine(True, choices)
    _assert_close(_solve(nl, program), expected)


def test_closed_conjunct_outside_the_quantifier():
    """A ∧ ∃X ψ ≡ ∃X (A ∧ ψ): a closed deterministic atom outside the
    quantified variables does not prevent the projection."""
    choices = {"selected_study": STUDIES}
    program = SEGREGATION + (
        "\nq(t, PROB) :- flag('on'), seg(t, s), selected_study(s)"
    )
    expected = _enumerated(
        choices,
        lambda w, c, a: _segregated(w, c["selected_study"], a[0]),
        answers=tuple((t,) for t in TERMS)
    )
    nl = _engine(True, choices)
    nl.add_tuple_set([("on",)], name="flag")
    _assert_close(_solve(nl, program), expected)


def test_two_choices_are_projected_in_turn():
    choices = {"selected_study": STUDIES, "selected_term": TERMS}
    program = SEGREGATION + (
        "\nq(PROB) :- seg(t, s), selected_study(s), selected_term(t)"
    )
    expected = _enumerated(
        choices,
        lambda w, c, _: _segregated(
            w, c["selected_study"], c["selected_term"]
        )
    )
    nl = _engine(True, choices)
    _assert_close(_solve(nl, program), expected)


def test_choice_also_within_a_negation_is_not_projected():
    """The choice occurs in the negation too: its events are not
    independent of the rest, and the query is left to the next
    solvers."""
    choices = {"selected_study": STUDIES}
    program = (
        "q(t, PROB) :- mentions(s, t), selected_study(s), "
        "~exists(s2; mentions(s2, t), selected_study(s2), (s2 != s))"
    )
    with pytest.raises(UnsupportedSolverError):
        _solve(_engine(True, choices), program)


def test_variable_which_is_not_a_separator_is_not_projected():
    """In ∃s seg(t, s), the atoms mentions(s2, t) of the negation do not
    have s: the events of different studies share tuples, and the query
    is left to the next solvers, which answer it."""
    program = SEGREGATION + "\nq(t, PROB) :- seg(t, s)"
    expected = _enumerated(
        {},
        lambda w, c, a: any(_segregated(w, s, a[0]) for s in STUDIES),
        answers=tuple((t,) for t in TERMS)
    )
    with pytest.raises(UnsupportedSolverError):
        _solve(_engine(True, {}), program)
    _assert_close(_solve(_engine(False, {}), program), expected)
