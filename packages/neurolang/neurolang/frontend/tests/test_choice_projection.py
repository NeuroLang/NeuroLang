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

from nl_expr.exceptions import UnsupportedSolverError
from ...frontend import NeurolangPDL
from nl_core.probabilistic import dalvi_suciu_lift

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


def test_choice_also_within_a_negation_is_unified():
    """The choice occurs in the negation too: where selected_study(s)
    holds, selected_study(s2) holds only for s2 = s, which contradicts
    s2 ≠ s, and the negation is true."""
    choices = {"selected_study": STUDIES}
    program = (
        "q(t, PROB) :- mentions(s, t), selected_study(s), "
        "~exists(s2; mentions(s2, t), selected_study(s2), (s2 != s))"
    )
    expected = _enumerated(
        choices,
        lambda w, c, a: (c["selected_study"], a[0]) in w,
        answers=tuple((t,) for t in TERMS)
    )
    _assert_close(_solve(_engine(True, choices), program), expected)


@pytest.mark.parametrize("different", ["(s2 != s)", "(s != s2)"])
def test_studies_segregating_a_term_are_exclusive(different):
    """In ∃s seg(t, s), the atoms mentions(s2, t) of the negation do not
    have s, which is not a separator variable; but a study s segregating
    t mentions t, and another study s' ≠ s segregating it does not: the
    events of different studies are mutually exclusive, and their
    probabilities are added."""
    program = (
        SEGREGATION.replace("(s2 != s)", different) +
        "\nq(t, PROB) :- seg(t, s)"
    )
    expected = _enumerated(
        {},
        lambda w, c, a: any(_segregated(w, s, a[0]) for s in STUDIES),
        answers=tuple((t,) for t in TERMS)
    )
    _assert_close(_solve(_engine(True, {}), program), expected)


def test_terms_segregated_by_a_study_are_independent():
    """In ∃t seg(t, s), t is a separator variable: the events of
    different terms have different tuples of mentions."""
    program = SEGREGATION + "\nq(s, PROB) :- seg(t, s)"
    expected = _enumerated(
        {},
        lambda w, c, a: any(_segregated(w, a[0], t) for t in TERMS),
        answers=tuple((s,) for s in STUDIES)
    )
    _assert_close(_solve(_engine(True, {}), program), expected)


def test_negation_with_another_condition_is_not_exclusive():
    """Two studies may both mention t while no other study mentioning
    t mentions pain: the events of different studies are not
    exclusive, and are not added."""
    program = (
        "seg(t, s) :- mentions(s, t), "
        "~exists(s2; mentions(s2, t), mentions(s2, 'pain'), (s2 != s))"
        "\nq(t, PROB) :- seg(t, s)"
    )
    expected = _enumerated(
        {},
        lambda w, c, a: any(
            (s, a[0]) in w and not any(
                (s2, a[0]) in w and (s2, "pain") in w
                for s2 in STUDIES if s2 != s
            )
            for s in STUDIES
        ),
        answers=tuple((t,) for t in TERMS)
    )
    with pytest.raises(UnsupportedSolverError):
        _solve(_engine(True, {}), program)
    _assert_close(_solve(_engine(False, {}), program), expected)


@pytest.mark.parametrize("comparison, holds", [
    ("", lambda t1, t2: True),
    (", (t1 < t2)", lambda t1, t2: t1 < t2),
    (", (t1 != 'pain')", lambda t1, t2: t1 != "pain"),
])
def test_terms_mentioned_together_by_the_selected_study(comparison, holds):
    """mentions(s, t1) ∧ mentions(s, t2): different tuples when t1 ≠ t2,
    the same when t1 = t2, as Dalvi and Suciu take the answer variables
    for constants; a comparison of t1 and t2 selects the answers."""
    choices = {"selected_study": STUDIES}
    program = (
        "q(t1, t2, PROB) :- mentions(s, t1), mentions(s, t2), "
        f"selected_study(s){comparison}"
    )
    expected = _enumerated(
        choices,
        lambda w, c, a: (
            (c["selected_study"], a[0]) in w and
            (c["selected_study"], a[1]) in w and
            holds(*a)
        ),
        answers=tuple(itertools.product(TERMS, TERMS))
    )
    _assert_close(_solve(_engine(True, choices), program), expected)


def test_comparison_within_a_union_is_not_lifted():
    """A comparison of a rule of a union is not a conjunct of the query,
    and is left to the next solvers."""
    choices = {"selected_study": STUDIES}
    program = (
        "co(t1, t2, s) :- mentions(s, t1), mentions(s, t2), "
        "selected_study(s), (t1 < t2)\n"
        "co(t1, t2, s) :- mentions(s, t1), mentions(s, 'pain'), "
        "mentions(s, t2)\n"
        "q(t1, t2, PROB) :- co(t1, t2, s)"
    )
    with pytest.raises(UnsupportedSolverError):
        _solve(_engine(True, choices), program)
