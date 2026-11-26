"""
A join on a variable that a relation holds as text and another as numbers:
the backends refused it with their own errors (a bare ``ValueError`` of
pandas), it is now one error of the relational algebra that names the
variable. Numbers of different types are still joined.
"""
import pytest

from nl_expr.exceptions import IncompatibleJoinColumnsError

from ...frontend import NeurolangDL, NeurolangPDL


def _engine(frontend, **kwargs):
    nl = frontend(**kwargs)
    nl.add_tuple_set([(0, "a"), (1, "b")], name="m")
    nl.add_tuple_set([(1,), (2,)], name="u")
    nl.add_tuple_set([(1.0,), (3.0,)], name="f")
    return nl


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize("program", [
    "ans(t) :- m(s, t), u(t)",
    "ans(t) :- u(t), m(s, t)",
    "ans(t) :- m(s, t), ~u(t)",
])
def test_text_and_numbers_are_not_joined(frontend, program):
    with pytest.raises(IncompatibleJoinColumnsError, match="'t'"):
        _engine(frontend).execute_datalog_program(program)


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_the_message_says_the_same_in_both_orders(frontend):
    messages = set()
    for program in ("ans(t) :- m(s, t), u(t)", "ans(t) :- u(t), m(s, t)"):
        with pytest.raises(IncompatibleJoinColumnsError) as error:
            _engine(frontend).execute_datalog_program(program)
        messages.add(str(error.value).split(" (types")[0])
    assert len(messages) == 1


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_numbers_of_different_types_are_joined(frontend):
    nl = _engine(frontend)
    nl.add_tuple_set([(1, 5)], name="n")
    result = nl.execute_datalog_program("ans(t) :- n(t, w), f(t)")
    assert sorted(result) == [(1,)]


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_a_comparison_of_text_and_a_number_is_empty(frontend):
    """The comparison, unlike the join, was never refused."""
    result = _engine(frontend).execute_datalog_program(
        "ans(t) :- m(s, t), (t == 1)"
    )
    assert list(result) == []


def test_the_probability_of_a_term_known_by_a_number():
    """
    The query that reported it: a union over a choice, joined with a
    probabilistic relation of numbers on the term, a text (on the default
    stack; the lifted solvers decline it as not liftable, with matching
    types as well).
    """
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(3)], name="selected_study"
    )
    nl.add_tuple_set([(0, "a"), (1, "a"), (2, "b")], name="mentions")
    nl.add_probabilistic_facts_from_tuples([(0.3, 1), (0.2, 2)], name="u0")
    with pytest.raises(IncompatibleJoinColumnsError, match="'t'"):
        nl.execute_datalog_program(
            "a(t, s) :- mentions(s, t), selected_study(s)\n"
            "ans(t, p) :- PROB[a(t, s), u0(t)] = p"
        )
