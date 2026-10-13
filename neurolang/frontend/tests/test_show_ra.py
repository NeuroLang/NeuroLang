"""Tests for the relational algebra plan display (``show_ra``)."""
import re

import pytest

from .. import NeurolangDL, NeurolangPDL
from ..show_ra import ShowRAOptions, show_ra_options


def build_engine():
    """
    Build a small engine to display plans on.

    It holds deterministic relations ``p``, ``edge``, ``label`` and
    ``weight``, the probabilistic facts ``active`` and the uniform
    probabilistic choice ``picked``.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,), (3,)], name="p")
    nl.add_tuple_set([(1, 2), (2, 3), (3, 4)], name="edge")
    nl.add_tuple_set([(1, "a"), (2, "b"), (3, "a")], name="label")
    nl.add_tuple_set([("a", 10), ("b", 20)], name="weight")
    nl.add_probabilistic_facts_from_tuples(
        [(0.8, 1), (0.4, 2), (0.6, 3)], name="active"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,), (3,)], name="picked"
    )
    return nl


def show_ra_output(capsys, program, show_ra=True, engine=None):
    """Return what displaying the plans of ``program`` prints."""
    if engine is None:
        engine = build_engine()
    engine.execute_datalog_program(program, show_ra=show_ra)
    return capsys.readouterr().out


def test_show_ra_options_defaults():
    options = ShowRAOptions()
    assert options.collapse is True
    assert options.analyze is False


@pytest.mark.parametrize(
    "show_ra,expected",
    [
        (False, None),
        (None, None),
        (True, ShowRAOptions()),
        (ShowRAOptions(analyze=True), ShowRAOptions(analyze=True)),
    ],
)
def test_show_ra_options(show_ra, expected):
    assert show_ra_options(show_ra) == expected


@pytest.mark.parametrize(
    "show_ra",
    [True, ShowRAOptions(), ShowRAOptions(collapse=False, analyze=True)],
)
@pytest.mark.parametrize("engine_class", [NeurolangDL, NeurolangPDL])
def test_show_ra_accepts_options(show_ra, engine_class, capsys):
    """Check that ``show_ra`` takes True or a ShowRAOptions."""
    nl = engine_class()
    nl.add_tuple_set([(1,), (2,)], name="p")
    nl.execute_datalog_program("ans(x) :- p(x).", show_ra=show_ra)
    assert "── deterministic stratum ──" in capsys.readouterr().out


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")


@pytest.mark.parametrize("engine_class", [NeurolangDL, NeurolangPDL])
def test_query_head_displayed_as_ans(engine_class, capsys):
    """Check that the predicate generated for a query is shown as ans."""
    nl = engine_class()
    nl.add_tuple_set([(1,), (2,)], name="p")
    out = show_ra_output(capsys, "ans(x) :- p(x).", engine=nl)
    assert "── rule ans(x) :-" in out
    assert not UUID.search(out)


def test_query_head_displayed_as_query_when_ans_exists(capsys):
    """Check that the query is shown as query when ans is a predicate."""
    nl = build_engine()
    nl.add_tuple_set([(1,)], name="ans")
    with nl.scope as e:
        nl.query((e.x,), e.p[e.x], show_ra=True)
    out = capsys.readouterr().out
    assert "── rule query(x) :-" in out
    assert not UUID.search(out)


def test_conditional_query_head_displayed_as_ans(capsys):
    """Check that the fresh predicate of a ``//`` query is shown as ans."""
    out = show_ra_output(
        capsys,
        "al(x, l) :- active(x), label(x, l).\n"
        "ans(l, p) :- PROB[al(x, l) // picked(x)] = p.",
    )
    assert "ans^cond_num(l, PROB(l))" in out
    assert "ans^cond_den(PROB())" in out
    assert "fresh_" not in out


def test_deterministic_leaves_show_their_size(capsys):
    out = show_ra_output(
        capsys, "ans(x, w) :- edge(x, y), label(y, l), weight(l, w)."
    )
    assert "edge ‹3 rows›" in out
    assert "label ‹3 rows›" in out
    assert "weight ‹2 rows›" in out


def test_recursive_leaves_have_no_size(capsys):
    """Check that relations growing at each iteration show no size."""
    out = show_ra_output(
        capsys,
        "reach(x, y) :- edge(x, y).\n"
        "reach(x, y) :- reach(x, z), edge(z, y).\n"
        "ans(x, y) :- reach(x, y).",
    )
    assert "edge ‹3 rows›" in out
    assert re.search(r"reach\b(?! ‹)", out)
    assert "reach ‹" not in out


def test_probabilistic_leaves_named_after_their_predicate(capsys):
    """Check that CP-Logic tables are shown by predicate name and size."""
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert "active ‹3 rows›" in out
    assert "picked ‹3 rows›" in out
    assert "label ‹3 rows›" in out
    assert "Table" not in out
