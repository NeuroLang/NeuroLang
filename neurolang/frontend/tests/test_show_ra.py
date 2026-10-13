"""Tests for the relational algebra plan display (``show_ra``)."""
import re
from typing import AbstractSet

import pytest

from .. import NeurolangDL, NeurolangPDL
from ...datalog import Implication
from ...expressions import Constant, Symbol
from ...logic import Conjunction
from ...relational_algebra import (
    NameColumns,
    NaturalJoin,
    str2columnstr_constant,
)
from ...utils import NamedRelationalAlgebraFrozenSet
from ..show_ra import (
    ShowRAOptions,
    probability_column_roles,
    show_ra_options,
)
from .test_probabilistic_frontend import build_neurosynth_like_engine


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
    assert "edge(x, y) ‹3 rows›" in out
    assert "label(y, l) ‹3 rows›" in out
    assert "weight(l, w) ‹2 rows›" in out


def test_recursive_leaves_have_no_size(capsys):
    """Check that relations growing at each iteration show no size."""
    out = show_ra_output(
        capsys,
        "reach(x, y) :- edge(x, y).\n"
        "reach(x, y) :- reach(x, z), edge(z, y).\n"
        "ans(x, y) :- reach(x, y).",
    )
    assert "edge(z, y) ‹3 rows›" in out
    assert "├─ reach(x, z)\n" in out
    assert "reach(x, z) ‹" not in out


def test_probabilistic_leaves_named_after_their_predicate(capsys):
    """Check that CP-Logic tables are shown by predicate name and size."""
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert "active(p(active), x) ‹3 rows›" in out
    assert "picked(p(picked), x) ‹3 rows›" in out
    assert "label(x, l) ‹3 rows›" in out
    assert "Table" not in out


def rule_blocks(out):
    """Split show-ra output into (rule header, plan) pairs."""
    blocks = []
    for chunk in out.split("── rule ")[1:]:
        header, _, plan = chunk.partition(" ──\n")
        blocks.append((header, plan.split("──")[0]))
    return blocks


def test_plan_columns_use_rule_variable_names(capsys):
    """
    Check that plan columns of fresh rule variables match the rule.

    SQUALL rules have fresh variables, displayed s₀, s₁, … in the rule;
    the plan columns that are those variables must be displayed the same.
    """
    nl = build_engine()
    nl.add_tuple_set([("a",), ("b",)], name="person")
    nl.add_tuple_set([("a", "b")], name="knows")
    nl.execute_squall_program(
        "obtain every Person that knows a Person.", show_ra=True
    )
    (header, plan), = rule_blocks(capsys.readouterr().out)
    assert "knows(s₀, s₁)" in header
    assert "knows(s₀, s₁)" in plan
    assert "fresh_" not in plan


NEUROSYNTH_INIT = (
    "schaefer_label(l) :- schaefer(l, r).\n"
    "labels(l, i, j, k) :- voxel(i, j, k), schaefer(l, r).\n"
    "mentions(s, t) :- term_in_study_tfidf(s, t, v), (v > 0.03).\n"
    "reports(s, i, j, k) :- peak_reported(i, j, k, s).\n"
)

FORWARD_INFERENCE = (
    "define as Label_reports with inferred probability every "
    "Schaefer_label that labels a Voxel in 3D that a Selected_study "
    "reports given the Selected_study mentions 'memory'. "
    "obtain every Label_reports ."
)


def forward_inference_output(capsys, show_ra=True):
    """Display the plans of a SQUALL forward inference query."""
    nl = build_neurosynth_like_engine()
    nl.execute_datalog_program(NEUROSYNTH_INIT)
    nl.execute_squall_program(FORWARD_INFERENCE, show_ra=show_ra)
    return capsys.readouterr().out


def test_fresh_plan_columns_do_not_collide_with_rule_variables(capsys):
    """Check that fresh columns that are no rule variable are c₀, c₁, …."""
    blocks = rule_blocks(forward_inference_output(capsys))
    probabilistic = [
        (header, plan) for header, plan in blocks if "PROB" in header
    ]
    assert len(probabilistic) == 2
    for header, plan in probabilistic:
        rule_vars = set(re.findall(r"s[₀-₉]+", header))
        assert rule_vars
        assert set(re.findall(r"s[₀-₉]+", plan)) <= rule_vars
        assert "fresh_" not in plan
        assert re.search(r"c[₀-₉]+", plan)


def test_probability_columns_labelled_by_role(capsys):
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert "σ[p > 0.0]" in out
    assert "γ[by=(x), p = sum(p)]" in out
    assert "π_ext[p(active) * p(picked) * 1.0 -> p" in out
    assert "active(p(active), x)" in out
    assert "picked(p(picked), x)" in out


def test_probability_label_avoids_rule_variable_p(capsys):
    """Check that the probability column is not called like a variable."""
    out = show_ra_output(
        capsys,
        "z(p, PROB) :- active(p), picked(p).\nans(x, q) :- z(x, q).",
    )
    assert "σ[prob > 0.0]" in out
    assert "p(active) * p(picked) -> prob" in out


def test_probability_columns_of_a_repeated_table_are_numbered():
    table = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("p", "x"), [(0.5, 1)]),
        verify_type=False,
    )
    x, y = Symbol("x"), Symbol("y")
    plan = NaturalJoin(
        NameColumns(table, (str2columnstr_constant("fresh_1"), str2columnstr_constant("x"))),
        NameColumns(table, (str2columnstr_constant("fresh_2"), str2columnstr_constant("y"))),
    )
    rule = Implication(Symbol("z")(x), Conjunction((
        Symbol("active")(x), Symbol("active")(y)
    )))
    roles = probability_column_roles(
        plan, "fresh_0", {id(table): "active"}, rule
    )
    assert roles == {
        "fresh_0": "p",
        "fresh_1": "p(active)₁",
        "fresh_2": "p(active)₂",
    }


THREE_WAY_JOIN = (
    "ans(x, w) :- edge(x, y), label(y, l), weight(l, w), (w > 15)."
)


def test_collapse_is_the_default(capsys):
    out = show_ra_output(capsys, THREE_WAY_JOIN)
    assert "├─ edge(x, y) ‹3 rows›" in out
    assert "weight(l, w) ‹2 rows›" in out
    assert "ρ[" not in out
    assert "π[" not in out


def test_collapse_false_prints_full_plans(capsys):
    out = show_ra_output(
        capsys, THREE_WAY_JOIN, show_ra=ShowRAOptions(collapse=False)
    )
    assert "ρ[x, y]" in out
    assert "π[#0, #1]" in out
    assert "edge ‹3 rows›" in out
    assert "edge(x, y)" not in out.split("──\n", 2)[-1]


def test_collapse_probabilistic_tables(capsys):
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert "active(p(active), x) ‹3 rows›" in out
    assert "π_ext[p(active) * p(picked) * 1.0 -> p, " in out
    assert "x -> x" not in out


def test_collapse_constant_argument(capsys):
    out = show_ra_output(capsys, "ans(x) :- edge(x, 2).")
    assert "edge(x, 2) ‹from 3 rows›" in out
