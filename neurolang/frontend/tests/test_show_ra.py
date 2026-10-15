"""Tests for the relational algebra plan display (``show_ra``)."""
import operator
import re
from typing import AbstractSet

import pytest

from .. import NeurolangDL, NeurolangPDL
from ...datalog import Implication
from ...datalog.expressions import AdornedSymbol
from ...expressions import Constant, Symbol
from ...logic import Conjunction
from ...probabilistic import dalvi_suciu_lift
from ...relational_algebra import (
    NameColumns,
    NaturalJoin,
    str2columnstr_constant,
)
from ...utils import NamedRelationalAlgebraFrozenSet
from ...relational_algebra.pretty_printer import UpperBound, rows_label
from ..show_ra import (
    ShowRAOptions,
    add_rows,
    relation_rows,
    conditional_division_names,
    iterations_label,
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


def test_probabilistic_plans_name_their_solver(capsys):
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert "[solver: small dichotomy theorem (safe plan)]" in out


def test_probabilistic_plans_name_lifted_solver(capsys):
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_tuple_set([(1, "a"), (2, "b")], name="label")
    nl.add_probabilistic_facts_from_tuples(
        [(0.8, 1), (0.4, 2)], name="active"
    )
    out = show_ra_output(
        capsys,
        "z(l, PROB) :- active(x), label(x, l).\nans(l, q) :- z(l, q).",
        engine=nl,
    )
    assert "[solver: Dalvi–Suciu lifted inference]" in out


def knows_engine():
    """Return an engine whose ``knowing`` query needs the WMC solver.

    ``person`` is probabilistic, and ``knowing`` contains the #P-hard
    self-join ``∃z, w person(z) ∧ knows(z, w) ∧ person(w)``.
    """
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [(1.0, "a"), (1.0, "b")], name="person"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(0.5, "a", "b")], name="knows"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="chosen"
    )
    return nl


KNOWING = (
    "knowing(x, PROB) :- knows(x, y), person(x), person(y), "
    "knows(z, w), person(z), person(w).\n"
)


def test_wmc_plans_are_lineage_plans(capsys):
    out = show_ra_output(
        capsys, KNOWING + "ans(x, q) :- knowing(x, q).",
        engine=knows_engine(),
    )
    assert (
        "[solver: weighted model counting (knowledge compilation)]" in out
    )
    assert "lineage plan" in out
    assert "knows(x, y) ‹1 row›" in out
    assert "person(x) ‹2 rows›" in out


def test_probabilistic_stratum_is_solved_by_one_solver(capsys):
    """
    Check that all queries of a stratum show the solver solving them all.

    As in the evaluation, a query the small dichotomy solver can solve
    is solved, with the others of its stratum, by the first solver that
    solves them all (here WMC).
    """
    program = (
        KNOWING
        + "picking(x, PROB) :- chosen(x), person(x).\n"
        + "both(x, q, r) :- knowing(x, q), picking(x, r).\n"
        + "ans(x, q, r) :- both(x, q, r)."
    )
    out = show_ra_output(capsys, program, engine=knows_engine())
    assert out.count("[solver: weighted model counting") == 2
    assert "small dichotomy" not in out
    res = knows_engine().execute_datalog_program(program)
    assert len(res) == 1


def test_non_query_probabilistic_rules_are_inlined(capsys):
    out = show_ra_output(
        capsys,
        "al(x, l) :- active(x), label(x, l).\n"
        "ans(l, p) :- PROB[al(x, l) // picked(x)] = p.",
    )
    block = out.split("── rule al(x, l) :-")[1]
    assert block.split("──\n")[1].startswith(
        "(probabilistic rule: inlined into the queries using it"
    )


TRANSITIVE_CLOSURE = (
    "reach(x, y) :- edge(x, y).\n"
    "reach(x, y) :- reach(x, z), edge(z, y).\n"
    "ans(x, y) :- reach(x, y)."
)


def test_recursive_rules_printed_once_with_fixpoint(capsys):
    out = show_ra_output(capsys, TRANSITIVE_CLOSURE)
    assert out.count("── rule reach(x, y) :-\n    edge(x, y) ──") == 1
    assert out.count("── rule reach(x, y) :-\n    reach(x, z),") == 1
    assert "(fixpoint reached after 3 iterations)" in out


def test_recursive_rule_delta_plan_marked(capsys):
    """Check that plans binding a predicate to its delta are marked Δ."""
    out = show_ra_output(capsys, TRANSITIVE_CLOSURE)
    recursive = out.split("    reach(x, z),\n    edge(z, y) ──\n")[1]
    assert recursive.splitlines()[:6] == [
        "iteration 1:",
        "⋈",
        "├─ Δreach(x, z)",
        "└─ edge(z, y) ‹3 rows›",
        "iterations 2–3:",
        "⋈",
    ]
    assert "(Δreach: only the reach tuples derived in the current " \
        "iteration)" in out


def test_non_recursive_stratum_has_no_fixpoint(capsys):
    out = show_ra_output(capsys, THREE_WAY_JOIN)
    assert "fixpoint" not in out
    assert "iteration" not in out


@pytest.mark.parametrize("iterations,label", [
    ([1], "iteration 1"),
    ([2, 3], "iterations 2–3"),
    ([1, 2, 3, 5], "iterations 1–3, 5"),
])
def test_iterations_label(iterations, label):
    assert iterations_label(iterations) == label


COUNT_PER_LABEL = (
    "per_label(l, count(x)) :- label(x, l).\n"
    "ans(l, n) :- per_label(l, n)."
)


def test_aggregation_rule_plan_includes_the_aggregation(capsys):
    out = show_ra_output(capsys, COUNT_PER_LABEL)
    plan = out.split("── rule per_label(l, count(x)) :-\n    label(x, l) ──\n")
    assert plan[1].splitlines()[:2] == [
        "γ[by=(l), count(x)]",
        "└─ label(x, l) ‹3 rows›",
    ]


def test_query_on_a_named_predicate_is_printed(capsys):
    """Check that the query over a user predicate is printed last."""
    out = show_ra_output(capsys, COUNT_PER_LABEL)
    assert out.endswith(
        "── query ans(l, n) :-\n    per_label(l, n) ──\n"
        "per_label(l, n) ‹2 rows›\n"
    )


def test_query_with_constant_on_a_named_predicate(capsys):
    out = show_ra_output(
        capsys,
        "reach(x, y) :- edge(x, y).\n"
        "reach(x, y) :- reach(x, z), edge(z, y).\n"
        "ans(y) :- reach(1, y).",
    )
    # the constant is propagated by magic sets into the query rule
    assert "── rule ans(y) :-\n    reach^bf_0(1, y) ──" in out
    assert "reach^bf_0(1, y) ‹from 3 rows›" in out


def test_query_on_a_probabilistic_predicate_has_no_size(capsys):
    """Check that the query step over a probabilistic result has no size."""
    out = show_ra_output(
        capsys,
        "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
        "ans(x, q) :- z(x, q).",
    )
    assert out.endswith("── query ans(x, q) :-\n    z(x, q) ──\nz(x, q)\n")


CONDITIONAL = (
    "al(x, l) :- active(x), label(x, l).\n"
    "ans(l, p) :- PROB[al(x, l) // picked(x)] = p."
)


def test_postprobabilistic_rules_and_plans_are_printed(capsys):
    """Check the division of a conditional query is printed with a plan."""
    out = show_ra_output(capsys, CONDITIONAL)
    post = out.split("── post-probabilistic stratum ──\n")[1]
    assert post.startswith(
        "── rule ans(l, p) :-\n"
        "    p = p_num ÷ p_den,\n"
        "    ans^cond_num(l, p_num),\n"
        "    ans^cond_den(p_den) ──\n"
    )
    assert "π_ext[p_num / p_den -> p" in post
    assert "\n└─ ×\n" in post
    # The probabilistic results are not computed: no size is shown.
    assert "├─ ans^cond_num(l, p_num)\n" in post
    assert "└─ ans^cond_den(p_den)\n" in post


def _two_choices_engine():
    """Return an engine with uniform choices P over a, b and Q over a, c."""
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="P"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("c",)], name="Q"
    )
    return nl


def test_probabilistic_stratum_header(capsys):
    nl = _two_choices_engine()
    with nl.scope as e:
        e.Z[e.PROB[e.x], e.x] = e.P[e.x] & e.Q[e.x]
        nl.execute_datalog_program("ans(p, x) :- Z(p, x).", show_ra=True)
    out = capsys.readouterr().out
    assert "── deterministic stratum ──" in out
    assert "── probabilistic stratum ──" in out


def test_postprobabilistic_chain_of_rules(capsys):
    """
    Check that the post-probabilistic stratum prints its rules.

    ``ans`` uses the result of the probabilistic rule ``Z`` and ``ans2``
    uses ``ans``, so both land in the post-probabilistic stratum.
    """
    nl = _two_choices_engine()
    with nl.scope as e:
        e.Z[e.PROB[e.x], e.x] = e.P[e.x] & e.Q[e.x]
        e.ans[e.x, e.p] = e.Z[e.p, e.x]
        e.ans2[e.x] = e.ans[e.x, e.p] & (e.p > 0.1)
        nl.query((e.x,), e.ans2[e.x], show_ra=True)
    out = capsys.readouterr().out
    assert "── deterministic stratum ──" in out
    assert "── probabilistic stratum ──" in out
    assert "── post-probabilistic stratum ──" in out
    post = out.split("── post-probabilistic stratum ──")
    assert "── rule ans(x, p) :-\n    Z(p, x) ──\nZ(p, x)\n" in post[1]
    assert "── rule ans2(x) :-" in post[1]
    assert "σ[p > 0.1]" in post[1]


def test_division_names_need_fresh_unused_names():
    p, p1, p2, l = (Symbol.fresh() for _ in range(4))
    num = AdornedSymbol(Symbol("q"), "cond_num", None)
    den = AdornedSymbol(Symbol("q"), "cond_den", None)
    body = (
        num(l, p1), den(p2),
        Constant(operator.eq)(p, Constant(operator.truediv)(p1, p2)),
    )
    rule = Implication(Symbol("q")(l, p), Conjunction(body))
    assert conditional_division_names(rule) == {
        p.name: "p", p1.name: "p_num", p2.name: "p_den"
    }
    user_p = Symbol("p")
    clash = Implication(
        Symbol("q")(user_p, p), Conjunction(body + (Symbol("r")(user_p),))
    )
    assert conditional_division_names(clash) == {}
    plain = Implication(Symbol("q")(l), Conjunction((num(l, p1),)))
    assert conditional_division_names(plain) == {}


def test_conditional_query_header(capsys):
    out = show_ra_output(capsys, CONDITIONAL)
    prob = out.split("── probabilistic stratum ──\n")[1]
    conditional = prob.split(
        "── conditional query ans(l, p) = P(al(x, l) | picked(x)) ──\n"
    )[1]
    assert conditional.startswith(
        "numerator:\n── rule ans^cond_num(l, PROB(l)) :-"
    )
    assert "\ndenominator:\n── rule ans^cond_den(PROB()) :-" in conditional
    assert conditional.index("numerator:") < conditional.index(
        "denominator:"
    )


def test_conditional_query_header_shares_variable_names(capsys):
    """Check the header and both parts name fresh variables alike."""
    out = forward_inference_output(capsys)
    header = out.split("── conditional query ")[1].split(" ──\n")[0]
    # the rules are those rewritten with magic sets for 'memory'
    assert header.startswith("label_reports_0(s₀, p) = P(\n")
    assert "  | mentions^fb_0(s₁, 'memory'), selected_study(s₁)\n)" in header
    assert "schaefer_label^b_0(s₀)" in header
    assert "── rule label_reports_0^cond_den(PROB()) :-\n" \
        "    mentions^fb_0(s₁, 'memory'),\n" \
        "    selected_study(s₁) ──\n" in out


ANALYZE = ShowRAOptions(analyze=True)


def test_analyze_annotates_deterministic_plan_rows(capsys):
    out = show_ra_output(capsys, THREE_WAY_JOIN, show_ra=ANALYZE)
    assert "⋈ ‹1 row›" in out
    assert "└─ σ[w > 15] ‹1 row›" in out
    assert "edge(x, y) ‹3 rows›" in out


def test_analyze_partial_atom_rows(capsys):
    out = show_ra_output(capsys, "ans(x) :- edge(x, 2).", show_ra=ANALYZE)
    assert "edge(x, 2) ‹1 row from 3›" in out


def test_analyze_recursive_rows_summed_over_iterations(capsys):
    out = show_ra_output(capsys, TRANSITIVE_CLOSURE, show_ra=ANALYZE)
    assert "iterations 2–3:\n(rows summed over 2 iterations)\n⋈ ‹" in out
    assert "├─ Δreach(x, z) ‹3 rows›" in out
    assert out.endswith("reach(x, y) ‹6 rows›\n")


def test_analyze_aggregation_rows(capsys):
    out = show_ra_output(capsys, COUNT_PER_LABEL, show_ra=ANALYZE)
    assert "γ[by=(l), count(x)] ‹2 rows›" in out


@pytest.mark.parametrize("program", [
    "z(x, PROB) :- active(x), picked(x), label(x, l).\n"
    "ans(x, q) :- z(x, q).",
    CONDITIONAL,
])
def test_analyze_probabilistic_rows_match_evaluation(program, capsys):
    """Check the analyzed query produces as many rows as evaluating it."""
    expected = len(build_engine().execute_datalog_program(program))
    out = show_ra_output(capsys, program, show_ra=ANALYZE)
    final = [
        line for line in out.splitlines()
        if line.startswith(("π_ext[p_num / p_den", "z(x, q)"))
    ]
    assert final[-1].endswith(f"‹{expected} rows›")
    assert "σ[p > 0.0] ‹" in out


def test_analyze_conditional_parts_rows(capsys):
    out = show_ra_output(capsys, CONDITIONAL, show_ra=ANALYZE)
    assert "σ[p > 0.0] ‹2 rows›\n└─ γ[by=(l), p = sum(p)] ‹2 rows›" in out
    assert "σ[p > 0.0] ‹1 row›\n└─ γ[by=(), p = sum(p)] ‹1 row›" in out
    assert "├─ ans^cond_num(l, p_num) ‹2 rows›" in out


def test_analyze_wmc_has_no_row_counts(capsys):
    out = show_ra_output(
        capsys, KNOWING + "ans(x, q) :- knowing(x, q).",
        show_ra=ANALYZE, engine=knows_engine(),
    )
    assert "(no row counts: weighted model counting evaluates the " \
        "query as a whole)" in out
    assert out.endswith("knowing(x, q) ‹1 row›\n")


def test_analyze_does_not_change_evaluation():
    nl = build_engine()
    before = nl.execute_datalog_program(CONDITIONAL)
    nl.execute_datalog_program(CONDITIONAL, show_ra=ANALYZE)
    after = nl.execute_datalog_program(CONDITIONAL)
    assert before == after


@pytest.mark.parametrize("show_ra", [True, ANALYZE])
def test_show_ra_returns_empty_without_running_query(show_ra):
    res = build_engine().execute_datalog_program(THREE_WAY_JOIN, show_ra=show_ra)
    assert res.value.is_empty()


def test_relation_rows_of_expensive_values_is_an_upper_bound():
    class Region:
        def __hash__(self):
            return 0

    region = Region()
    relation = NamedRelationalAlgebraFrozenSet(
        ("r", "x"), [(region, 1), (region, 1), (region, 2)]
    )
    relation = relation.projection("r")
    rows = relation_rows(relation)
    assert isinstance(rows, UpperBound)
    assert rows >= 1
    assert rows_label(UpperBound(3)) == "‹≤3 rows›"
    plain = NamedRelationalAlgebraFrozenSet(("x",), [(1,), (2,)])
    assert relation_rows(plain.projection("x")) == 2
    total = {}
    add_rows(total, {1: 2})
    add_rows(total, {1: UpperBound(3)})
    assert total == {1: 5} and isinstance(total[1], UpperBound)
