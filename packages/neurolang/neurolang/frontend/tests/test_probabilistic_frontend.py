import itertools
from typing import AbstractSet, Callable, Tuple

import numpy as np
import pandas as pd
import pytest

from nl_expr.exceptions import (
    NegativeFormulaNotNamedRelationException,
    NegativeFormulaNotSafeRangeException,
    NeuroLangException,
    SymbolNotFoundError,
    UnboundVariablesError,
    UnsupportedProgramError,
    UnsupportedQueryError,
    WrongArgumentsInPredicateError,
)
from nl_expr.expressions import Symbol
from nl_expr.logic import Union
from nl_core.datalog.horn_clauses import Fol2DatalogTranslationException
from nl_core.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
)
from nl_core.probabilistic.exceptions import (
    ForbiddenConditionalQueryNonConjunctive,
    RepeatedTuplesInProbabilisticRelationError,
    UnsupportedSolverError,
)
from ...regions import ExplicitVBR, SphericalVolume
from nl_core.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from ..probabilistic_frontend import NeurolangPDL


def assert_almost_equal(set_a, set_b):
    for tupl_a in set_a:
        assert any(
            all(
                np.isclose(term_a, term_b)
                if isinstance(term_a, (float, np.float32, np.float64))
                else term_a == term_b
                for term_a, term_b in zip(tupl_a, tupl_b)
            )
            for tupl_b in set_b
        )


def test_add_uniform_probabilistic_choice_set():
    nl = NeurolangPDL()

    prob = [(a,) for a in range(10)]
    prob_set = nl.add_uniform_probabilistic_choice_over_set(prob, "prob")
    res = nl[prob_set]

    assert prob_set.type is AbstractSet[Tuple[float, int]]
    assert res.type is AbstractSet[Tuple[float, int]]
    assert res.value == frozenset((1 / len(prob), a) for a in range(10))

    d = [("a",), ("b",), ("c",), ("d",)]
    data = nl.add_uniform_probabilistic_choice_over_set(d, name="data")
    res_d = nl[data]

    assert data.type is AbstractSet[Tuple[float, str]]
    assert res_d.type is AbstractSet[Tuple[float, str]]

    d = pd.DataFrame([("a",), ("b",), ("c",), ("d",)], columns=["ids"])
    data2 = nl.add_uniform_probabilistic_choice_over_set(d, name="data2")
    res_d2 = nl[data]

    assert data2.type is AbstractSet[Tuple[float, str]]
    assert res_d2.type is AbstractSet[Tuple[float, str]]


def build_neurosynth_like_engine():
    """
    Build a small engine with the EDB of the neurosynth engine.

    It holds a few studies, peaks, terms and (label, region) Schaefer
    pairs, and a uniform probabilistic choice over the studies.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(10, 20, 30, "s1"), (11, 21, 31, "s2")], name="peak_reported"
    )
    nl.add_tuple_set([("s1",), ("s2",)], name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="selected_study"
    )
    nl.add_tuple_set(
        [("memory", "s1", 0.5), ("memory", "s2", 0.6)],
        name="term_in_study_tfidf",
    )
    nl.add_tuple_set([("A", "rA"), ("B", "rB")], name="schaefer")
    nl.add_tuple_set([(10, 20, 30), (11, 21, 31)], name="voxel")
    return nl


def test_show_ra_deterministic_stratum(capsys):
    from neurolang.frontend import NeurolangPDL
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    nl.execute_datalog_program("ans(x) :- P(x).", show_ra=True)
    captured = capsys.readouterr()
    assert "── deterministic stratum ──" in captured.out
    assert "P" in captured.out


def test_show_ra_conditional_query_provenance():
    """
    Check that show-ra builds the plan of a conditional (MARG) rule.

    The frontend usually rewrites ``//`` into SUCC queries, but a rule
    whose body is still a ``Condition`` (e.g. one built directly in the
    IR) reaches the show-ra MARG branch, which referenced an unbound
    solver and failed.
    """
    from nl_core.datalog import Implication
    from nl_core.datalog.instance import MapInstance
    from nl_expr.expressions import Symbol
    from nl_core.probabilistic.expressions import (
        PROB,
        Condition,
        ProbabilisticQuery,
    )
    from nl_core.probabilistic.query_resolution import (
        _build_probabilistic_program,
    )
    from nl_core.relational_algebra_provenance import ProvenanceAlgebraSet
    from ..show_ra import ShowRAStratumSolver

    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="P"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(0.5, "a"), (0.3, "b")], name="Q"
    )
    x = Symbol("x")
    rule = Implication(
        Symbol("Z")(x, ProbabilisticQuery(PROB, (x,))),
        Condition(Symbol("P")(x), Symbol("Q")(x)),
    )
    cpl, prob_idb = _build_probabilistic_program(
        MapInstance(),
        nl.program_ir.probabilistic_facts(),
        nl.program_ir.probabilistic_choices(),
        Union((rule,)),
        nl.check_qbased_pfact_tuple_unicity,
    )
    provset = ShowRAStratumSolver(nl).rule_provenance_set(
        prob_idb.formulas[0], cpl
    )
    assert isinstance(provset, ProvenanceAlgebraSet)


def test_show_ra_probabilistic_with_det_idb():
    """
    Regression: --show-ra with a conditional query over deterministic IDB.

    --show-ra crashed with IndexError at
    relational_algebra_provenance.py:423 when a conditional MARG query
    depends on deterministic IDB rules and magic rewrite fails.
    """
    nl = build_neurosynth_like_engine()
    with nl.scope as e:
        e.schaefer_label[e.l] = e.schaefer[e.l, e.r]
        e.labels[e.l, e.x, e.y, e.z] = (
            e.voxel[e.x, e.y, e.z] & e.schaefer[e.l, e.r]
        )
        e.reports[e.s, e.x, e.y, e.z] = e.peak_reported[e.x, e.y, e.z, e.s]
        e.mentions[e.s, e.t] = (
            e.term_in_study_tfidf[e.s, e.t, e.v] & (e.v > 0.03)
        )
        nl.execute_squall_program(
            "define as Label_reports with inferred probability every "
            "Schaefer_label that labels a Voxel in 3D that a Selected_study "
            "reports given the Selected_study mentions 'memory'."
        )
        # Direct query on the WLQ predicate (no ans(...) wrapper) hits the
        # probabilistic IDB path, which falls back to _solve after magic
        # rewrite fails and crashes in selection_rap_eq_columnint.
        # Before the fix this raises IndexError; after the fix it returns
        # without error and the probabilistic stratum is printed.
        res = nl.query(
            (e.l, e.p), e.label_reports[e.l, e.p], show_ra=True
        )
        assert res is not None


@pytest.mark.parametrize("show_ra", [False, True])
def test_show_ra_reports_wrong_arity_error(show_ra):
    """
    Check that show-ra reports an arity error as evaluation does.

    ``schaefer`` has one column but the rule uses two. The show-ra chase
    used to swallow the error and print the plan of an empty relation.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([("A",), ("B",)], name="schaefer")
    with pytest.raises(WrongArgumentsInPredicateError):
        nl.execute_datalog_program(
            "label(l) :- schaefer(l, r).\nans(l) :- label(l).",
            show_ra=show_ra,
        )


@pytest.mark.parametrize("show_ra", [False, True])
def test_show_ra_reports_unsolvable_query_error(show_ra, capsys):
    """
    Check that show-ra reports why a query without magic sets fails.

    SQUALL turns the noun ``P`` into the predicate ``p``, which does not
    exist. Show-ra used to hide the error and print a deterministic plan.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    with pytest.raises(SymbolNotFoundError, match="Symbol not found p"):
        nl.execute_squall_program("obtain every P.", show_ra=show_ra)
    assert "── deterministic stratum ──" not in capsys.readouterr().out


def test_show_rewritten_datalog_prints_all_rules(capsys):
    """--show-rewritten must print all reachable Datalog rules, not just the final query."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    nl.execute_datalog_program(
        """
        active(x) :- person(x), plays(x).
        fanatic(x) :- active(x), plays(x).
        ans(x) :- fanatic(x).
        """,
        show_rewritten=True,
        dry_run=True,
    )
    captured = capsys.readouterr()
    assert "active" in captured.out
    assert "fanatic" in captured.out


@pytest.mark.parametrize(
    "flags",
    [
        dict(dry_run=True),
        dict(show_rewritten=True),
        dict(show_rewritten=True, show_ra=True),
    ],
)
def test_show_rewritten_query_on_rule_with_constant(flags, capsys):
    """
    Check that the rewritten program of a query on a rule is printed.

    The query on ``found`` is rewritten with magic sets, through the
    constant ``'pain'``. Printing it used to fail with a KeyError on the
    rewritten query predicate, whose rules were not in the program yet.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([("s1", "pain"), ("s2", "noise")], name="mentions")
    nl.add_tuple_set([("s1",), ("s2",)], name="study")
    nl.execute_datalog_program(
        "talks(s, t) :- mentions(s, t)\n"
        "found(s) :- study(s), talks(s, 'pain')"
    )
    with nl.scope as e:
        nl.query((e.s,), e.found(e.s), **flags)
    out = capsys.readouterr().out
    assert "── rewritten program ──" in out
    assert "magic_talks" in out


def test_dry_run_prints_magic_rules_of_a_bound_query(capsys):
    """
    A dry run of a query binding a rule's argument prints its magic rules.

    They used to be missing from the program when the rewritten program
    was printed, which made the dry run fail with a KeyError.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="study")
    nl.add_tuple_set([(1, "pain", 0.5), (2, "fear", 0.5)], name="tfidf")
    nl.execute_datalog_program("mentions(s, t) :- tfidf(s, t, v), (v > 0.03)")
    with nl.scope:
        nl.execute_squall_program(
            "obtain every Study that mentions 'pain'.", dry_run=True
        )
    out = capsys.readouterr().out
    assert "magic_mentions" in out
    assert "tfidf(s, t, v)" in out


SIMILARITIES = {
    ("math", "a"): .9, ("math", "b"): .8, ("math", "c"): .1,
    ("lang", "c"): .9,
}

CLOSE = "close(q, s) :- study(s) & (sim == f(q, s)) & (sim > .5)\n"
P_GIVEN_CLOSE = (
    "p_given_close(q, i, PROB) :- (peak(i, s) & selected_study(s)) "
    "// (close(q, s) & selected_study(s))\n"
)


def build_similarity_engine():
    """
    Build an engine with a scalar function ``f(q, s)`` of a query text.

    ``close(q, s)`` holds for the studies ``s`` similar to ``q``: a and b
    for 'math', c for 'lang'. Study a reports peaks 1 and 2, b reports
    3 and c reports 1.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, "a"), (2, "a"), (3, "b"), (1, "c")], name="peak")
    nl.add_tuple_set([("a",), ("b",), ("c",)], name="study")
    nl.add_tuple_set([("math",)], name="topic")
    nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="selected_study"
    )

    @nl.add_symbol
    def f(q: str, s: str) -> float:
        return SIMILARITIES.get((q, s), 0.)

    return nl


def test_marg_query_constant_flows_through_rule_head():
    """
    Check that a constant of a query binds a variable of a MARG rule.

    ``q`` is bound only by the constant 'math' of the query, which magic
    sets carry through the head of ``p_given_close`` into ``close``. The
    result is that of binding ``q`` with ``topic(q)`` in ``close``.
    """
    expected = {(1, .5), (2, .5), (3, .5)}
    nl = build_similarity_engine()
    res = nl.execute_datalog_program(
        CLOSE + P_GIVEN_CLOSE + "ans(i, p) :- p_given_close('math', i, p)"
    )
    assert set(res) == expected

    nl = build_similarity_engine()
    res = nl.execute_datalog_program(
        CLOSE.replace("study(s)", "topic(q) & study(s)") + P_GIVEN_CLOSE
        + "ans(i, p) :- p_given_close('math', i, p)"
    )
    assert set(res) == expected


def test_marg_query_conditioned_on_rule_with_constant():
    """Check a MARG rule conditioned on a rule with a constant."""
    nl = build_similarity_engine()
    res = nl.execute_datalog_program(
        CLOSE
        + "math_study(s) :- close('math', s)\n"
        + "p_given_math(i, PROB) :- (peak(i, s) & selected_study(s)) "
        "// (math_study(s) & selected_study(s))\n"
        + "ans(i, p) :- p_given_math(i, p)"
    )
    assert set(res) == {(1, .5), (2, .5), (3, .5)}


def test_marg_query_several_constants():
    """
    Check a MARG rule called with two constants in one rule.

    The probabilities of 'math' cannot bind the call for 'lang', as they
    are computed after the magic sets.
    """
    nl = build_similarity_engine()
    res = nl.execute_datalog_program(
        CLOSE + P_GIVEN_CLOSE
        + "both(i, p, r) :- p_given_close('math', i, p) "
        "& p_given_close('lang', i, r)\n"
        + "ans(i, p, r) :- both(i, p, r)"
    )
    assert set(res) == {(1, .5, 1.)}


def test_marg_query_unbound_variable():
    """Check the error of a MARG query leaving a function argument free."""
    nl = build_similarity_engine()
    with pytest.raises(UnboundVariablesError, match=r"In close: .*\bq\b"):
        nl.execute_datalog_program(
            CLOSE + P_GIVEN_CLOSE + "ans(q, i, p) :- p_given_close(q, i, p)"
        )


def test_deterministic_query():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    with nl.scope as e:
        e.query1[e.y] = data1[e.x] & data2[e.x, e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    q1 = res["query1"].as_pandas_dataframe().values
    assert len(q1) == 4
    for elem in q1:
        assert elem[0] in ["a", "b", "c", "d"]


def test_probabilistic_query():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_uniform_probabilistic_choice_over_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_uniform_probabilistic_choice_over_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.query1[e.y, e.PROB[e.y]] = data1[e.x] & data2[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.query1[e.y] & data3[e.y]
        with pytest.raises(UnsupportedProgramError):
            nl.solve_all()


def test_marg_query():
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        {(0.2, "a"), (0.3, "b"), (0.5, "c")}, name="P"
    )
    nl.add_tuple_set({(4, "a"), (5, "a"), (5, "b")}, name="Q")
    nl.add_probabilistic_facts_from_tuples(
        {
            (0.2, 1, "a"),
            (0.1, 1, "b"),
            (0.7, 2, "b"),
            (0.9, 2, "c"),
        },
        name="R",
    )

    with nl.scope as e:
        e.Z[e.x, e.z, e.PROB[e.x, e.z]] = (
            e.Q(e.x, e.y) & e.P(e.y)
        ) // (e.R(e.z, e.y) & e.P(e.y))
        res = nl.solve_all()

    expected = RelationalAlgebraFrozenSet(
        {
            (4, 1, 0.2 * 0.2 / (0.2 * 0.2 + 0.3 * 0.1)),
            (5, 1, (0.2 * 0.2 + 0.3 * 0.1) / (0.2 * 0.2 + 0.3 * 0.1)),
            (5, 2, 0.3 * 0.7 / (0.3 * 0.7 + 0.9 * 0.5)),
        }
    )
    assert_almost_equal(res["Z"], expected)


def test_mixed_queries():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",), ("d",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.tmp[e.x, e.y] = data1[e.x] & data2[e.x, e.y]
        e.query1[e.x, e.y, e.PROB[e.x, e.y]] = e.tmp[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.tmp[e.x, e.y] & data3[e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    assert "query2" in res.keys()
    assert len(res["query2"].columns) == 2
    q2 = res["query2"].as_pandas_dataframe().values
    assert len(q2) == 4
    for elem in q2:
        assert elem[1] == 0.25
        assert elem[0] in ["a", "b", "c", "d"]


def test_simple_within_language_succ_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("e",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.solve_all()
    assert "Z" in res.keys()
    df = res["Z"].as_pandas_dataframe()
    assert len(df) == 1
    assert tuple(df.values[0])[0] == "a"
    assert np.isclose(tuple(df.values[0])[1], 1 / 9)


def test_within_language_succ_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [
            ("a", "b"),
            ("b", "c"),
            ("b", "d"),
        ],
        name="P",
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x, e.y] & Q[e.x]
        res = nl.query((e.x, e.p), e.Z(e.x, e.p))
    expected = RelationalAlgebraFrozenSet(
        [
            ("b", 1 / 3 * 1 / 2 + 1 / 3 * 1 / 2),
            ("a", 1 / 3 * 1 / 2),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.query((e.x, e.p), e.Z[e.x, e.p])
        res_2 = nl.query(e.p, e.Z(e.x, e.p))
    expected = RelationalAlgebraFrozenSet(
        [
            ("a", 1 / 9),
            ("c", 1 / 9),
        ]
    )
    assert_almost_equal(res, expected)
    assert res.row_type == Tuple[str, float]
    assert_almost_equal(res_2, expected.projection(1))
    assert res_2.row_type == Tuple[float]


def test_solve_query_prob_col_not_last():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.PROB[e.x], e.x] = P[e.x] & Q[e.x]
        res = nl.query((e.p, e.x), e.Z[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        [
            (1 / 9, "a"),
            (1 / 9, "c"),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_boolean_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.ans[e.PROB()] = P[e.x] & Q[e.x]
        res = nl.query((e.p), e.ans[e.p])

    expected = RelationalAlgebraFrozenSet([(2 / 9,)])
    assert_almost_equal(res, expected)


def test_solve_query_with_constant():
    nl = NeurolangPDL()
    P = nl.add_probabilistic_choice_from_tuples(
        {(0.2, "a"), (0.3, "b"), (0.5, "c")}, name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.ans[e.x, e.PROB(e.x)] = P[e.x] & Q[e.x]
        res = nl.query((e.p), e.ans["a", e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            (0.2 * 1/3,),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_complex_stratified_query():
    """
    R(1, 2) : 0.3
    R(1, 4) : 0.7
    R(2, 2) : 0.2
    R(2, 4) : 0.6
    R(2, 6) : 0.8
    Q(4) : 0.2 v Q(6) : 0.8
    A(x, PROB[x])       :- Q(x), R(1, x), R(2, x)
    B(x, y, PROB[x, y]) :- Q(y), R(1, x), R(2, x)
    C(x, y, p1, p2)     :- A(x, p1), B(x, y, p2)

    """
    nl = NeurolangPDL()
    R = nl.add_probabilistic_facts_from_tuples(
        [(0.3, 1, 2), (0.7, 1, 4), (0.2, 2, 2), (0.6, 2, 4), (0.8, 2, 6)],
        name="R",
    )
    Q = nl.add_probabilistic_choice_from_tuples([(0.2, 4), (0.8, 6)], name="Q")
    with nl.scope as e:
        e.A[e.x, e.PROB[e.x]] = Q[e.x] & R[1, e.x] & R[2, e.x]
        e.B[e.x, e.y, e.PROB[e.x, e.y]] = Q[e.y] & R[1, e.x] & R[2, e.x]
        e.C[e.x, e.y, e.p1, e.p2] = e.A[e.x, e.p1] & e.B[e.x, e.y, e.p2]
        res = nl.query((e.x, e.y, e.h, e.z), e.C[e.x, e.y, e.h, e.z])
    expected = RelationalAlgebraFrozenSet(
        [
            (4, 4, 0.2 * 0.7 * 0.6, 0.2 * 0.7 * 0.6),
            (4, 6, 0.2 * 0.7 * 0.6, 0.8 * 0.7 * 0.6),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_complex_stratified_query_with_deterministic_part():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("a",), ("b",), ("c",)], name="A")
    B = nl.add_tuple_set([("a",), ("b",)], name="B")
    P = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.8, "b")],
        name="P",
    )
    with nl.scope as e:
        e.C[e.x, e.y] = A[e.x] & A[e.y] & B[e.x]
        e.D[e.x, e.y, e.PROB[e.x, e.y]] = e.C[e.x, e.y] & P[e.y]
        res = nl.query((e.x, e.y, e.p), e.D[e.x, e.y, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            ("a", "a", 0.2),
            ("a", "b", 0.8),
            ("b", "a", 0.2),
            ("b", "b", 0.8),
        ]
    )
    assert_almost_equal(res, expected)


def test_neurolange_dl_deterministic_negation():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    s[x, y] = ~q(x, x) & q(x, y)

    res = neurolang.solve_all()

    assert res["s"].to_unnamed() == {(i, j) for i, j in dataset if i != j}


def test_neurolange_dl_probabilistic_negation():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    s[x, y, prob(x, y)] = ~r(x, y) & q(x, y)

    result = neurolang.solve_all()["s"].to_unnamed()
    expected = {(i, j, 1 - p) for (p, i, j) in dataset}
    assert_almost_equal(result, expected)


@pytest.mark.skip
def test_neurolange_dl_probabilistic_negation_not_safe():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    z = neurolang.new_symbol(name="z")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    s[x, y, prob(x, y)] = ~r(x, z) & q(x, y)

    with pytest.raises(NegativeFormulaNotSafeRangeException):
        neurolang.solve_all()


@pytest.mark.skip
def test_neurolange_dl_probabilistic_negation_rule():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    t = neurolang.new_symbol(name="t")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    z = neurolang.new_symbol(name="z")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    t[x, y] = r(x, y) & q(y, z)
    s[x, y, prob(x, y)] = ~t(x, x) & q(x, y)

    with pytest.raises(NegativeFormulaNotNamedRelationException):
        neurolang.solve_all()


def test_neurolang_dl_aggregation():
    neurolang = NeurolangPDL()
    q = neurolang.new_symbol(name="q")
    p = neurolang.new_symbol(name="p")
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    for i in range(10):
        q[i % 2, i] = True

    p[x, sum_(y)] = q[x, y]

    sol = neurolang.query(r(x, y), p(x, y))

    res_q = {(0, 2 + 4 + 8), (1, 1 + 5 + 9)}

    assert len(sol) == 2
    assert sol[r] == res_q
    assert sol[p] == res_q


def test_post_probabilistic_aggregation():
    nl = NeurolangPDL()
    A = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.9, "b"), (0.5, "c")],
        name="A",
    )
    B = nl.add_probabilistic_choice_from_tuples(
        [(0.2, "a", "c"), (0.7, "b", "c"), (0.1, "a", "d")],
        name="B",
    )

    @nl.add_symbol
    def mysum(x):
        return sum(x)

    with nl.scope as e:
        e.C[e.x, e.y, e.PROB[e.x, e.y]] = A[e.x] & B[e.x, e.y]
        e.D[e.x, e.mysum(e.p)] = e.C[e.x, e.y, e.p]
        res = nl.query((e.x, e.s), e.D[e.x, e.s])

    assert len(res) == 2
    expected = {("a", 0.2 * 0.2 + 0.2 * 0.1), ("b", 0.9 * 0.7)}
    assert_almost_equal(res.to_unnamed(), expected)



@pytest.mark.parametrize(
    "query, columns, rows",
    [
        ("ans(count(s)) :- pain(s)", ("count",), {(2,)}),
        (
            "ans(t, count(s)) :- term(s, t)", ("t", "count"),
            {("pain", 2), ("fear", 1)},
        ),
    ],
)
def test_aggregate_query_on_a_rule(query, columns, rows):
    """
    An aggregate in the head of a query on one rule-defined predicate.

    It used to raise AttributeError: 'FunctionApplication' object has no
    attribute 'name', the aggregate being taken for a variable.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1", "pain"), ("s2", "pain"), ("s2", "fear")], name="tfidf"
    )
    nl.execute_datalog_program(
        "pain(s) :- tfidf(s, 'pain')\nterm(s, t) :- tfidf(s, t)"
    )
    res = nl.execute_datalog_program(query)
    assert res.columns == columns
    result = res.as_pandas_dataframe().itertuples(index=False, name=None)
    assert set(result) == rows

def test_empty_result_query():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("f",), ("d",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "a"),
            (0.7, "b"),
            (0.6, "c"),
        ],
        name="B",
    )
    with nl.scope as e:
        e.Q[e.x, e.PROB[e.x]] = A[e.x] & B[e.x]
        res = nl.query((e.x, e.p), e.Q[e.x, e.p])
    assert res.is_empty()
    with nl.scope as e:
        e.Q[e.x] = A[e.x] & A["b"]
        res = nl.query((e.x,), e.Q[e.x])
    assert res.is_empty()


def test_forbidden_query_on_probabilistic_predicate():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("f",), ("d",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.7, "b"), (0.6, "c")], name="B"
    )
    with pytest.raises(UnsupportedQueryError):
        with nl.scope as e:
            e.Q[e.x] = A[e.x] & B[e.x] & A["b"]
            nl.query((e.x,), e.Q[e.x])


def test_empty_boolean_query_result():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("a",), ("b",), ("c",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [(0.4, "a"), (0.5, "b")], name="B"
    )
    C = nl.add_probabilistic_choice_from_tuples(
        [(0.8, "c"), (0.2, "f")], name="C"
    )
    with nl.scope as e:
        e.D[e.x, e.PROB[e.x]] = A[e.x] & B[e.x] & C[e.x]
        res = nl.query(tuple(), e.D[e.x, e.p])
    assert not res


def test_trivial_probability_query_result():
    nl = NeurolangPDL()
    b = set(((0.4, "a"), (0.5, "b")))
    B = nl.add_probabilistic_facts_from_tuples(b, name='B')
    with nl.scope as e:
        e.D[e.x, e.PROB[e.x]] = B[e.x]
        res = nl.query((e.p, e.x), e.D[e.x, e.p])
    assert set(res) == b



def _mentions_engine():
    """Return an engine where studies mention terms, a study chosen."""
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1", "pain"), ("s1", "fear"), ("s2", "pain")], name="mentions"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="selected_study"
    )
    return nl


def _rows(result):
    return {
        (t, round(p, 4))
        for t, p in result.as_pandas_dataframe().itertuples(index=False)
    }


def test_datalog_query_with_probability_in_its_head():
    """``ans(t, PROB(t)) :- …`` used to return an empty result."""
    nl = _mentions_engine()
    res = nl.execute_datalog_program(
        "ans(t, PROB(t)) :- mentions(s, t), selected_study(s)"
    )
    assert res.columns == ("t", "PROB")
    assert _rows(res) == {("pain", 1.0), ("fear", 0.5)}


def test_datalog_conditional_query_with_probability_in_its_head():
    """``ans(t, PROB(t)) :- … // …`` used to raise AttributeError."""
    nl = _mentions_engine()
    res = nl.execute_datalog_program(
        "ans(t, PROB(t)) :- (mentions(s, t), selected_study(s)) // "
        "(selected_study(s), mentions(s, 'fear'))"
    )
    assert _rows(res) == {("pain", 1.0), ("fear", 1.0)}




@pytest.mark.parametrize(
    "query",
    [
        "ans(t, PROB) :- mentions(s, t), selected_study(s)",
        "ans(t, PROB(t)) :- mentions(s, t), selected_study(s)",
    ],
)
def test_datalog_query_with_implicit_probability_in_its_head(query):
    """``PROB`` alone in the head stands for ``PROB`` of the other terms."""
    nl = _mentions_engine()
    res = nl.execute_datalog_program(query)
    assert res.columns == ("t", "PROB")
    assert _rows(res) == {("pain", 1.0), ("fear", 0.5)}

@pytest.mark.parametrize(
    "option", [{"dry_run": True}, {"show_rewritten": True}, {"show_ra": True}]
)
def test_query_with_probability_in_its_head_printed(option, capsys):
    nl = _mentions_engine()
    nl.execute_datalog_program(
        "ans(t, PROB(t)) :- mentions(s, t), selected_study(s)", **option
    )
    assert "mentions(s, t)" in capsys.readouterr().out

def test_query_with_probability_in_its_head_leaves_program_unchanged():
    nl = _mentions_engine()
    before = set(nl.symbol_table.symbols_by_type(object))
    nl.execute_datalog_program(
        "ans(t, PROB(t)) :- mentions(s, t), selected_study(s)"
    )
    assert set(nl.symbol_table.symbols_by_type(object)) == before

def test_equality():
    nl = NeurolangPDL()
    r1 = nl.add_tuple_set([(i,) for i in range(5)], name="r1")

    with nl.scope as e:
        e.r2[e.x] = r1(e.y) & (e.x == e.y * 2)

        sol = nl.query((e.x,), e.r2[e.x])

    assert set(sol) == set((2 * i,) for i in range(5))

    r2 = nl.add_tuple_set([("Hola",), ("Hello",), ("Bonjour",)], name="r2")

    @nl.add_symbol
    def lower(input: str) -> str:
        return input.lower()

    with nl.scope as e:
        e.r3[e.x] = r2(e.y) & (e.x == lower(e.y))

        sol = nl.query((e.x,), e.r3[e.x])

    assert set(sol) == set((("hola",), ("hello",), ("bonjour",)))


def test_equality2():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("Hola", "var"), ("Hello", "var2"), ("Bonjour", "var")],
        name="test_var",
    )

    @nl.add_symbol
    def word_lower(word: str) -> str:
        return str(word).lower()

    with nl.scope as e:
        e.low[e.lower_] = e.test_var[e.word, "var"] & (
            e.lower_ == word_lower(e.word)
        )

        query = nl.query((e.lower_,), e.low[e.lower_])

    assert set(query) == set((("hola",), ("bonjour",)))


def test_result_both_deterministic_and_post_probabilistic():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("this", "is", "inglese"),
            ("questo", "è", "italiano"),
            ("ceci", "est", "francese"),
        ],
        name="parole",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.4, "valentino"),
            (0.6, "pietro"),
        ],
        name="persona_scelta",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.1, "italiano"),
            (0.7, "inglese"),
            (0.2, "francese"),
        ],
        name="lingua_parlata",
    )
    with nl.scope as e:
        e.e_una_lingua[e.lingua] = e.parole[e.subject, e.verb, e.lingua]
        e.lingua_da_lua_decisa[
            e.persona, e.lingua, e.PROB[e.persona, e.lingua]
        ] = (
            e.e_una_lingua[e.lingua]
            & e.persona_scelta[e.persona]
            & e.lingua_parlata[e.lingua]
        )
        e.utilizzare_le_probabilita[e.lingua, e.p] = e.lingua_da_lua_decisa[
            ..., e.lingua, e.p
        ] & (e.p > 0.1)
        res = nl.solve_all()
    assert "e_una_lingua" in res
    assert "lingua_da_lua_decisa" in res
    assert "utilizzare_le_probabilita" in res
    assert len(res["utilizzare_le_probabilita"]) == 3
    expected = RelationalAlgebraFrozenSet(
        [
            ("francese", 0.12),
            ("inglese", 0.7 * 0.4),
            ("inglese", 0.7 * 0.6),
        ]
    )
    assert_almost_equal(res["utilizzare_le_probabilita"], expected)


def test_result_query_relation_correct_column_names():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with nl.environment as e:
        e.climbs[e.p, e.PROB[e.p, e.city], e.city] = (
            e.person[e.p] & e.lives_in[e.p, e.city] & e.does_not_smoke[e.p]
        )
        solution = nl.solve_all()
    assert all(name in solution for name in ["climbs", "person", "lives_in"])


def test_add_constraints_and_rewrite():
    nl = NeurolangPDL()

    nl.add_tuple_set(
        [
            ("neurolang"),
        ],
        name="project",
    )

    nl.add_tuple_set(
        [
            ("neurolang", "db"),
            ("neurolang", "logic"),
        ],
        name="inArea",
    )

    with nl.scope as e:
        nl.add_constraint(
            e.project[e.x] & e.inArea[e.x, e.y],
            e.hasCollaborator[e.z, e.y, e.x],
        )

        e.p[e.b] = e.hasCollaborator[e.a, "db", e.b]
        res = nl.solve_all()

    assert list(res["p"])[0] == ("neurolang",)


def test_solve_marg_query():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.2, "alice", "running"),
            (0.8, "bob", "climbing"),
        ],
        name="practice",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with nl.environment as e:
        e.query[e.p, e.PROB[e.p, e.city, e.sport], e.city, e.sport] = (
            e.person[e.p] & e.lives_in[e.p, e.city] & e.does_not_smoke[e.p]
        ) // e.practice[e.p, e.sport]
        solution = nl.solve_all()
    assert all(name in solution for name in ["query", "person", "lives_in"])
    with nl.environment as e:
        result = nl.query(
            (e.p, e.prob, e.city, e.sport),
            e.query(e.p, e.prob, e.city, e.sport),
        )
    expected = RelationalAlgebraFrozenSet(
        iterable=[
            ("alice", 0.9, "paris", "running"),
            ("bob", 0.8, "marseille", "climbing"),
        ],
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (2, 0.2),
            (7, 0.8),
            (4, 0.4),
        ],
        name="A",
    )
    with nl.environment as e:
        (e.B @ (e.p / 2))[e.x] = e.A[e.x, e.p]
        e.Query[e.PROB[e.x], e.x] = e.B[e.x]
        result = nl.query((e.x, e.p), e.Query[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        [
            (2, 0.1),
            (7, 0.4),
            (4, 0.2),
        ]
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact_empty():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (2, 0.2),
            (7, 0.8),
            (4, 0.4),
        ],
        name="A",
    )
    with nl.environment as e:
        (e.B @ (e.p / 2))[e.x] = e.A[e.x, e.p] & (e.p > 0.8)
        e.Query[e.PROB[e.x], e.x] = e.B[e.x]
        result = nl.query((e.x, e.p), e.Query[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        []
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact_region_volume():
    nl = NeurolangPDL()

    @nl.add_symbol
    def volume(s: SphericalVolume) -> float:
        return (4 / 3) * np.pi * s.radius ** 3

    assert volume.symbol_name in nl.functions

    nl.add_tuple_set(
        [
            ("contained", SphericalVolume((0, 0, 0), 1)),
            ("container", SphericalVolume((0, 0, 0), 2)),
        ],
        name="my_sphere",
    )

    with nl.environment as e:
        (e.Z @ (e.volume[e.contained] / e.volume[e.container]))[
            e.contained, e.container
        ] = (
            e.my_sphere["contained", e.contained]
            & e.my_sphere["container", e.container]
        )
        e.Query[
            e.contained, e.container, e.PROB[e.contained, e.container]
        ] = e.Z[e.contained, e.container]
        res = nl.query((e.p,), e.Query[e.contained, e.container, e.p])
    expected = NamedRelationalAlgebraFrozenSet(("p",), [(1 / 2 ** 3,)])
    assert_almost_equal(res, expected)


def test_solve_marg_query_disjunction():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.2, "alice", "running"),
            (0.8, "bob", "climbing"),
        ],
        name="practice",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with pytest.raises(
        (
            ForbiddenConditionalQueryNonConjunctive,
            Fol2DatalogTranslationException,
        )
    ):
        with nl.environment as e:
            e.query[e.p, e.PROB[e.p, e.city, e.sport], e.city, e.sport] = (
                e.person[e.p]
                & (e.lives_in[e.p, e.city] | e.does_not_smoke[e.p])
            ) // e.practice[e.p, e.sport]


def test_query_without_safe_plan():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice"),
            (0.8, "bob"),
        ],
        name="names",
    )

    expected = NamedRelationalAlgebraFrozenSet(
        columns=('x', 'y', 'PROB'),
        iterable={
            ('alice', 'alice', 0.2),
            ('bob', 'bob', 0.8),
            ('alice', 'bob', 0.8 * 0.2),
            ('bob', 'alice', 0.8 * 0.2)
        }
    )

    with nl.environment as e:
        e.q[e.x, e.y, e.PROB[e.x, e.y]] = e.names[e.x] & e.names[e.y]

    res = nl.solve_all()['q']
    assert res == expected


def test_query_without_safe_plan_2():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice", 1),
            (0.4, "alice", 2),
            (0.8, "bob", 2),
            (0.6, "bob", 1),
        ],
        name="names",
    )

    expected = NamedRelationalAlgebraFrozenSet(
        columns=('x', 'y', 'PROB'),
        iterable={
            ('alice', 'alice', 0.52),
            ('bob', 'bob', 0.92),
            ('alice', 'bob', 0.40160000000000007),
            ('bob', 'alice', 0.40160000000000007)
        }
    )

    with nl.environment as e:
        e.q[e.x, e.y, e.PROB[e.x, e.y]] = e.names[e.x, e.s] & e.names[e.y, e.s]

    res = nl.solve_all()['q']
    assert res == expected


def test_query_without_safe_fails():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice"),
            (0.8, "bob"),
        ],
        name="names",
    )

    with nl.environment as e:
        e.q[e.x, e.y, e.PROB[e.x, e.y]] = e.names[e.x] & e.names[e.y]

    with pytest.raises(dalvi_suciu_lift.NonLiftableException):
        nl.solve_all()


def test_query_self_join_matrix_query_ds():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice", 1),
            (0.8, "bob", 1),
            (0.4, "sam", 2),
            (0.3, "alice", 1)
        ],
        name="names",
    )

    with nl.environment as e:
        e.q[e.PROB()] = e.names[e.x, e.z] & e.names[e.y, e.z]

    res = nl.solve_all()['q']

    assert len(res) == 1
    assert res.arity == 1
    assert list(res)[0].PROB == 0.9328


def test_cbma_two_term_conjunctive_query():
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [
            ("s1",),
            ("s2",),
            ("s3",),
        ],
        name="SelectedStudy",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.1, "t1", "s1"),
            (0.2, "t2", "s1"),
            (0.3, "t1", "s2"),
            (0.4, "t2", "s2"),
            (0.5, "t1", "s3"),
            (0.6, "t2", "s3"),
        ],
        name="TermInStudy",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.6, "v1", "s1"),
            (0.5, "v2", "s1"),
            (0.4, "v1", "s2"),
            (0.3, "v2", "s2"),
            (0.2, "v1", "s3"),
            (0.1, "v2", "s3"),
        ],
        name="VoxelReported",
    )
    with nl.environment as e:
        e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
        e.Activation[e.v] = e.SelectedStudy[e.s] & e.VoxelReported[e.v, e.s]
        e.probmap[e.v, e.PROB[e.v]] = (e.Activation[e.v]) // (
            e.TermAssociation["t1"] & e.TermAssociation["t2"]
        )
        res = nl.query((e.v, e.p), e.probmap[e.v, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            (
                "v1",
                (0.6 * 0.1 * 0.2 + 0.4 * 0.3 * 0.4 + 0.2 * 0.5 * 0.6)
                / (0.1 * 0.2 + 0.3 * 0.4 + 0.5 * 0.6),
            ),
            (
                "v2",
                (0.5 * 0.1 * 0.2 + 0.3 * 0.4 * 0.3 + 0.1 * 0.5 * 0.6)
                / (0.1 * 0.2 + 0.3 * 0.4 + 0.5 * 0.6),
            ),
        ]
    )
    assert_almost_equal(res, expected)


def test_cbma_prob_query_with_negation():
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [
            ("s1",),
            ("s2",),
            ("s3",),
        ],
        name="SelectedStudy",
    )
    nl.add_tuple_set([("nA",), ("nB",)], name="Network")
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.1, "t1", "s1"),
            (0.2, "t2", "s1"),
            (0.3, "t1", "s2"),
            (0.4, "t2", "s2"),
            (0.5, "t1", "s3"),
            (0.6, "t2", "s3"),
        ],
        name="TermInStudy",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.6, "nA", "s1"),
            (0.5, "nB", "s1"),
            (0.4, "nA", "s2"),
            (0.3, "nB", "s2"),
            (0.2, "nA", "s3"),
            (0.1, "nB", "s3"),
        ],
        name="NetworkReported",
    )
    with nl.environment as e:
        e.ProbTermAssociation[e.t, e.n, e.PROB[e.t, e.n]] = (
            e.TermInStudy[e.t, e.s] & e.SelectedStudy[e.s]
        ) // (
            ~e.NetworkReported[e.n, e.s]
            & e.Network[e.n]
            & e.SelectedStudy[e.s]
        )
        res = nl.query((e.t, e.n, e.p), e.ProbTermAssociation[e.t, e.n, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            ("t1", "nA", 0.344444),
            ("t1", "nB", 0.338095),
            ("t2", "nA", 0.444444),
            ("t2", "nB", 0.438095),
        ]
    )
    assert_almost_equal(res, expected)


def test_negated_conditioning_literal_with_distinguished_variable():
    """
    Regression test for negating a distinguished-variable literal in `//`.

    A conditional-probability query negating a (deterministic)
    relation whose argument is a free variable shared with the
    query's head returned an incorrect probability.

    `TranslateProbabilisticQueryMixin.rewrite_conditional_query` builds the
    numerator of `//` by flatly conjoining the conditioned and conditioning
    sides into one within-language query. That numerator query was wrong
    whenever it contained a negated literal with a distinguished-variable
    argument (here, `r`) -- in fact wrong for ANY negated deterministic
    literal in a probabilistic query, distinguished or not, `//`-wrapped
    or a bare SUCC query. The actual defect was in
    `small_dichotomy_theorem_based_solver`'s hierarchical-query safety
    check: it only ever inspected the probabilistic-only sub-conjunction
    of the query, so a `Negation` wrapping a plain deterministic relation
    like `Active` (joined with the probabilistic choice `Selected` via a
    shared variable) silently slipped past it instead of triggering the
    `NotHierarchicalQueryException` fallback to the (correct) Dalvi-Suciu
    lifted solver. The fix widens that check
    (`_extract_deterministic_negated_atoms`) to also count negated
    deterministic atoms.

    Fixture: 6 studies, 2 regions (A = {0,1,2}, B = {3,4,5}), 2 terms (pain
    mentioned by {0,1,5}, joy by {2,3,4}). Hand-computed:
    P(pain | not active in A) = P(mentions pain & study in {3,4,5}) / 3
                               = 1/3 (only study 5 qualifies)
    and symmetrically for the other three (term, region) pairs.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(s,) for s in range(6)], name="TrainStudy")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="Selected"
    )
    nl.add_tuple_set(
        [(0, "A"), (1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "B")],
        name="Active",
    )
    nl.add_tuple_set(
        [(0, "pain"), (1, "pain"), (2, "joy"), (3, "joy"), (4, "joy"), (5, "pain")],
        name="Mentions",
    )
    nl.add_tuple_set([("A",), ("B",)], name="Region")

    with nl.environment as e:
        e.PTermGivenNotRegion[e.t, e.r, e.PROB[e.t, e.r]] = (
            e.Mentions[e.s, e.t] & e.Selected[e.s]
        ) // (
            ~e.Active[e.s, e.r]
            & e.Region[e.r]
            & e.TrainStudy[e.s]
            & e.Selected[e.s]
        )
        res = nl.query(
            (e.t, e.r, e.p), e.PTermGivenNotRegion[e.t, e.r, e.p]
        )
    expected = RelationalAlgebraFrozenSet(
        [
            ("pain", "A", 1 / 3),
            ("joy", "A", 2 / 3),
            ("pain", "B", 2 / 3),
            ("joy", "B", 1 / 3),
        ]
    )
    assert_almost_equal(res, expected)


def test_negated_conditioning_literal_with_two_distinguished_variables():
    """
    Regression test for a negated literal with two distinguished variables.

    The fix must correctly handle a negated deterministic literal
    regardless of how many of its arguments are distinguished (shared
    with the query head) -- here `r1` AND `r2`, both from the single
    negated literal `~CoActive[s, r1, r2]`. The solver-level fix
    (`small_dichotomy_theorem_based_solver._extract_deterministic_negated_atoms`)
    never special-cases variable count or distinguished-ness at all --
    it forces ANY negated deterministic atom to fall back to the
    (correct) Dalvi-Suciu lifted solver -- so this is mostly a
    confirmation that nothing about multiple shared variables breaks
    that fallback path.

    Fixture: 6 studies, regions {A, B, C}; `CoActive(s, r1, r2)` holds
    (symmetrically) for study 0 in {A, B} and study 3 in {B, C}. Terms:
    pain mentioned by {0,1,5}, joy by {2,3,4}. Hand-computed:
    P(pain | not co-active in (A, B)) = P(mentions pain & study != 0) / 5
                                        = 2/5 (studies {1,2,3,4,5}, pain
                                          in {1,5})
    P(joy | not co-active in (B, C)) = P(mentions joy & study != 3) / 5
                                        = 2/5 (studies {0,1,2,4,5}, joy
                                          in {2,4})
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(s,) for s in range(6)], name="TrainStudy")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="Selected"
    )
    nl.add_tuple_set(
        [(0, "A", "B"), (0, "B", "A"), (3, "B", "C"), (3, "C", "B")],
        name="CoActive",
    )
    nl.add_tuple_set(
        [(0, "pain"), (1, "pain"), (2, "joy"), (3, "joy"), (4, "joy"), (5, "pain")],
        name="Mentions",
    )
    nl.add_tuple_set([("A",), ("B",), ("C",)], name="Region")

    with nl.environment as e:
        e.PTermGivenNotCoActive[
            e.t, e.r1, e.r2, e.PROB[e.t, e.r1, e.r2]
        ] = (e.Mentions[e.s, e.t] & e.Selected[e.s]) // (
            ~e.CoActive[e.s, e.r1, e.r2]
            & e.Region[e.r1]
            & e.Region[e.r2]
            & e.TrainStudy[e.s]
            & e.Selected[e.s]
        )
        res = nl.query(
            (e.t, e.r1, e.r2, e.p),
            e.PTermGivenNotCoActive[e.t, e.r1, e.r2, e.p],
        )
    res_df = res.as_pandas_dataframe().set_index(["t", "r1", "r2"])
    np.testing.assert_almost_equal(res_df.loc[("pain", "A", "B"), "p"], 2 / 5)
    np.testing.assert_almost_equal(res_df.loc[("joy", "B", "C"), "p"], 2 / 5)
    np.testing.assert_almost_equal(res_df.loc[("pain", "A", "A"), "p"], 1 / 2)


def test_query_based_spatial_prior():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (5, 5, 5, "s1"),
            (7, 5, 5, "s1"),
            (5, 5, 5, "s2"),
            (10, 10, 10, "s2"),
            (10, 10, 11, "s2"),
        ],
        name="FocusReported",
    )
    nl.add_tuple_set(
        list(itertools.product(range(15), range(15), range(15))), name="Voxel"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="SelectedStudy"
    )
    exp = nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    with nl.environment as e:
        (e.VoxelReported @ e.max(exp(-(e.d ** 2) / 5.0)))[
            e.i1, e.j1, e.k1, e.s
        ] = (
            e.FocusReported(e.i2, e.j2, e.k2, e.s)
            & e.Voxel(e.i1, e.j1, e.k1)
            & (e.d == e.EUCLIDEAN(e.i1, e.j1, e.k1, e.i2, e.j2, e.k2))
            & (e.d <= 1)
        )
        e.Activation[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = e.VoxelReported(
            e.i, e.j, e.k, e.s
        ) & e.SelectedStudy(e.s)
        result = nl.query(
            (e.i, e.j, e.k, e.p), e.Activation(e.i, e.j, e.k, e.p)
        )
    expected = RelationalAlgebraFrozenSet(
        [
            (4, 5, 5, np.exp(-1 / 5)),
            (5, 4, 5, np.exp(-1 / 5)),
            (5, 5, 4, np.exp(-1 / 5)),
            (5, 5, 5, 1.0),
            (5, 5, 6, np.exp(-1 / 5)),
            (5, 6, 5, np.exp(-1 / 5)),
            (6, 5, 5, np.exp(-1 / 5)),
            (5, 6, 5, np.exp(-1 / 5)),
            (6, 5, 5, np.exp(-1 / 5)),
            (7, 4, 5, np.exp(-1 / 5) / 2),
            (7, 5, 4, np.exp(-1 / 5) / 2),
            (7, 5, 5, 1 / 2),
            (7, 5, 6, np.exp(-1 / 5) / 2),
            (7, 6, 5, np.exp(-1 / 5) / 2),
            (8, 5, 5, np.exp(-1 / 5) / 2),
            (9, 10, 10, np.exp(-1 / 5) / 2),
            (9, 10, 11, np.exp(-1 / 5) / 2),
            (10, 9, 10, np.exp(-1 / 5) / 2),
            (10, 9, 11, np.exp(-1 / 5) / 2),
            (10, 10, 9, np.exp(-1 / 5) / 2),
            (10, 10, 10, 0.5),
            (10, 10, 11, 0.5),
            (10, 10, 12, np.exp(-1 / 5) / 2),
            (10, 11, 10, np.exp(-1 / 5) / 2),
            (10, 11, 11, np.exp(-1 / 5) / 2),
            (11, 10, 10, np.exp(-1 / 5) / 2),
            (11, 10, 11, np.exp(-1 / 5) / 2),
        ]
    )
    assert_almost_equal(result, expected)


def test_simple_sigmoid():
    nl = NeurolangPDL()
    unnormalised = nl.add_tuple_set(
        [
            ("a", 1),
            ("b", 2),
            ("c", 7),
        ],
        name="unnormalised",
    )
    exp = nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    with nl.environment as e:
        (e.prob @ (1 / (1 + exp(-e.x))))[e.l] = unnormalised[e.l, e.x]
        e.Query[e.l, e.PROB[e.l]] = e.prob[e.l]
        res = nl.query((e.l, e.p), e.Query[e.l, e.p])
    expected = {(l, (1 / (1 + np.exp(-x)))) for l, x in unnormalised.value}
    assert_almost_equal(res, expected)


def test_postprob_conjunct_with_wlq_result():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1",), ("s2",)],
        name="S",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)],
        name="SS",
    )
    nl.add_tuple_set(
        [("t1", "s1"), ("t1", "s2"), ("t2", "s2")],
        name="TIS",
    )

    with nl.environment as e:
        e.TheWLQ[e.t, e.PROB(e.t)] = e.SS(e.s) & e.TIS(e.t, e.s)
        e.P[e.t, e.count(e.s)] = e.TIS(e.t, e.s)
        e.Q[e.count(e.s)] = e.S(e.s)
        e.TheQuery[e.t, e.N, e.m, e.p] = (
            e.TheWLQ(e.t, e.p) & e.P(e.t, e.m) & e.Q(e.N)
        )
        sol = nl.query((e.t, e.N, e.m, e.p), e.TheQuery(e.t, e.N, e.m, e.p))
    expected = {
        ("t1", 2, 2, 1.0),
        ("t2", 2, 1, 0.5),
    }
    assert_almost_equal(sol, expected)


def test_no_tuple_unicity_qbased_pfact():
    nl = NeurolangPDL(check_qbased_pfact_tuple_unicity=True)
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.scope as e:
        (e.Q @ e.p)[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        with pytest.raises(RepeatedTuplesInProbabilisticRelationError):
            nl.query((e.x, e.p), e.Query(e.x, e.p))
    nl = NeurolangPDL(check_qbased_pfact_tuple_unicity=False)
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.scope as e:
        (e.Q @ e.p)[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        try:
            nl.query((e.x, e.p), e.Query(e.x, e.p))
        except NeuroLangException:
            pytest.fail(
                "Expected this test not to raise an exception as "
                "the tuple unicity check on query-based probabilistic tables "
                "is disabled"
            )


def test_qbased_pfact_max_prob():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.environment as e:
        (e.Q @ e.max(e.p))[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {("a", 0.9), ("b", 0.5)}
    assert_almost_equal(sol, expected)


def test_qbased_pchoice():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.3, "c")], name="P")
    nl.add_tuple_set([(1, "a"), (1, "c"), (2, "b")], name="R")
    with nl.environment as e:
        (e.Q ^ e.p)[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x, e.y) & e.Q(e.y)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {(1, 0.5), (2, 0.5)}
    assert_almost_equal(sol, expected)


def test_qbased_pchoice_equiprobable():
    nl = NeurolangPDL()
    nl.add_tuple_set([("a",), ("b",), ("c",)], name="P")
    nl.add_tuple_set([(1, "a"), (1, "c"), (2, "b")], name="R")
    with nl.environment as e:
        e.S[e.count(e.x)] = e.P(e.x)
        (e.Q ^ (1. / e.p))[e.x] = e.P(e.x) & e.S(e.p)
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x, e.y) & e.Q(e.y)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {(1, 2 / 3.), (2, 1 / 3.)}
    assert_almost_equal(sol, expected)


def test_qbased_pchoice_unused_by_the_query():
    nl = NeurolangPDL()
    nl.add_tuple_set([("a",), ("b",), ("c",)], name="P")
    nl.add_tuple_set([(1,), (3,)], name="R")
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,)], name="Selected"
    )
    with nl.environment as e:
        e.S[e.count(e.x)] = e.P(e.x)
        (e.Q ^ (1. / e.p))[e.x] = e.P(e.x) & e.S(e.p)
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x) & e.Selected(e.x)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {(1, 0.5)}
    assert_almost_equal(sol, expected)


def test_qbased_pchoice_empty():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0.2, "a"), (0.8, "b")], name="P")
    nl.add_tuple_set([(1, "a"), (2, "b")], name="R")
    with nl.environment as e:
        (e.Q ^ e.p)[e.x] = e.P(e.p, e.x) & (e.p > 0.8)
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x, e.y) & e.Q(e.y)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    assert_almost_equal(sol, RelationalAlgebraFrozenSet([]))


def test_noisy_or_probabilistic_query():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "a", "b"),
            (0.7, "a", "c"),
            (0.9, "b", "c"),
        ],
        name="R",
    )
    with nl.scope as e:
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x, e.y)
        sol = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 1 - (1 - 0.2) * (1 - 0.7)),
        ("b", 1 - (1 - 0.9)),
    }
    assert_almost_equal(sol, expected)


def test_disjunctive_probfact_rule():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.7, "a"),
            (0.1, "b"),
            (0.8, "c"),
        ],
        name="P",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.4, "a"),
            (0.8, "b"),
        ],
        name="Q",
    )
    with nl.scope as e:
        e.Z[e.x] = e.P(e.x)
        e.Z[e.x] = e.Q(e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Z(e.x)
        sol = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 1 - (1 - 0.7) * (1 - 0.4)),
        ("b", 1 - (1 - 0.1) * (1 - 0.8)),
        ("c", 0.8),
    }
    assert_almost_equal(sol, expected)


def test_probchoice_disjunction():
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.7, "a"),
            (0.3, "b"),
        ],
        name="P",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "a"),
            (0.9, "c"),
        ],
        name="Q",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "d", "c"),
            (0.1, "b", "a"),
            (0.4, "b", "b"),
        ],
        name="R",
    )
    with nl.scope as e:
        e.Z[e.x] = e.P(e.x) & e.Q(e.x)
        e.Z[e.x] = e.P(e.x) & e.R(e.x, e.y)
        e.Query[e.x, e.PROB(e.x)] = e.Z(e.x)
        sol = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 0.7 * 0.8),
        ("b", 0.3 * (1 - (1 - 0.1) * (1 - 0.4))),
    }
    assert_almost_equal(sol, expected)


def test_probchoice_disjunction_probfact_probchoice():
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.7, "a"),
            (0.3, "b"),
        ],
        name="P",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "a"),
            (0.9, "c"),
        ],
        name="Q",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "d", "c"),
            (0.1, "b", "a"),
            (0.4, "b", "b"),
        ],
        name="R",
    )
    with nl.scope as e:
        e.Z[e.x] = e.P(e.x) & e.Q(e.x)
        e.Z[e.x] = e.R(e.x, e.y)
        e.Query[e.x, e.PROB(e.x)] = e.Z(e.x)
        sol = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 0.7 * 0.8),
        ("b", 1 - (1 - 0.1) * (1 - 0.4)),
        ("d", 0.2),
    }
    assert_almost_equal(sol, expected)


def test_simple_disjunction_probfacts():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.7, "a"),
            (0.3, "b"),
        ],
        name="P",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "a"),
            (0.9, "c"),
        ],
        name="Q",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "d"),
            (0.1, "b"),
        ],
        name="R",
    )
    with nl.scope as e:
        e.Z[e.x] = e.P(e.x) & e.Q(e.x)
        e.Z[e.x] = e.R(e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Z(e.x)
        sol = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 0.7 * 0.8),
        ("b", 0.1),
        ("d", 0.2),
    }
    assert_almost_equal(sol, expected)


def test_pchoice_with_both_eqvar_and_free_var():
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.7, "a", "b"),
            (0.2, "a", "c"),
            (0.1, "b", "b"),
        ],
        name="Pchoice",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.9, "a"),
            (0.2, "b"),
            (1.0, "c"),
        ],
        name="Pfact",
    )
    with nl.scope as e:
        e.Query[e.x, e.PROB(e.x)] = e.Pchoice(e.x, e.y) & e.Pfact(e.y)
        result = nl.query((e.x, e.prob), e.Query(e.x, e.prob))
    expected = {
        ("a", 0.7 * 0.2 + 0.2 * 1.0),
        ("b", 0.1 * 0.2),
    }
    assert_almost_equal(result, expected)


def test_current_program_with_probfact():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (10751455, "emotion", 0.052538),
            (10808134, "emotion", 0.244368),
            (10913505, "emotion", 0.059463),
        ],
        name="TermInStudyTFIDF",
    )
    exp = nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    with nl.environment as e:
        (e.TermInStudy @ (1 / (1 + exp(-300 * (e.tfidf - 0.001)))))[
            e.t, e.s
        ] = e.TermInStudyTFIDF(e.s, e.t, e.tfidf)
    prog = nl.current_program
    assert len(prog) == 2


def test_min_and_avg_aggregations():
    nl = NeurolangPDL()
    nl.add_tuple_set([("a", 3), ("a", 1), ("b", 2)], name="p")
    low = nl.execute_datalog_program(
        "low(x, v) :- AGGREGATE[x](p(x, y) @ min(y)) = v\n"
        "ans(x, v) :- low(x, v)"
    )
    assert set(low) == {("a", 1), ("b", 2)}
    average = nl.execute_datalog_program(
        "average(v) :- AGGREGATE[](p(x, y) @ avg(y)) = v\n"
        "ans(v) :- average(v)"
    )
    assert set(average) == {(2.0,)}


def test_constant_on_empty_deterministic_idb_in_probabilistic_rule():
    """
    Solve a probabilistic rule over an empty deterministic predicate.

    ``m`` has no tuple, and so no relation in the solution of the
    deterministic stratum: selecting ``m(s, "memory")`` crashed.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1", "memory", 0.5), ("s2", "pain", 0.6)], name="tfidf"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="selected_study"
    )
    nl.execute_datalog_program(
        "m(s, t) :- tfidf(s, t, v) & (v > 100)\n"
        "q(x, PROB(x)) :- m(s, 'memory'), selected_study(s), tfidf(s, x, v)"
    )
    with nl.scope as e:
        res = nl.query((e.x, e.p), e.q(e.x, e.p))
    assert len(res) == 0


@pytest.mark.parametrize("equality", ("(t == 't1')", "('t1' == t)"))
@pytest.mark.parametrize("negated", (False, True))
def test_equality_to_a_constant_within_an_existential(equality, negated):
    """
    An equality to a constant within ``exists(…)`` is not inlined.

    The equality rewrites looked for equalities in the whole body but only
    rewrote the top-level conjuncts, rewriting such a rule again and
    again until the recursion limit.
    """
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        [(0.4, "s1"), (0.6, "s2")], name="C"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(0.5, "s1", "t1"), (0.5, "s2", "t2")], name="M"
    )
    negation = "~" if negated else ""
    res = nl.execute_datalog_program(
        f"q(s, PROB) :- C(s), {negation}exists(t; M(s, t), {equality})\n"
        "ans(s, p) :- q(s, p)"
    )
    probabilities = {
        s: p for s, p in res.as_pandas_dataframe().itertuples(index=False)
        if p > 0
    }
    expected = {"s1": 0.2, "s2": 0.6} if negated else {"s1": 0.2}
    assert probabilities.keys() == expected.keys()
    for s, p in expected.items():
        assert probabilities[s] == pytest.approx(p)


def test_forward_inference_given_a_single_term_is_lifted():
    """The probability that the selected study reports a peak near a
    voxel, given that it mentions t and no other term, is lifted: the
    choice within the negation is absorbed by the one outside it.
    """
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_tuple_set(
        [(0, 0, 0, "s0"), (3, 3, 3, "s2"), (0, 0, 1, "s3")],
        name="peak_reported",
    )
    nl.add_tuple_set([(0, 0, 0), (3, 3, 3)], name="voxel")
    nl.add_tuple_set(
        [
            ("s0", "pain"), ("s1", "pain"), ("s1", "joy"),
            ("s2", "joy"), ("s3", "joy"),
        ],
        name="mentions",
    )
    nl.add_tuple_set([("pain",), ("joy",)], name="term")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s0",), ("s1",), ("s2",), ("s3",)], name="selected_study"
    )

    @nl.add_symbol
    def euclidean(
        i: int, j: int, k: int, i1: int, j1: int, k1: int
    ) -> float:
        return float(np.sqrt((i - i1) ** 2 + (j - j1) ** 2 + (k - k1) ** 2))

    nl.execute_datalog_program(
        "near_peak(i, j, k, s) :- voxel(i, j, k) & "
        "peak_reported(i1, j1, k1, s), "
        "(euclidean(i, j, k, i1, j1, k1) <= 2)\n"
        "activation(i, j, k, s) :- near_peak(i, j, k, s), "
        "selected_study(s)\n"
        "term_association(s, t) :- mentions(s, t), selected_study(s)\n"
        "forward(i, j, k, t, PROB) :- activation(i, j, k, s) // "
        "(term_association(s, t) & ~exists(t1; term(t1), "
        "term_association(s, t1), (t1 != t)))\n"
    )
    res = nl.solve_all()["forward"].as_pandas_dataframe()

    assert set(res.itertuples(index=False, name=None)) == {
        (0, 0, 0, "pain", 1.0),
        (0, 0, 0, "joy", 0.5),
        (3, 3, 3, "joy", 0.5),
    }


@pytest.mark.parametrize("kind", ("facts", "choice"))
def test_argument_types_of_probabilistic_relations(kind):
    """The types of the arguments of a probabilistic relation are those of
    its columns after the probability column."""
    nl = NeurolangPDL()
    tuples = [(0.5, "a", "b"), (0.3, "b", "a"), (0.2, "a", "a")]
    if kind == "facts":
        nl.add_probabilistic_facts_from_tuples(tuples, name="R")
    else:
        nl.add_probabilistic_choice_from_tuples(tuples, name="R")
    nl.execute_datalog_program("q(x, PROB) :- R(x, y), R(y, x)")
    q = nl.program_ir.intensional_database()[Symbol("q")].formulas[0]
    assert all(
        arg.type is str
        for atom in q.antecedent.formulas for arg in atom.args
    ), q.antecedent
    result = nl.solve_all()["q"].as_pandas_dataframe()
    probabilities = {
        row[0]: row[1] for row in result.itertuples(index=False)
    }
    if kind == "facts":
        expected = {"a": 1 - (1 - 0.5 * 0.3) * (1 - 0.2), "b": 0.5 * 0.3}
    else:
        expected = {"a": 0.2}
    assert probabilities.keys() == expected.keys()
    for value, p in expected.items():
        assert probabilities[value] == pytest.approx(p)


@pytest.mark.parametrize("program, expected", (
    (
        "q(PROB) :- selected_study(s), "
        "~exists(s2; selected_study(s2), (s2 != s))",
        {(): 1.},
    ),
    (
        "q(t, PROB) :- mentions(s, t), selected_study(s), "
        "~exists(s2; mentions(s2, t), selected_study(s2), (s2 != s))",
        {("a",): (0.4 + 0.7) / 3},
    ),
), ids=("choice twice", "choice twice with facts"))
def test_weighted_model_counting_of_a_choice_occurring_twice(
    program, expected
):
    """A probabilistic choice is a single random variable: a selected
    study is the only one selected."""
    from nl_core.probabilistic import weighted_model_counting
    nl = NeurolangPDL(
        probabilistic_solvers=(weighted_model_counting.solve_succ_query,),
        probabilistic_marg_solvers=(weighted_model_counting.solve_marg_query,),
    )
    nl.add_probabilistic_facts_from_tuples(
        [(0.4, "s1", "a"), (0.7, "s2", "a")], name="mentions"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",), ("s3",)], name="selected_study"
    )
    nl.execute_datalog_program(program)
    result = nl.solve_all()["q"].as_pandas_dataframe()
    probabilities = {
        tuple(row[:-1]): row[-1]
        for row in result.itertuples(index=False)
    }
    assert probabilities.keys() == expected.keys()
    for answer, p in expected.items():
        assert probabilities[answer] == pytest.approx(p)


@pytest.mark.parametrize("query, expected", (
    ("ans(x, p) :- m(x, p), d(x)", {("a", .5), ("b", .5)}),
    ("ans(x, p) :- m(x, p), (x == 'a')", {("a", .5)}),
    ("ans(x, q) :- m(x, p), n(x, p1), (q == p1 / p)",
     {("a", .4 / .5), ("b", .1 / .5)}),
))
def test_conjunctive_query_on_probabilistic_queries(query, expected):
    """A query whose body is a conjunction is solved with the
    probabilistic queries it depends on."""
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set([("a",), ("b",)], name="c")
    nl.add_probabilistic_facts_from_tuples([(.4, "a"), (.1, "b")], name="f")
    nl.add_tuple_set([("a",), ("b",)], name="d")
    result = nl.execute_datalog_program(
        "m(x, PROB) :- c(x)\n"
        "n(x, PROB) :- f(x)\n" + query
    )

    rows = set(result.as_pandas_dataframe().itertuples(index=False))
    assert len(rows) == len(expected)
    for (x, p), (x_expected, p_expected) in zip(
        sorted(rows), sorted(expected)
    ):
        assert x == x_expected
        assert p == pytest.approx(p_expected)


def _engine_with_two_witnesses(selected, solvers=None):
    """`R` has two rows for x = 1, so ∃y R(1, y) has two witnesses."""
    if solvers is None:
        nl = NeurolangPDL()
    else:
        nl = NeurolangPDL(
            probabilistic_solvers=(solvers.solve_succ_query,),
            probabilistic_marg_solvers=(solvers.solve_marg_query,),
        )
    nl.add_tuple_set([(1, "a"), (1, "c"), (2, "b")], name="R")
    if selected == "facts":
        nl.add_probabilistic_facts_from_tuples(
            [(0.5, 1), (0.5, 2)], name="Selected"
        )
    else:
        nl.add_uniform_probabilistic_choice_over_set(
            [(1,), (2,)], name="Selected"
        )
    return nl


def _query_with_deterministic_existential(nl):
    with nl.environment as e:
        e.Query[e.x, e.PROB(e.x)] = e.R(e.x, e.y) & e.Selected(e.x)
        return nl.query((e.x, e.p), e.Query(e.x, e.p))


@pytest.mark.parametrize("selected", ["facts", "choice"])
@pytest.mark.parametrize(
    "solvers", [None, dalvi_suciu_lift], ids=["default", "lift"]
)
def test_existential_variable_of_deterministic_relation_only(
    selected, solvers
):
    """P(∃y R(1, y) ∧ Selected(1)) is P(Selected(1)) however many rows
    of the deterministic `R` witness ∃y: they are the same event."""
    nl = _engine_with_two_witnesses(selected, solvers)
    sol = _query_with_deterministic_existential(nl)
    assert len(sol) == 2
    assert_almost_equal(sol, {(1, 0.5), (2, 0.5)})


@pytest.mark.parametrize("selected", ["facts", "choice"])
def test_small_dichotomy_declines_existential_of_deterministic_relation(
    selected
):
    nl = _engine_with_two_witnesses(
        selected, small_dichotomy_theorem_based_solver
    )
    with pytest.raises(UnsupportedSolverError):
        _query_with_deterministic_existential(nl)


def test_predicate_with_a_deterministic_and_a_choice_rule():
    """
    A predicate with a rule joining a probabilistic choice and a
    deterministic rule is probabilistic. It was typed from whichever of
    its rules came first, which depended on the hash seed: typed
    deterministic, ``b``, true only of the choice of study 1, was lost.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(0, "a"), (0, "d"), (0, "e"), (1, "b"), (1, "d"), (2, "d")],
        name="mentions",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(0,), (1,), (2,)], name="selected_study"
    )
    res = nl.execute_datalog_program(
        """
        a(t) :- mentions(s, t), selected_study(s).
        a(t) :- mentions(s, t), mentions(s, 'a').
        ans(t, p) :- PROB[a(t)] = p.
        """
    )
    assert {
        (t, round(p, 6)) for t, p in res.as_pandas_dataframe().values
    } == {("a", 1.0), ("b", round(1 / 3, 6)), ("d", 1.0), ("e", 1.0)}


def test_predicate_with_a_deterministic_and_a_post_probabilistic_rule():
    """
    A predicate with a deterministic rule and a rule using the result of
    a probabilistic query is solved after the query, whatever the order
    of its rules: solved before it, it missed the query's results.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(0, "a"), (1, "b"), (2, "b")], name="mentions")
    nl.add_tuple_set([("z", 0.5)], name="prior")
    nl.add_uniform_probabilistic_choice_over_set(
        [(0,), (1,), (2,)], name="selected_study"
    )
    res = nl.execute_datalog_program(
        """
        q(t, p) :- PROB[mentions(s, t), selected_study(s)] = p.
        c(t, p) :- prior(t, p).
        c(t, p) :- q(t, p).
        ans(t, p) :- c(t, p).
        """
    )
    assert {
        (t, round(p, 6)) for t, p in res.as_pandas_dataframe().values
    } == {("a", round(1 / 3, 6)), ("b", round(2 / 3, 6)), ("z", 0.5)}


def test_predicate_with_a_choice_rule_and_a_post_probabilistic_rule():
    """
    A predicate with a rule using the result of a probabilistic query and
    a rule joining a probabilistic choice is typed probabilistic: a
    probabilistic query on it depends on the other query, which is
    rejected.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set([(0, "a"), (1, "b"), (2, "b")], name="mentions")
    nl.add_uniform_probabilistic_choice_over_set(
        [(0,), (1,), (2,)], name="selected_study"
    )
    with pytest.raises(UnsupportedProgramError):
        nl.execute_datalog_program(
            """
            q(t, p) :- PROB[mentions(s, t), selected_study(s)] = p.
            c(t) :- q(t, p).
            c(t) :- mentions(s, t), selected_study(s).
            ans(t, p) :- PROB[c(t)] = p.
            """
        )


@pytest.mark.parametrize("probability", [
    "PROB[active(l, s) // motor(s)] = p",
    "PROB[active(l, s)] = p",
])
def test_join_of_a_probability_with_its_maximum(probability):
    """
    The relation of a probabilistic query joined with the maximum of its
    probabilities has every row: the post-probabilistic rules over the
    aggregation, two rules away from it, were left unsolved.
    """
    studies = {0: "a", 1: "a", 2: "b", 3: "ab"}
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(s, l) for s, ls in studies.items() for l in ls], name="act"
    )
    nl.add_tuple_set([(0,), (1,), (3,)], name="mentions_motor")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )
    res = nl.execute_datalog_program(
        "active(l, s) :- act(s, l), selected_study(s)\n"
        "motor(s) :- mentions_motor(s), selected_study(s)\n"
        f"rp(l, p) :- {probability}\n"
        "best(m) :- AGGREGATE[](rp(l, p) @ max(p)) = m\n"
        "ans(l, p, m) :- rp(l, p), best(m)"
    )
    given = {0, 1, 3} if "motor" in probability else set(studies)
    expected = {
        l: sum(l in studies[s] for s in given) / len(given) for l in "ab"
    }
    assert {
        (l, round(p, 6), round(m, 6))
        for l, p, m in res.as_pandas_dataframe().values
    } == {
        (l, round(p, 6), round(max(expected.values()), 6))
        for l, p in expected.items()
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_probability_of_a_study_activating_each_voxel_is_its_share(seed):
    """
    PROB over the uniform choice of a study with a peak within a radius
    of each voxel is the share of the studies with such a peak, computed
    here by brute force: the plan joins the pairs of voxels and studies
    with the probabilities of the choice and sums them per voxel.
    """
    rng = np.random.default_rng(seed)
    voxels = {tuple(v) for v in rng.integers(0, 6, size=(60, 3))}
    peaks = {
        (*p, s) for p, s in zip(
            rng.integers(0, 6, size=(40, 3)), rng.integers(0, 12, 40)
        )
    }
    studies = sorted({s for *_, s in peaks} | {100, 101})
    nl = NeurolangPDL()
    nl.add_tuple_set(voxels, name="voxel")
    nl.add_tuple_set(peaks, name="peak")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "ans(i, j, k, p) :- PROB[voxel(i, j, k), "
        "(euclidean(i, j, k, i1, j1, k1) < 2), peak(i1, j1, k1, s), "
        "selected_study(s)] = p"
    ).as_pandas_dataframe()
    got = {
        (int(i), int(j), int(k)): p
        for i, j, k, p in result.itertuples(index=False, name=None)
    }
    expected = {}
    for v in voxels:
        near = {
            s for *q, s in peaks
            if sum((a - b) ** 2 for a, b in zip(v, q)) < 4
        }
        if near:
            expected[tuple(int(x) for x in v)] = len(near) / len(studies)
    assert got.keys() == expected.keys()
    for voxel, p in expected.items():
        assert got[voxel] == pytest.approx(p)


def test_builtin_on_region_relation_in_probabilistic_rule_is_lifted():
    """
    A builtin filtering the labels of a relation of regions, in the body
    of a probabilistic rule, is lifted: the filtered relation is a
    deterministic one. The query is solved without weighted model
    counting, as when the labels are filtered by a deterministic rule.
    """
    nl = NeurolangPDL(
        probabilistic_solvers=(
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
        ),
        probabilistic_marg_solvers=(
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
        ),
    )
    nl.add_symbol(
        lambda prefix, s: s.startswith(prefix), name="startswith",
        type_=Callable[[str, str], bool],
    )
    rng = np.random.default_rng(0)
    labels = [f"net{'AB'[i % 2]} {i}" for i in range(8)]
    nl.add_tuple_set(
        [
            (label, ExplicitVBR(rng.integers(0, 5, (3, 3)), np.eye(4)))
            for label in labels
        ],
        name="atlas",
    )
    nl.add_tuple_set(
        [
            (label, s) for label in labels for s in range(6)
            if rng.random() < .5
        ],
        name="activated",
    )
    nl.add_tuple_set([(s,) for s in range(0, 6, 2)], name="mentions")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="selected"
    )
    common = (
        "m(s) :- mentions(s), selected(s)\n"
        "ans(l, p) :- PROB[active(l, s) // m(s)] = p\n"
    )
    with nl.scope:
        inline = nl.execute_datalog_program(
            "active(l, s) :- atlas(l, r), startswith('netA', l), "
            "activated(l, s), selected(s)\n" + common
        ).as_pandas_dataframe()
    with nl.scope:
        deterministic = nl.execute_datalog_program(
            "label(l) :- atlas(l, r), startswith('netA', l)\n"
            "active(l, s) :- label(l), activated(l, s), selected(s)\n"
            + common
        ).as_pandas_dataframe()
    assert len(inline) > 0
    assert inline.sort_values("l").reset_index(drop=True).equals(
        deterministic.sort_values("l").reset_index(drop=True)
    )
