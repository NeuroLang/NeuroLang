"""Tests of the ``percentile`` aggregation of the frontends."""
import math

import pytest

from neurolang.frontend import NeurolangDL, NeurolangPDL

PROGRAM = """
cutoff(g, percentile(v, 90)) :- score(g, s, v)
top(g, s) :- score(g, s, v) & cutoff(g, t) & (v >= t)
"""


@pytest.fixture(params=[NeurolangDL, NeurolangPDL])
def nl(request):
    nl = request.param()
    rows = [(g, s, float(s)) for g in "ab" for s in range(100)]
    rows.append(("a", 1000, math.nan))
    nl.add_tuple_set(rows, name="score")
    nl.execute_datalog_program(PROGRAM)
    return nl


def test_percentile_per_group_ignores_nan(nl):
    res = nl.execute_datalog_program("ans(g, t) :- cutoff(g, t)")
    assert dict(res) == pytest.approx({"a": 89.1, "b": 89.1})


def test_top_percent_of_a_column(nl):
    res = nl.execute_datalog_program("ans(g, s) :- top(g, s)")
    expected = {(g, s) for g in "ab" for s in range(90, 100)}
    assert set(res) == expected


def test_percentile_with_ties_keeps_every_tie(nl):
    nl.add_tuple_set([("c", s, 1.0) for s in range(10)], name="score")
    res = nl.execute_datalog_program("ans(s) :- top('c', s)")
    assert {s for s, in res} == set(range(10))


def test_two_percentiles_of_a_head(nl):
    res = nl.execute_datalog_program(
        "ci(g, percentile(v, 2.5), percentile(v, 97.5)) :- score(g, s, v)\n"
        "ans(g, lo, hi) :- ci(g, lo, hi)"
    )
    for g, lo, hi in res:
        assert (lo, hi) == pytest.approx((2.475, 96.525))
    assert {g for g, _, _ in res} == {"a", "b"}


def test_count_is_a_builtin_aggregation(nl):
    res = nl.execute_datalog_program(
        "n(g, count(s)) :- score(g, s, v)\nans(g, k) :- n(g, k)"
    )
    assert set(res) == {("a", 101), ("b", 100)}


def test_top_percent_cutoff_bound_by_a_query_constant():
    nl = NeurolangPDL()
    studies = range(100)
    nl.add_tuple_set([(s,) for s in studies], name="study")
    nl.add_tuple_set([(s % 7, s) for s in studies], name="peak")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )

    @nl.add_symbol
    def similarity(q: str, s: int) -> float:
        return float(s if q == "math" else -s)

    nl.execute_datalog_program(
        "cutoff(q, percentile(v, 90)) :- study(s) & "
        "(v == similarity(q, s))\n"
        "close(q, s) :- study(s) & (v == similarity(q, s)) & "
        "cutoff(q, t) & (v >= t)\n"
        "vg(q, i, PROB) :- (peak(i, s) & selected_study(s)) "
        "// (close(q, s) & selected_study(s))"
    )
    res = nl.execute_datalog_program("ans(s) :- close('math', s)")
    assert {s for s, in res} == set(range(90, 100))
    res = nl.execute_datalog_program("ans(i, p) :- vg('math', i, p)")
    expected = {
        i: sum(s % 7 == i for s in range(90, 100)) / 10 for i in range(7)
    }
    assert dict(res) == pytest.approx(expected)
