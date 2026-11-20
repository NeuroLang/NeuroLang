import numpy as np
import pytest

from ..deterministic_frontend import NeurolangDL
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


def test_deterministic_segregation_query():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("a",), ("b",), ("c",)],
        name="Network",
    )
    nl.add_tuple_set(
        [("s1",), ("s2",), ("s3",)],
        name="Study",
    )
    nl.add_tuple_set(
        [
            ("a", "s1"),
            ("b", "s1"),
            ("b", "s2"),
            ("c", "s2"),
            ("c", "s3"),
        ],
        name="NetworkReported",
    )
    with nl.environment as e:
        e.SegregationQuery[e.n, e.s] = e.NetworkReported(
            e.n, e.s
        ) & ~nl.exists(e.n2, e.NetworkReported(e.n2, e.s) & (e.n2 != e.n))
        result = nl.query((e.n, e.s), e.SegregationQuery(e.n, e.s))
    expected = {
        ("c", "s3"),
    }
    assert_almost_equal(result, expected)


def test_two_negations_over_columns_of_mixed_types():
    """The plan subtracts named relations whose columns are in different
    orders; the type of each difference must follow its own columns, or a
    join after it unifies a string column with an integer one."""
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(1, "a", "left"), (2, "a", "left"), (2, "b", "left"),
         (3, "a", "left")],
        name="t",
    )
    nl.add_tuple_set(
        [(1, "left"), (2, "left"), (3, "left"), (3, "right")], name="ih"
    )
    nl.add_tuple_set([("left", "right"), ("right", "left")], name="opp")
    result = nl.execute_datalog_program(
        'ans(x, h) :- t(x, "a", h), ~t(x, "b", h), opp(h, o), ~ih(x, o)'
    )
    assert set(result) == {(1, "left")}


def _solve(frontend, program):
    nl = frontend()
    nl.add_tuple_set([(1,), (2,), (3,)], name="p")
    nl.add_tuple_set([(1,)], name="q")
    nl.execute_datalog_program(program)
    return set(nl.solve_all()["r"].as_pandas_dataframe().itertuples(
        index=False, name=None
    ))


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize("aggregation", ["sum", "count", "max"])
@pytest.mark.parametrize("program", [
    "r({agg}(x)) :- p(x), ~q(x)",
    "r(m) :- AGGREGATE[()]((p(x) & ~q(x)) @ {agg}(x)) = m",
])
def test_aggregation_over_negated_body(frontend, aggregation, program):
    """An aggregation over a body with a negated atom equals the aggregation
    over an intermediate rule that holds the negation."""
    expected = _solve(
        frontend, f"s(x) :- p(x), ~q(x)\nr({aggregation}(x)) :- s(x)"
    )
    assert expected == {
        "sum": {(5,)}, "count": {(2,)}, "max": {(3,)}
    }[aggregation]
    assert _solve(frontend, program.format(agg=aggregation)) == expected


def test_probabilistic_query_on_aggregation_over_negated_body():
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,), (3,)], name="p")
    nl.add_tuple_set([(1,)], name="q")
    nl.add_probabilistic_facts_from_tuples(
        [(0.5, 2), (0.3, 3)], name="pp"
    )
    nl.execute_datalog_program(
        "r(x, count(y)) :- p(x), p(y), ~q(y)\n"
        "z(x, c, PROB) :- pp(x), r(x, c)"
    )
    result = nl.solve_all()["z"].as_pandas_dataframe()
    assert_almost_equal(
        set(result.itertuples(index=False, name=None)),
        {(2, 2, 0.5), (3, 2, 0.3)},
    )
    assert len(result) == 2
