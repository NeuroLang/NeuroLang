"""Arithmetic in the head of a rule is no aggregation (sum, count...):
it is refused, with the way to write it, rather than aggregated."""
import pytest

from nl_expr.exceptions import ForbiddenExpressionError
from neurolang.frontend import NeurolangDL


@pytest.fixture
def nl():
    nl = NeurolangDL()
    nl.add_tuple_set([("a", "x", 2.0), ("a", "y", 4.0), ("b", "x", 6.0)],
                     name="r")
    return nl


@pytest.mark.parametrize("program", [
    "ans(l2, n / 2.0) :- r('a', l2, n)",
    "ans(l, l2, n * 2.0) :- r(l, l2, n)",
    "ans(x, sum(n) / 2) :- r(x, l2, n)",
    "q(l2, n + 1) :- r(l, l2, n)\nans(x, v) :- q(x, v)",
])
def test_arithmetic_in_a_head_is_refused(nl, program):
    with pytest.raises(ForbiddenExpressionError, match="in the body"):
        nl.execute_datalog_program(program)


def test_arithmetic_in_the_body_and_aggregations(nl):
    result = nl.execute_datalog_program(
        "ans(l2, v) :- r('a', l2, n), (v == n / 2.0)")
    assert set(result.as_pandas_dataframe().itertuples(index=False,
                                                       name=None)) == {
        ("x", 1.0), ("y", 2.0)}
    result = nl.execute_datalog_program("ans(x, sum(n)) :- r(x, l2, n)")
    assert set(result.as_pandas_dataframe().itertuples(index=False,
                                                       name=None)) == {
        ("a", 6.0), ("b", 6.0)}
