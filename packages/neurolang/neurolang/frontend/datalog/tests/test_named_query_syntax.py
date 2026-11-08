"""The syntax of programs of several queries, ``ans[name](...) :- ...``."""
from ..standard_syntax import (
    is_query_head,
    parser,
    query_name,
)


class TestSyntax:
    def test_the_head_of_a_named_query(self):
        queries = parser(
            "ans[first](x) :- R(x)\nans(y) :- S(y)\nans[c]() :- R(x)"
        ).formulas
        assert all(is_query_head(q.head) for q in queries)
        assert [query_name(q.head) for q in queries] == [
            "first", "ans", "c"
        ]
        assert [len(q.head.args) for q in queries] == [1, 1, 0]

    def test_a_named_query_with_aggregates_and_probabilities(self):
        queries = parser(
            "ans[n](x, count(y)) :- R(x, y)\nans[p](x, PROB(x)) :- R(x, y)"
        ).formulas
        assert [query_name(q.head) for q in queries] == ["n", "p"]

    def test_other_heads_are_not_queries(self):
        rule = parser("answer(x) :- R(x)").formulas[0]
        assert not is_query_head(rule.consequent)
