"""Programs of several queries, ``ans[name](...) :- ...``, solved together."""
import pytest

from nl_expr.exceptions import NeuroLangException

from ...frontend import NeurolangDL
from ...frontend.probabilistic_frontend import NeurolangPDL

PATHS = """
path(x, y) :- edge(x, y)
path(x, y) :- path(x, z), edge(z, y)
"""


def rows(answer):
    """Return the sorted rows of an answer."""
    return sorted(map(tuple, answer.as_pandas_dataframe().values.tolist()))


@pytest.fixture(params=[NeurolangDL, NeurolangPDL])
def nl(request):
    nl = request.param()
    nl.add_tuple_set([(1, 2), (2, 3), (3, 4), (4, 5)], name="edge")
    return nl


@pytest.fixture
def pdl():
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 2), (2, 3), (3, 4), (4, 5)], name="edge")
    nl.add_probabilistic_facts_from_tuples(
        [(0.3, 1), (0.6, 2), (0.9, 3)], name="P"
    )
    return nl


class TestSeveralQueries:
    def test_the_answers_by_name(self, nl):
        answers = nl.execute_datalog_program(PATHS + """
            ans[from1](y) :- path(1, y)
            ans[from3](y) :- path(3, y)
            ans[direct](x, y) :- path(x, y), edge(x, y)
        """)
        assert list(answers) == ["from1", "from3", "direct"]
        assert rows(answers["from1"]) == [(2,), (3,), (4,), (5,)]
        assert rows(answers["from3"]) == [(4,), (5,)]
        assert rows(answers["direct"]) == [(1, 2), (2, 3), (3, 4), (4, 5)]
        assert answers["from1"].columns == ("y",)

    def test_a_query_alone_is_not_a_dict(self, nl):
        answer = nl.execute_datalog_program(
            PATHS + "ans[from3](y) :- path(3, y)"
        )
        assert rows(answer) == [(4,), (5,)]

    def test_the_unnamed_query_is_named_ans(self, nl):
        answers = nl.execute_datalog_program(PATHS + """
            ans(y) :- path(4, y)
            ans[from3](y) :- path(3, y)
        """)
        assert list(answers) == ["ans", "from3"]
        assert rows(answers["ans"]) == [(5,)]

    @pytest.mark.parametrize("queries", [
        "ans[a](x) :- edge(x, y)\nans[a](y) :- edge(x, y)",
        "ans(x) :- edge(x, y)\nans(y) :- edge(x, y)",
    ])
    def test_two_queries_cannot_share_a_name(self, nl, queries):
        with pytest.raises(NeuroLangException, match="more than one query"):
            nl.execute_datalog_program(queries)

    def test_the_rules_of_the_program_are_kept(self, nl):
        nl.execute_datalog_program(PATHS + """
            ans[a](y) :- path(1, y)
            ans[b](y) :- path(2, y)
        """)
        answer = nl.execute_datalog_program("ans(y) :- path(3, y)")
        assert rows(answer) == [(4,), (5,)]

    def test_an_unbound_variable_is_reported(self, nl):
        with pytest.raises(NeuroLangException, match="consequent|nbound"):
            nl.execute_datalog_program(
                "ans[a](x) :- edge(x, y)\nans[b](z) :- edge(x, y)"
            )

    @pytest.mark.parametrize("option", ["show_rewritten", "dry_run"])
    def test_the_rewritten_program_of_all_the_queries(
        self, nl, capsys, option
    ):
        answers = nl.execute_datalog_program(PATHS + """
            ans[from1](y) :- path(1, y)
            ans[from3](y) :- path(3, y)
        """, **{option: True})
        out = capsys.readouterr().out
        assert out.count("── rewritten program ──") == 1
        assert out.count("magic_path^bf_0(") >= 2
        assert (answers is None) == (option == "dry_run")

    def test_the_plans_are_those_of_each_query(self, nl, capsys):
        nl.execute_datalog_program(PATHS + """
            ans[from1](y) :- path(1, y)
            ans[from3](y) :- path(3, y)
        """, show_ra=True)
        out = capsys.readouterr().out
        assert "── from1 ──" in out and "── from3 ──" in out

    def test_boolean_queries(self, nl):
        answers = nl.execute_datalog_program(
            "ans[some]() :- edge(x, 2)\nans[none]() :- edge(x, 1)"
        )
        assert answers == {"some": True, "none": False}


class TestSharing:
    """The queries are solved as one: what they share is computed once."""

    def test_one_chase_computes_the_shared_predicate_once(
        self, nl, monkeypatch
    ):
        chases = []
        build = nl.chase_class.build_chase_solution

        def counting(chase, *args, **kwargs):
            chases.append(chase)
            return build(chase, *args, **kwargs)

        monkeypatch.setattr(nl.chase_class, "build_chase_solution", counting)
        nl.execute_datalog_program(PATHS + """
            ans[from1](y) :- path(1, y)
            ans[from3](y) :- path(3, y)
        """)
        assert len(chases) == 1

    def test_the_rules_of_a_shared_predicate_are_not_repeated(
        self, nl, monkeypatch
    ):
        """Both queries bind the first argument of path: its two rules are
        rewritten once, with the bindings of both queries as seeds."""
        unions = []
        union_of_rules = type(nl)._union_of_rules

        def recording(unions_of_rules):
            union = union_of_rules(unions_of_rules)
            unions.append(union)
            return union

        monkeypatch.setattr(
            type(nl), "_union_of_rules", staticmethod(recording)
        )
        nl.execute_datalog_program(PATHS + """
            ans[from1](y) :- path(1, y)
            ans[from3](y) :- path(3, y)
        """)
        (union,) = unions
        names = [rule.consequent.functor.name for rule in union.formulas]
        assert sum(name.startswith("path^") for name in names) == 2
        assert sum(name.startswith("magic_path^") for name in names) == 2

    def test_each_query_keeps_the_bindings_of_its_own(self, nl):
        """With the magic seeds of all the queries, a query still gets only
        the answers of its own bindings."""
        answers = nl.execute_datalog_program(PATHS + """
            ans[a](y) :- path(1, y), edge(y, 3)
            ans[b](y) :- path(3, y)
        """)
        assert rows(answers["a"]) == [(2,)]
        assert rows(answers["b"]) == [(4,), (5,)]


QUERIES = {
    "from1": "ans(y) :- path(1, y)",
    "from3": "ans(y) :- path(3, y)",
    "all": "ans(x, y) :- path(x, y)",
    "count": "ans(x, count(y)) :- path(x, y)",
    "yes": "ans() :- path(1, 5)",
}


@pytest.mark.parametrize("names", [
    ["from1", "from3"], ["from1", "all"], ["count", "from3", "yes"],
    ["from1", "from3", "all", "count", "yes"],
])
def test_together_as_alone(nl, names):
    """Solved together, each query has the answer it has alone."""
    alone = {
        name: nl.execute_datalog_program(PATHS + QUERIES[name])
        for name in names
    }
    together = nl.execute_datalog_program(PATHS + "\n".join(
        QUERIES[name].replace("ans(", f"ans[{name}](") for name in names
    ))
    for name in names:
        if isinstance(alone[name], bool):
            assert together[name] is alone[name]
        else:
            assert rows(together[name]) == rows(alone[name])
            assert together[name].columns == alone[name].columns


def test_aggregates_over_predicates_that_other_queries_bind(nl):
    """A query binding a predicate does not restrict the aggregate another
    query computes over it."""
    program = PATHS + "cnt(x, count(y)) :- path(x, y)\n"
    queries = {
        "all": "ans(x, n) :- cnt(x, n)",
        "one": "ans(n) :- cnt(1, n)",
        "path": "ans(y) :- path(1, y)",
        "reach": "ans(n) :- cnt(x, n), path(x, 5)",
    }
    alone = {}
    for name, query in queries.items():
        with nl.scope:
            alone[name] = rows(nl.execute_datalog_program(program + query))
    with nl.scope:
        together = nl.execute_datalog_program(program + "\n".join(
            query.replace("ans(", f"ans[{name}](")
            for name, query in queries.items()
        ))
    assert {name: rows(a) for name, a in together.items()} == alone


class TestProbabilistic:
    PROGRAM = PATHS + """
        Q(x, PROB(x)) :- P(x), edge(x, y)
        R(x, PROB(x)) :- P(x), path(x, 4)
    """

    def test_probabilities_by_query(self, pdl):
        answers = pdl.execute_datalog_program(self.PROGRAM + """
            ans[q](x, p) :- Q(x, p)
            ans[r](x, p) :- R(x, p)
            ans[from3](y) :- path(3, y)
        """)
        assert rows(answers["q"]) == [(1, 0.3), (2, 0.6), (3, 0.9)]
        assert rows(answers["r"]) == [(1, 0.3), (2, 0.6), (3, 0.9)]
        assert rows(answers["from3"]) == [(4,), (5,)]

    def test_together_as_alone(self, pdl):
        alone = {
            "q": pdl.execute_datalog_program(
                self.PROGRAM + "ans(x, p) :- Q(x, p), (x > 1)"
            ),
            "n": pdl.execute_datalog_program(
                self.PROGRAM + "ans(x, count(y)) :- path(x, y)"
            ),
        }
        together = pdl.execute_datalog_program(self.PROGRAM + """
            ans[q](x, p) :- Q(x, p), (x > 1)
            ans[n](x, count(y)) :- path(x, y)
        """)
        assert rows(together["q"]) == rows(alone["q"])
        assert rows(together["n"]) == rows(alone["n"])

    def test_the_probability_of_a_query_head(self, pdl):
        answers = pdl.execute_datalog_program("""
            ans[p](x, PROB(x)) :- P(x), edge(x, y)
            ans[e](x) :- edge(x, 3)
        """)
        assert rows(answers["p"]) == [(1, 0.3), (2, 0.6), (3, 0.9)]
        assert rows(answers["e"]) == [(2,)]

    def test_a_query_on_a_probabilistic_predicate_is_refused(self, pdl):
        from nl_expr.exceptions import UnsupportedQueryError

        with pytest.raises(UnsupportedQueryError):
            pdl.execute_datalog_program(
                "ans[a](x) :- P(x)\nans[b](x) :- edge(x, y)"
            )
