"""Tests for the graph of a query: its data, rules and probabilities."""

import io

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from nl_expr.expressions import Symbol  # noqa: E402
from neurolang_apps.utils.tui.app import (  # noqa: E402
    InteractiveTuiApp, TuiSettings
)
from neurolang_apps.utils.tui.query_graph import (  # noqa: E402
    graph_page, query_graph
)
from neurolang_apps.utils.tui.web import WebRepl  # noqa: E402

CO_ACTIVATION = (
    "ans(r1, r2, p) :- PROB[(activated(r1, s), activated(r2, s), "
    "selected_study(s), (r1 < r2)) // (mentions(s, 'memory'), "
    "selected_study(s))] = p"
)


def _engine():
    """An engine as the registry's: data, a choice of a study, rules."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("A", 1), ("B", 1), ("A", 2)], name="activated")
    nl.add_tuple_set([(1, "memory", .1), (2, "pain", .2)], name="tfidf")
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,)], name="selected_study"
    )
    nl.add_probabilistic_facts_from_tuples([(.5, "A")], name="reliable")
    nl.execute_datalog_program(
        "mentions(s, t) :- tfidf(s, t, v), (v > 0.05)\n"
        "p_given(t, r, PROB) :- (activated(r, s) & selected_study(s)) "
        "// (mentions(s, t) & selected_study(s))"
    )
    return nl


@pytest.fixture
def nl():
    return _engine()


def _name(node):
    """The name of a node in the tests: PROB for a probability in a body."""
    return "PROB" if node.id.startswith("fresh_") else node.id


def _nodes(graph):
    return {_name(node): node for node in graph.nodes}


def _kinds(graph):
    return {name: node.kind for name, node in _nodes(graph).items()}


def _edges(graph):
    """The edges as (source, target, kind), once per atom using them."""
    names = {node.id: _name(node) for node in graph.nodes}
    return sorted(
        (names[e.source], names[e.target], e.kind) for e in graph.edges
    )


def _prob(graph):
    [node] = [n for n in graph.nodes if n.kind == "probability"]
    return node


def test_the_graph_goes_from_the_data_to_the_answer(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    assert _kinds(graph) == {
        "activated": "data", "tfidf": "data", "selected_study": "choice",
        "mentions": "rule", "PROB": "probability", "ans": "answer",
    }
    # activated is used twice, joined with itself: an edge for each use.
    assert _edges(graph) == sorted([
        ("activated", "PROB", "uses"), ("activated", "PROB", "uses"),
        ("selected_study", "PROB", "uses"),
        ("mentions", "PROB", "given"),
        ("selected_study", "PROB", "given"),
        ("tfidf", "mentions", "uses"),
        ("PROB", "ans", "uses"),
    ])


def test_each_atom_is_a_row_where_its_edge_arrives(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    prob = _prob(graph)
    assert [(row.kind, row.text) for row in prob.rows] == [
        ("atom", "activated(r1, s)"), ("atom", "activated(r2, s)"),
        ("atom", "selected_study(s)"), ("filter", "r1 < r2"),
        ("heading", "given"),
        ("atom", "mentions(s, 'memory')"), ("atom", "selected_study(s)"),
    ]
    names = {node.id: _name(node) for node in graph.nodes}
    arrivals = sorted(
        (names[e.source], prob.rows[e.row].text) for e in graph.edges
        if e.target == prob.id
    )
    assert arrivals == [
        ("activated", "activated(r1, s)"), ("activated", "activated(r2, s)"),
        ("mentions", "mentions(s, 'memory')"),
        ("selected_study", "selected_study(s)"),
        ("selected_study", "selected_study(s)"),
    ]
    [answer] = [n for n in graph.nodes if n.kind == "answer"]
    assert answer.rows[0].text.startswith("PROB[")
    assert "fresh" not in answer.rows[0].text


def test_a_rule_tells_how_it_combines_its_rows(nl):
    graph = query_graph(
        nl,
        "pair(r1, r2) :- activated(r1, s), activated(r2, s), (r1 < r2)\n"
        "both(r, t) :- activated(r, s), tfidf(u, t, v)\n"
        "n(r, count(s)) :- activated(r, s)\n"
        "half(r, h) :- n(r, c), (h == c / 2)\n"
        "ans(r1, r2, t, c) :- pair(r1, r2), both(r1, t), n(r1, c), "
        "half(r1, h)",
    )
    nodes = _nodes(graph)
    assert nodes["pair"].label == "pair(r1, r2)"
    assert "joined on s" in nodes["pair"].notes
    assert "a row for each (r1, r2), for some s" in nodes["pair"].notes
    assert "every combination of their rows" in nodes["both"].notes
    assert "count(s) for each (r)" in nodes["n"].notes
    assert [n for n in nodes["half"].notes if n.startswith("h = ")]
    assert not [n for n in nodes["half"].notes if "for some" in n]
    assert "joined on r1" in nodes["ans"].notes


def test_a_probability_tells_how_it_is_computed(nl):
    notes = _prob(query_graph(nl, CO_ACTIVATION)).notes
    assert "joined on s" in notes
    assert "p = P(A, C) / P(C): A above given, C below" in notes
    assert (
        "selected_study chooses one of its 2 rows, each with probability 1/2"
        in notes
    )
    assert (
        "so p is the share of those choices where A and C hold, among "
        "those where C holds" in notes
    )
    assert "a probability p for each (r1, r2), summing over s" in notes
    plain = _prob(query_graph(
        nl, "ans(r, p) :- PROB[activated(r, s), selected_study(s)] = p"
    )).notes
    assert "p = P(A)" in plain
    assert "so p is the share of those choices where A holds" in plain


def test_no_share_is_claimed_with_probabilistic_facts_upstream(nl):
    notes = _prob(query_graph(
        nl,
        "trusted(r) :- reliable(r)\n"
        "ans(r, p) :- PROB[activated(r, s), selected_study(s), trusted(r)]"
        " = p",
    )).notes
    assert not any("share" in note for note in notes)
    assert any(
        "selected_study" in note and "reliable" in note for note in notes
    )


def test_no_share_is_claimed_for_a_choice_not_uniform(nl):
    nl.add_probabilistic_choice_from_tuples(
        [(.9, 1), (.1, 2)], name="weighted"
    )
    notes = _prob(query_graph(
        nl, "ans(r, p) :- PROB[activated(r, s), weighted(s)] = p"
    )).notes
    assert not any("share" in note for note in notes)
    assert any("weighted" in note for note in notes)


def test_the_edges_tell_what_flows_through_them(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    names = {node.id: _name(node) for node in graph.nodes}
    flows = {(names[e.source], names[e.target], e.flow) for e in graph.edges}
    assert ("tfidf", "mentions", "rows") in flows
    assert ("selected_study", "PROB", "uncertain") in flows
    assert ("PROB", "ans", "probabilities") in flows


def test_the_answer_knows_its_probability_columns(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    assert _nodes(graph)["ans"].probabilities == (2,)
    graph = query_graph(nl, "ans(p, r) :- p_given('memory', r, p)")
    assert _nodes(graph)["ans"].probabilities == (0,)


def test_a_conditional_probability_of_the_engine_is_read_back(nl):
    """The engine solves p_given as p_given^cond_num / p_given^cond_den:
    the graph shows p_given as it is written."""
    graph = query_graph(nl, "ans(r, p) :- p_given('memory', r, p)")
    assert _kinds(graph)["p_given"] == "probability"
    assert _nodes(graph)["p_given"].label == "p_given(t, r, PROB)"
    assert _edges(graph) == sorted([
        ("activated", "p_given", "uses"),
        ("selected_study", "p_given", "uses"),
        ("mentions", "p_given", "given"),
        ("selected_study", "p_given", "given"),
        ("tfidf", "mentions", "uses"),
        ("p_given", "ans", "uses"),
    ])


def test_a_copy_of_a_stored_relation_of_no_name_is_data(nl):
    """As activated, which the engine precomputes and stores."""
    stored = Symbol.fresh()
    nl.program_ir.add_extensional_predicate_from_tuples(
        stored, [("A",), ("B",)]
    )
    nl.execute_datalog_program(f"region(r) :- {stored.name}(r)")
    graph = query_graph(nl, "ans(r) :- region(r)")
    assert _kinds(graph) == {"region": "data", "ans": "answer"}
    assert "2 rows" in _nodes(graph)["region"].detail


def test_rules_using_a_random_choice_are_probabilistic(nl):
    graph = query_graph(
        nl,
        "active(r, s) :- activated(r, s), selected_study(s)\n"
        "trusted(r) :- reliable(r)\n"
        "named(r) :- activated(r, s)\n"
        "maybe(r) :: 0.5 :- activated(r, s)\n"
        "likely(r) :- maybe(r)\n"
        "ans(r, p) :- PROB[active(r, s), trusted(r), likely(r)] = p, named(r)",
    )
    kinds = _kinds(graph)
    assert kinds["active"] == kinds["trusted"] == "probabilistic"
    assert kinds["maybe"] == kinds["likely"] == "probabilistic"
    assert kinds["reliable"] == "facts"
    assert kinds["named"] == "rule"
    assert kinds["ans"] == "answer"
    assert any("selected_study" in n for n in _nodes(graph)["active"].notes)


def test_a_negated_atom_is_its_own_kind_of_edge_and_row(nl):
    graph = query_graph(nl, "ans(s) :- activated(r, s), ~mentions(s, 'pain')")
    assert ("mentions", "ans", "negated") in _edges(graph)
    assert ("activated", "ans", "uses") in _edges(graph)
    rows = [(row.kind, row.text) for row in _nodes(graph)["ans"].rows]
    assert ("negated", "not mentions(s, 'pain')") in rows


def test_a_recursive_rule_depends_on_itself(nl):
    graph = query_graph(
        nl,
        "near(x, y) :- activated(x, s), activated(y, s)\n"
        "near(x, z) :- near(x, y), near(y, z)\n"
        "ans(x, y) :- near(x, y)",
    )
    assert ("near", "near", "uses") in _edges(graph)
    assert _kinds(graph)["near"] == "rule"
    assert [row.text for row in _nodes(graph)["near"].rows if (
        row.kind == "heading"
    )] == ["or"]


def test_the_data_show_their_columns_and_size(nl):
    graph = query_graph(
        nl, "ans(s) :- mentions(s, 'pain')",
        descriptions={"tfidf": "TF-IDF of the terms of the studies"},
        columns={"tfidf": ("study", "term", "tfidf")},
    )
    nodes = _nodes(graph)
    assert nodes["tfidf"].columns == "study, term, tfidf"
    assert "TF-IDF of the terms" in nodes["tfidf"].detail
    assert "2 rows" in nodes["tfidf"].detail
    assert "tfidf(s, t, v)" in nodes["mentions"].detail


def test_a_squall_query_has_its_graph(nl):
    graph = query_graph(
        nl, "obtain every Tfidf that mentions 'pain'.", squall=True
    )
    assert any(node.kind == "answer" for node in graph.nodes)


def test_aggregations_have_no_fresh_names(nl):
    graph = query_graph(
        nl, "q(r, n) :- AGGREGATE[r](activated(r, s) @ count(s)) = n\n"
        "ans(r, n) :- q(r, n)",
    )
    assert _kinds(graph) == {"q": "rule", "activated": "data", "ans": "answer"}
    assert not any("fresh" in n.label for n in graph.nodes)


def test_the_page_draws_every_node_and_explains_the_kinds(nl):
    page = graph_page(query_graph(nl, CO_ACTIVATION), "Last query")
    assert page.startswith("<!DOCTYPE html>") and "<svg" in page
    for name in ("activated", "selected_study", "mentions(s, t)", "ans"):
        assert f">{name}" in page
    assert "r1 &lt; r2" in page and "r1 < r2" not in page
    for kind in ("Data", "Random choice", "Probability", "uncertain rows"):
        assert kind in page
    assert 'class="histogram"' not in page


def test_the_page_draws_the_distribution_of_the_probabilities(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    frame = pd.DataFrame({"r1": ["A", "A"], "r2": ["B", "C"], "p": [.5, 1.]})
    page = graph_page(graph, "Last query", frame)
    assert 'class="histogram"' in page
    assert "p over the 2 rows" in page


def _app(nl):
    console = Console(file=io.StringIO(), width=100)
    return InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )


def test_graph_shows_the_graph_of_the_last_query(nl):
    app = _app(nl)
    pages = []
    app.show_graph = pages.append
    app._handle_dot_command(".graph")
    assert "No query has been run yet" in app.console.file.getvalue()
    assert not pages
    app._execute_and_display("ans(s) :- mentions(s, 'pain')")
    app._handle_dot_command(".graph")
    [page] = pages
    assert ">mentions(s, t)" in page and ">tfidf" in page


def test_graph_draws_the_probabilities_of_the_last_result_only(nl):
    app = _app(nl)
    pages = []
    app.show_graph = pages.append
    app._execute_and_display(
        "ans(r, p) :- PROB[activated(r, s), selected_study(s)] = p"
    )
    app._handle_dot_command(".graph")
    assert 'class="histogram"' in pages[-1]
    # A query that fails has no rows: those of the last one are not its.
    app._execute_and_display(
        "ans(r, p) :- PROB[activated(r, s), selected_study(s)] = p, "
        "undefined_here(r)"
    )
    app._handle_dot_command(".graph")
    assert 'class="histogram"' not in pages[-1]


def test_the_page_gets_the_graph_of_a_query_it_runs(nl):
    repl = WebRepl(nl, engine_name="neurosynth")
    reply = repl.submit(
        "ans(r, p) :- PROB[activated(r, s), selected_study(s)] = p"
    )
    assert ">activated" in reply.graph
    assert 'class="histogram"' in reply.graph
    assert not repl.submit(".help").graph
    assert ">activated" in repl.submit(".graph").graph
