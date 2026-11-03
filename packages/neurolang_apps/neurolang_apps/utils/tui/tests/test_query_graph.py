"""Tests for the graph of a query: its data, rules and probabilities."""

import io

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
    """An engine as the registry's: data, a choice of a study, a rule."""
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


def _kinds(graph):
    return {node.label: node.kind for node in graph.nodes}


def _edges(graph):
    labels = {node.id: node.label for node in graph.nodes}
    return {
        (labels[e.source], labels[e.target], e.kind) for e in graph.edges
    }


def test_the_graph_goes_from_the_data_to_the_answer(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    kinds = _kinds(graph)
    prob = next(label for label in kinds if label.startswith("PROB"))
    assert kinds == {
        "activated": "data", "tfidf": "data", "selected_study": "choice",
        "mentions": "rule", prob: "probability", "ans": "answer",
    }
    assert _edges(graph) == {
        ("activated", prob, "uses"),
        ("selected_study", prob, "uses"),
        ("mentions", prob, "given"),
        ("selected_study", prob, "given"),
        ("tfidf", "mentions", "uses"),
        (prob, "ans", "uses"),
    }


def test_a_probability_is_named_by_what_it_is_of_not_a_fresh_symbol(nl):
    graph = query_graph(nl, CO_ACTIVATION)
    [prob] = [node for node in graph.nodes if node.kind == "probability"]
    assert "fresh" not in prob.label + prob.detail
    assert "//" in prob.label
    assert "r1 < r2" in prob.detail and "mentions(s, 'memory')" in prob.detail


def test_a_conditional_probability_of_the_engine_is_read_back(nl):
    """The engine solves p_given as p_given^cond_num / p_given^cond_den:
    the graph shows p_given as it is written."""
    graph = query_graph(nl, "ans(r, p) :- p_given('memory', r, p)")
    assert _kinds(graph)["p_given"] == "probability"
    assert _edges(graph) == {
        ("activated", "p_given", "uses"),
        ("selected_study", "p_given", "uses"),
        ("mentions", "p_given", "given"),
        ("selected_study", "p_given", "given"),
        ("tfidf", "mentions", "uses"),
        ("p_given", "ans", "uses"),
    }


def test_a_copy_of_a_stored_relation_of_no_name_is_data(nl):
    """As activated, which the engine precomputes and stores."""
    stored = Symbol.fresh()
    nl.program_ir.add_extensional_predicate_from_tuples(
        stored, [("A",), ("B",)]
    )
    nl.execute_datalog_program(f"region(r) :- {stored.name}(r)")
    graph = query_graph(nl, "ans(r) :- region(r)")
    assert _kinds(graph) == {"region": "data", "ans": "answer"}
    assert "2 rows" in {n.label: n.detail for n in graph.nodes}["region"]


def test_rules_using_a_random_choice_are_probabilistic(nl):
    graph = query_graph(
        nl,
        "active(r, s) :- activated(r, s), selected_study(s)\n"
        "trusted(r) :- reliable(r)\n"
        "named(r) :- activated(r, s)\n"
        "ans(r, p) :- PROB[active(r, s), trusted(r)] = p, named(r)",
    )
    kinds = _kinds(graph)
    assert kinds["active"] == kinds["trusted"] == "probabilistic"
    assert kinds["reliable"] == "facts"
    assert kinds["named"] == "rule"
    assert kinds["ans"] == "answer"


def test_a_negated_atom_is_its_own_kind_of_edge(nl):
    graph = query_graph(nl, "ans(s) :- activated(r, s), ~mentions(s, 'pain')")
    assert ("mentions", "ans", "negated") in _edges(graph)
    assert ("activated", "ans", "uses") in _edges(graph)


def test_a_recursive_rule_depends_on_itself(nl):
    graph = query_graph(
        nl,
        "near(x, y) :- activated(x, s), activated(y, s)\n"
        "near(x, z) :- near(x, y), near(y, z)\n"
        "ans(x, y) :- near(x, y)",
    )
    assert ("near", "near", "uses") in _edges(graph)
    assert _kinds(graph)["near"] == "rule"


def test_the_rules_and_descriptions_are_the_details_of_the_nodes(nl):
    graph = query_graph(
        nl, "ans(s) :- mentions(s, 'pain')",
        descriptions={"tfidf": "TF-IDF of the terms of the studies"},
    )
    details = {node.label: node.detail for node in graph.nodes}
    assert "TF-IDF of the terms" in details["tfidf"]
    assert "2 rows" in details["tfidf"]
    assert "tfidf(s, t, v)" in details["mentions"]


def test_a_squall_query_has_its_graph(nl):
    graph = query_graph(
        nl, "obtain every Tfidf that mentions 'pain'.", squall=True
    )
    assert any(node.kind == "answer" for node in graph.nodes)


def test_the_page_draws_every_node_and_explains_the_kinds(nl):
    page = graph_page(query_graph(nl, CO_ACTIVATION), "Last query")
    assert page.startswith("<!DOCTYPE html>") and "<svg" in page
    for name in ("activated", "selected_study", "mentions", "ans"):
        assert f">{name}<" in page
    assert "r1 &lt; r2" in page and "r1 < r2" not in page
    for kind in ("Data", "Random choice", "Probability", "given"):
        assert kind in page


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
    assert ">mentions<" in page and ">tfidf<" in page


def test_the_page_gets_the_graph_of_a_query_it_runs(nl):
    repl = WebRepl(nl, engine_name="neurosynth")
    assert ">mentions<" in repl.submit("ans(s) :- mentions(s, 'pain')").graph
    assert not repl.submit(".help").graph
    assert ">mentions<" in repl.submit(".graph").graph
