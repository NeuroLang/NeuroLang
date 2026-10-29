"""Tests of the syntax highlighting of queries."""

import io

from neurolang_apps.utils import query_highlight
from neurolang_apps.utils.query_highlight import ansi, highlight, tokens


def kinds(line, squall=False):
    return [(kind, text) for kind, text in tokens(line, squall) if kind]


def test_datalog_tokens():
    assert kinds("ans(s, p) :- PROB[a(s)] = p, (p > 0.5), ~b(s, 'x')") == [
        ("keyword", "ans"), ("operator", ":-"), ("keyword", "PROB"),
        ("predicate", "a"), ("operator", "="), ("operator", ">"),
        ("number", "0.5"), ("operator", "~"), ("predicate", "b"),
        ("string", "'x'"),
    ]


def test_comments():
    assert kinds("% the studies (all) of 'pain'") == [
        ("comment", "% the studies (all) of 'pain'"),
    ]
    assert kinds("a(s, '50%') % note") == [
        ("predicate", "a"), ("string", "'50%'"), ("comment", "% note"),
    ]


def test_squall_tokens():
    assert kinds("obtain every Study that mentions 'pain'.", True) == [
        ("keyword", "obtain"), ("keyword", "every"), ("keyword", "that"),
        ("string", "'pain'"),
    ]


def test_ansi_keeps_the_lines():
    text = ansi("q(s) :- study(s)\nans(s) :- q(s)")
    assert text.count("\n") == 1
    assert "\033[34mq\033[0m(s) \033[1;34m:-\033[0m" in text
    assert "\033[" not in highlight("ans(s) :- q(s)", "datalog",
                                    io.StringIO())


def test_colour_only_on_a_terminal(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    assert query_highlight.use_color(Terminal())
    assert not query_highlight.use_color(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not query_highlight.use_color(Terminal())


PROBABILISTIC = {"selected_study": "choice", "reported": "fact"}


def test_probabilistic_predicates_in_datalog_and_logic():
    datalog = "p(t, PROB) :- term(t, s) // selected_study(s), reported(s)"
    assert [
        (kind, text) for kind, text in tokens(datalog, kinds=PROBABILISTIC)
        if kind in ("predicate", "choice", "fact")
    ] == [
        ("predicate", "p"), ("predicate", "term"),
        ("choice", "selected_study"), ("fact", "reported"),
    ]
    logic = "p(t, P[term(t, s) | selected_study(s)]) ← reported(t) ∨ q(t)"
    assert [
        (kind, text) for kind, text in tokens(logic, kinds=PROBABILISTIC)
        if kind
    ] == [
        ("predicate", "p"), ("predicate", "term"), ("operator", "|"),
        ("choice", "selected_study"), ("operator", "←"),
        ("fact", "reported"), ("operator", "∨"), ("predicate", "q"),
    ]
    assert "\033[1;31mselected_study\033[0m" in ansi(
        logic, kinds=PROBABILISTIC
    )


def test_probabilistic_kinds_of_a_program():
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples([(0.5, "a")], name="reported")
    nl.add_uniform_probabilistic_choice_over_set([("a",)], name="selected")
    nl.add_tuple_set([("a",)], name="study")
    assert query_highlight.probabilistic_kinds(nl.program_ir) == {
        "reported": "fact", "selected": "choice",
    }
