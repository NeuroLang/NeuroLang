"""Tests of the syntax highlighting of queries."""

import io

from neurolang.utils import query_highlight
from neurolang.utils.query_highlight import ansi, highlight, tokens


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
