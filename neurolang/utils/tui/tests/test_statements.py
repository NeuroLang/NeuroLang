"""Tests for the splitting of the REPL input into statements."""

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.utils.tui.statements import (  # noqa: E402
    is_complete,
    split_statements,
)


@pytest.mark.parametrize("text, squall, complete", [
    ("ans(x) :- P(x)", False, True),
    ("ans(x) :-", False, False),
    ("ans(x) :- P(x),", False, False),
    ("ans(x) :- P(x", False, False),
    ("ans(x) :- P(x, ')')", False, True),
    ("ans(x) :- P(x, '(')", False, True),
    (".help", False, True),
    (".mode squall", True, True),
    ("obtain every P.", True, True),
    ("obtain every P", True, False),
    ("", True, True),
])
def test_is_complete(text, squall, complete):
    assert is_complete(text, squall) is complete


def test_split_statements_multi_line_and_commands():
    lines = [
        "ans(x) :-\n", "    P(x),\n", "    Q(x)\n", ".more\n", "\n",
        "a(x) :- P(x)\n", "ans(x) :- a(x)\n",
    ]
    assert list(split_statements(lines, lambda: False)) == [
        "ans(x) :-\n    P(x),\n    Q(x)", ".more", "a(x) :- P(x)",
        "ans(x) :- a(x)",
    ]


def test_split_statements_blank_line_ends_incomplete_statement():
    lines = ["ans(x) :- P(x\n", "\n", ".quit\n"]
    assert list(split_statements(lines, lambda: False)) == [
        "ans(x) :- P(x", ".quit",
    ]


def test_split_statements_squall_until_period():
    lines = ["define as Active every person\n", "that plays.\n",
             "obtain every Active.\n"]
    assert list(split_statements(lines, lambda: True)) == [
        "define as Active every person\nthat plays.",
        "obtain every Active.",
    ]


def test_split_statements_follows_mode_changes():
    mode = {"squall": False}
    statements = []
    for statement in split_statements(
        [".mode squall\n", "obtain every\n", "P.\n"],
        lambda: mode["squall"],
    ):
        statements.append(statement)
        if statement == ".mode squall":
            mode["squall"] = True
    assert statements == [".mode squall", "obtain every\nP."]


def test_split_at_commands_keeps_programs_whole():
    from neurolang.utils.tui.statements import split_at_commands

    text = (
        "a(x) :- b(x)\nc(x) :- a(x)\n\n.sort x\n  .head 5\n"
        "ans(x) :-\n  c(x)"
    )
    assert split_at_commands(text) == [
        "a(x) :- b(x)\nc(x) :- a(x)", ".sort x", ".head 5",
        "ans(x) :-\n  c(x)",
    ]
    assert split_at_commands("ans(x) :- P(x)") == ["ans(x) :- P(x)"]
