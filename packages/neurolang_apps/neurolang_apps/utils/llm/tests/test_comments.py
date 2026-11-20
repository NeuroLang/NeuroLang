"""Tests of the comment line explaining each rule of the query."""

import json

import pytest

from ..agent import ReactAgent
from ..comments import (
    commented_query,
    given_comments,
    parse_comments,
    split_rules,
)
from ..validation import check_query
from .fakes import ScriptedClient, assistant_config, provider

QUERY = """\
pain(s) :- mentions(s, 'pain')
counts(t, count(s)) :- mentions(s, t),
    pain(s)
ans(t, n) :- counts(t, n)"""


def test_split_datalog_rules():
    rules = split_rules(QUERY, "datalog")
    assert rules == [
        "pain(s) :- mentions(s, 'pain')",
        "counts(t, count(s)) :- mentions(s, t),\n    pain(s)",
        "ans(t, n) :- counts(t, n)",
    ]


def test_split_datalog_rules_open_brackets_and_comments():
    query = (
        "% an old comment\n"
        "p(x) :- q(x, (\n  1))\n"
        "ans(x) :- p(x), (x == 'a, b :-')"
    )
    assert split_rules(query, "datalog") == [
        "p(x) :- q(x, (\n  1))", "ans(x) :- p(x), (x == 'a, b :-')",
    ]


@pytest.mark.squall
def test_split_squall_sentences():
    query = (
        "define as Painful every Study that mentions 'pain. chronic'. "
        "obtain every Study that reports 1.5."
    )
    assert split_rules(query, "squall") == [
        "define as Painful every Study that mentions 'pain. chronic'.",
        "obtain every Study that reports 1.5.",
    ]


def test_parse_comments():
    text = '```json\n["The studies mentioning pain", "Count %"]\n```'
    assert parse_comments(text, 2) == [
        "The studies mentioning pain", "Count",
    ]
    with pytest.raises(ValueError, match="2 strings"):
        parse_comments('["one"]', 2)
    with pytest.raises(ValueError, match="no JSON list"):
        parse_comments("Sure!", 1)


def test_commented_query_runs(nl):
    rules = split_rules(QUERY, "datalog")
    commented = commented_query(rules, ["pain studies", "", "the answer"])
    assert commented.splitlines()[0] == "% pain studies"
    assert "% \n" not in commented
    assert check_query(nl, commented, "datalog").ok


PAIN = "ans(s) :- mentions(s, 'pain')"
FINAL = "Action: final_answer\nAction Input: " + json.dumps(
    {"query": PAIN, "explanation": "x"}
)


def _react(nl, replies, **options):
    steps = []
    react = ReactAgent(
        nl, provider(), assistant_config(comment_rules=True, **options),
        "datalog", client=ScriptedClient(replies), on_step=steps.append,
    )
    return react, steps


def test_the_accepted_query_is_commented(nl):
    react, steps = _react(nl, [FINAL, '["The studies mentioning pain"]'])
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN
    assert result.commented == f"% The studies mentioning pain\n{PAIN}"
    assert result.calls == 2
    assert steps[-1].action == "comment_rules"
    request = react.client.requests[-1]["messages"]
    assert "1. " + PAIN in request[1]["content"]
    assert "Which studies mention pain?" in request[1]["content"]


def test_a_reply_without_comments_keeps_the_query(nl):
    react, steps = _react(nl, [FINAL, "The query finds the studies."])
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN
    assert result.commented == ""
    assert "No comments" in steps[-1].observation


def test_no_comments_without_a_valid_query(nl):
    react, _ = _react(nl, ["I do not know."], max_steps=1)
    result = react.run("Which studies mention pain?")
    assert not result.valid and len(react.client.requests) == 1


def test_comments_given_with_the_answer_need_no_request(nl):
    final = "Action: final_answer\nAction Input: " + json.dumps({
        "query": PAIN, "explanation": "x",
        "comments": ["The studies % mentioning\n pain"],
    })
    react, steps = _react(nl, [final])
    result = react.run("Which studies mention pain?")
    assert result.valid
    assert result.commented == f"% The studies mentioning pain\n{PAIN}"
    assert result.calls == 1
    assert all(step.action != "comment_rules" for step in steps)


def test_comments_not_one_per_rule_are_asked_for(nl):
    final = "Action: final_answer\nAction Input: " + json.dumps({
        "query": PAIN, "explanation": "x", "comments": ["one", "two"],
    })
    react, steps = _react(nl, [final, '["The studies mentioning pain"]'])
    result = react.run("Which studies mention pain?")
    assert result.commented == f"% The studies mentioning pain\n{PAIN}"
    assert result.calls == 2 and steps[-1].action == "comment_rules"


def test_given_comments_are_one_line_per_rule():
    assert given_comments(QUERY, "datalog", ["a", "b", "c"]) == (
        commented_query(split_rules(QUERY, "datalog"), ["a", "b", "c"])
    )
    assert given_comments(QUERY, "datalog", ["a", "b"]) == ""
    assert given_comments(QUERY, "datalog", ["a", " ", "c"]) == ""
    assert given_comments(QUERY, "datalog", None) == ""


def test_the_time_of_the_model_and_of_the_tools_is_kept(nl):
    react, steps = _react(nl, [FINAL, '["The studies mentioning pain"]'])
    result = react.run("Which studies mention pain?")
    assert result.model_seconds > 0 and result.tool_seconds > 0
    assert result.model_seconds + result.tool_seconds <= result.seconds
    final = next(s for s in steps if s.action == "final_answer")
    assert 0 < final.seconds <= result.tool_seconds
