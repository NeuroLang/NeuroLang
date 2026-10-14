import pytest

from ..context import EngineContext
from ..validation import arity_errors, check_query, run_preview


def symbols(nl):
    """Return the names of the symbols of the engine."""
    return {getattr(s, "name", s) for s in nl.symbol_table}


@pytest.mark.parametrize("query, language", [
    ("ans(s) :- mentions(s, 'pain')", "datalog"),
    ("obtain every Study that mentions 'pain'.", "squall"),
])
def test_valid_queries(nl, query, language):
    result = check_query(nl, query, language)
    assert result.ok, result.message
    if language == "squall":
        assert "mentions(s₀, 'pain')" in result.datalog


@pytest.mark.parametrize("query, language, message", [
    ("", "datalog", "empty"),
    ("ans(s) :- mentions(s, 'pain'", "datalog", "parse error"),
    ("obtain every Study that mentions 'pain'", "squall", "parse error"),
    ("q(s) :- study(s)", "datalog", "add a rule whose head is ans"),
    ("define as Q every Study.", "squall", "starting with 'obtain'"),
    ("ans(s) :- mention(s, 'pain')", "datalog",
     "predicate mention has no facts or rules — did you mean mentions?"),
    ("obtain every Study that mention 'pain'.", "squall",
     "did you mean mentions?"),
    ("ans(s) :- mentions(s)", "datalog",
     "mentions has 2 columns, not 1"),
    ("obtain every Mentions (?s).", "squall", "mentions has 2 columns"),
])
def test_invalid_queries(nl, query, language, message):
    result = check_query(nl, query, language)
    assert not result.ok
    assert message in result.message


def test_checks_leave_the_engine_unchanged(nl):
    before = symbols(nl)
    check_query(nl, "q(s) :- study(s)\nans(s) :- q(s)", "datalog")
    run_preview(nl, "define as Q every Study.\nobtain every Q.", "squall")
    assert symbols(nl) == before


def test_arity_errors_ignore_strings_and_functions(nl):
    assert arity_errors(nl, "ans(s) :- mentions(s, 'a, b'), exp(s)") == []


def test_run_preview(nl):
    preview = run_preview(nl, "ans(s, t) :- mentions(s, t)", "datalog", 2)
    assert preview.ok and preview.rows == 4
    assert preview.text.startswith("4 rows; columns s, t")
    assert len(preview.text.splitlines()) == 4
    squall = run_preview(
        nl, "obtain every Study that mentions 'pain'.", "squall"
    )
    assert squall.rows == 1 and "columns study" in squall.text


def test_run_preview_of_several_results(nl):
    preview = run_preview(
        nl, "obtain every Study as A. obtain every Study that mentions "
        "'fear' as B.", "squall",
    )
    assert preview.rows == 4
    assert "a: 3 rows" in preview.text and "b: 1 rows" in preview.text


def test_run_preview_error(nl):
    preview = run_preview(nl, "ans(s) :- mentions(s, 'pain'", "datalog")
    assert not preview.ok and "parse error" in preview.text


def test_schema_summary(nl):
    nl.execute_datalog_program("pain_study(s) :- mentions(s, 'pain')")
    summary = EngineContext(nl).schema_summary()
    lines = summary.splitlines()
    assert "- mentions(c0, c1) [stored, 4 rows]" in lines
    assert "- study(c0) [stored, 3 rows]" in lines
    assert any(line.startswith("- pain_study(s) [") for line in lines)
    assert lines[-1].startswith("Functions: ") and "exp" in lines[-1]


def test_schema_summary_uses_the_engine_configuration(nl, monkeypatch):
    declared = {"mentions": {
        "columns": ["study_id", "term"], "description": "Studies and terms",
    }}
    monkeypatch.setattr(
        "neurolang.utils.llm.context.declared_predicates",
        lambda name: declared,
    )
    context = EngineContext(nl, "neurosynth")
    assert (
        "- mentions(study_id, term) — Studies and terms [stored, 4 rows]"
        in context.schema_summary()
    )
    assert "matching 'pain'" in context.search_values(
        "mentions", "term", "pain"
    )


def test_describe_and_search(nl):
    context = EngineContext(nl)
    description = context.describe("mentions", rows=2)
    assert description.startswith("mentions(c0, c1) [stored, 4 rows]")
    assert "4 rows, e.g.:" in description
    assert "Did you mean mentions" in context.describe("mention")
    assert context.search_values("mentions", "c1", "PAIN") == (
        "Values of mentions.c1 matching 'PAIN': 'pain', 'chronic pain'"
    )
    assert "closest are: 'memory'" in context.search_values(
        "mentions", "c1", "memroy"
    )
    assert "no column 'term'" in context.search_values(
        "mentions", "term", "x"
    )
    assert "matches" in context.search_values("mentions", "c1", "zzzzzz")


def test_rows_of_a_rule(nl):
    nl.execute_datalog_program("pain_study(s) :- mentions(s, 'pain')")
    context = EngineContext(nl)
    frame = context.frame("pain_study")
    assert list(frame.columns) == ["s"] and list(frame["s"]) == [1]
    with pytest.raises(ValueError, match="Unknown predicate"):
        context.frame("nothing")
