"""
Tests for the error formatting module.

Real Datalog and SQUALL errors are produced with the actual parsers and the
formatted output is checked for location, hints and suggestions.
"""

import io

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

from ... import exceptions
from ...exceptions import (
    CouldNotTranslateConjunctionException,
    NeuroLangException,
    ParserError,
    SquallSemanticError,
    SymbolNotFoundError,
    WrongArgumentsInPredicateError,
)
from ..datalog.squall_syntax_lark import parser as squall_parser
from ..datalog.standard_syntax import parser as datalog_parser
from ..error_formatting import format_error, print_formatted_error, use_color

PARSERS = {"Datalog": datalog_parser, "SQUALL": squall_parser}


def _format_parse_error(code, language):
    """Parse *code*, expect a failure and return the formatted error."""
    with pytest.raises(NeuroLangException) as exc_info:
        PARSERS[language](code)
    return format_error(exc_info.value, code, language=language)


# ── Syntax hints ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "language, wrong, hint, fixed",
    [
        ("Datalog", "ans(x) :- ", "body is empty", "ans(x) :- R(x)"),
        ("Datalog", "ans(x) <- R(x)", "instead of '<-'", "ans(x) :- R(x)"),
        ("Datalog", "ans(x) <- R(x)", "instead of '<-'", "ans(x) ← R(x)"),
        ("Datalog", "R(x) => ans(x)", "head first", "ans(x) :- R(x)"),
        ("Datalog", "SELECT x FROM R", "not SQL", "ans(x) :- R(x)"),
        (
            "Datalog", "ans(x) :- R(x) WHERE x > 0", "with WHERE",
            "ans(x) :- R(x), (x > 0)",
        ),
        (
            "Datalog", "ans(x) :- R(x), not S(x)", "Use '~'",
            "ans(x) :- R(x), ~S(x)",
        ),
        (
            "SQUALL", "select every item.", "instead of 'select'",
            "obtain every item that plays.",
        ),
        (
            "SQUALL", "get every item.", "instead of 'get'",
            "obtain every item that plays.",
        ),
        (
            "SQUALL", "find every item that plays.", "instead of 'find'",
            "obtain every item that plays.",
        ),
    ],
)
def test_hint_shown_and_suggested_fix_parses(language, wrong, hint, fixed):
    out = _format_parse_error(wrong, language)
    assert out.startswith(f"{language} parse error")
    assert "Suggestions:" in out
    assert hint in out
    PARSERS[language](fixed)


@pytest.mark.parametrize(
    "language, code",
    [
        ("Datalog", "ans(x) :- R(x)."),
        ("SQUALL", "obtain every item which plays."),
        (
            "SQUALL",
            "define as Voxel_reported with a probability of the "
            "Proximity_indicator of the Peak_reported (?x2; ?y2; ?z2; ?s) "
            "for each ?x, ?y, ?z and for each ?s "
            "where (?x; ?y; ?z) is a Voxel.",
        ),
    ],
)
def test_valid_syntax_formerly_flagged_by_hints(language, code):
    """Trailing dots, 'which' and 'for each' are valid syntax."""
    PARSERS[language](code)


def test_hints_only_match_the_error_line():
    code = "A(x) :- R(x) % not where\nans(x) <- A(x)"
    out = _format_parse_error(code, "Datalog")
    assert "instead of '<-'" in out
    assert "Use '~'" not in out
    assert "with WHERE" not in out


def test_squall_errors_get_squall_hints_only():
    out = _format_parse_error("get every item.", "SQUALL")
    assert out.startswith("SQUALL parse error")
    assert "Datalog" not in out
    assert "':-'" not in out


def test_datalog_identifiers_are_not_taken_for_squall():
    out = _format_parse_error("ans(x) :- defined_region(x) <- R", "Datalog")
    assert out.startswith("Datalog parse error")


# ── Locations ──────────────────────────────────────────────────────────────


def test_caret_on_error_line_of_multiline_program():
    code = "A(x) :- R(x)\nB(x) :- A(x)\nans(x) <- B(x)"
    out = _format_parse_error(code, "Datalog")
    lines = out.splitlines()
    error_line = lines.index("    3 | ans(x) <- B(x)")
    assert lines[error_line + 1] == "      |        ^~~~"
    assert "Location: line 3, column 8" in out


@pytest.mark.parametrize(
    "language, code, location",
    [
        ("Datalog", "\n\nans(x) <- R(x)", "line 3, column 8"),
        ("SQUALL", "\n\nget every item.", "line 3, column 5"),
    ],
)
def test_leading_blank_lines_keep_line_numbers(language, code, location):
    out = _format_parse_error(code, language)
    assert f"Location: {location}" in out
    assert "    3 | " in out


def test_squall_semantic_error_keeps_line_numbers():
    code = "\n\ndefine as V for every Voxel ?v where a Study reports the Term"
    with pytest.raises(SquallSemanticError) as exc_info:
        squall_parser(code)
    out = format_error(exc_info.value, code, language="SQUALL")
    assert out.startswith("SQUALL semantic error")
    assert "    3 | define as V" in out
    assert "Location: line 3" in out


def test_unexpected_end_of_input():
    code = "obtain every item that plays and"
    out = _format_parse_error(code, "SQUALL")
    assert out.startswith("SQUALL parse error — unexpected end of input")
    assert f"Location: line 1, column {len(code) + 1}" in out
    expected = next(
        line for line in out.splitlines() if "Expected:" in line
    )
    assert expected.endswith("…")
    assert len(expected.split(", ")) == 9
    assert "_THAT" not in out
    assert len(out.splitlines()) < 10


def test_expected_terminals_shown_as_literals():
    out = _format_parse_error("ans(x) :- R(x) S(x)", "Datalog")
    assert "Expected: one of '::', ':='" in out
    assert "PROBA_OP" not in out


def test_raw_lark_error_is_formatted():
    with pytest.raises(UnexpectedInput) as exc_info:
        Lark('start: "hello" "world"', parser="lalr").parse("hello there")
    out = format_error(exc_info.value, "hello there")
    assert out.startswith("Datalog parse error — unexpected character")
    assert "Location: line 1, column 6" in out
    assert "Found: ' '" in out


# ── NeuroLang exceptions ───────────────────────────────────────────────────


def test_did_you_mean_uses_known_symbols():
    exc = SymbolNotFoundError("Symbol not found regin_reported")
    out = format_error(
        exc, "ans(x) :- regin_reported(x)",
        known_symbols=["region_reported", "Region_Reported", "peak"],
    )
    assert out.startswith("Undefined symbol")
    assert "Did you mean: region_reported" in out
    assert "Region_Reported" not in out


def test_symbol_not_found_without_close_match_gives_tip():
    exc = SymbolNotFoundError("Symbol xyzzy not found")
    out = format_error(exc, "ans(x) :- xyzzy(x)", known_symbols=["item"])
    assert "Did you mean" not in out
    assert f"Tip: {SymbolNotFoundError.suggestion}" in out


def test_arity_mismatch():
    exc = WrongArgumentsInPredicateError("wrong number of arguments")
    out = format_error(exc, "ans(x) :- R(x, y, z)")
    assert out.startswith("Arity mismatch")
    assert f"Tip: {WrongArgumentsInPredicateError.suggestion}" in out


def test_unbound_variables_tip_names_a_binding_atom():
    exc = exceptions.UnboundVariablesError(
        "reports_close", ("i1", "j1", "k1"),
        ("euclidean(i, j, k, i1, j1, k1) < 2.0",), "voxel(i1, j1, k1)",
    )
    out = format_error(exc, "reports_close(i, j, k, s) :- ...")
    assert out.startswith("Error — UnboundVariablesError")
    assert (
        "Error: In reports_close: the variables i1, j1, k1 appear only in "
        "`euclidean(i, j, k, i1, j1, k1) < 2.0`"
    ) in out
    assert "Tip: Add a predicate over i1, j1, k1 to the rule body, such as " \
        "voxel(i1, j1, k1)" in out
    assert "normal form" not in out


def test_unbound_variables_tip_without_example():
    exc = exceptions.UnboundVariablesError(None, ("m",), ("i < m",))
    out = format_error(exc)
    assert "Error: In a rule body: the variable m appears only in `i < m`" \
        in out
    assert "Tip: Add a predicate over m to the rule body" in out
    assert "equality" in out
    assert "normal form" not in out


def test_could_not_translate_conjunction_keeps_normal_form_tip():
    out = format_error(CouldNotTranslateConjunctionException("f(x)"))
    assert "Tip: Rewrite the formula in conjunctive or disjunctive normal " \
        "form" in out


def test_neurolang_exception_with_location():
    exc = ParserError("Parse failure", line=2, column=3)
    out = format_error(exc, "first\nsecond line")
    assert out.startswith("Error — ParserError")
    assert "    2 | second line\n      |   ^~~~" in out
    assert "Error: Parse failure" in out
    assert "Tip:" not in out


def test_squall_semantic_error_from_source_line_only():
    exc = SquallSemanticError(
        "Unresolved anaphora", line=4, column=3,
        source_line="where the Term",
    )
    out = format_error(exc)
    assert "    4 | where the Term\n      |   ^~~~" in out
    assert "Error: Unresolved anaphora" in out


def test_squall_semantic_error_without_location():
    out = format_error(SquallSemanticError("Some error"))
    assert out == "SQUALL semantic error\n  Error: Some error"


def test_non_neurolang_exception():
    out = format_error(ValueError("boom"))
    assert out == "Internal error — ValueError: boom"


# ── Colour ─────────────────────────────────────────────────────────────────


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_color_only_on_terminals(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert use_color(_Terminal())
    assert not use_color(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not use_color(_Terminal())


@pytest.mark.parametrize(
    "stream_type, no_color, coloured",
    [
        (_Terminal, None, True),
        (_Terminal, "1", False),
        (io.StringIO, None, False),
    ],
)
def test_print_formatted_error_colour(
    monkeypatch, stream_type, no_color, coloured
):
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    stream = stream_type()
    print_formatted_error(NeuroLangException("boom"), file=stream)
    assert ("\033[" in stream.getvalue()) is coloured
    assert "boom" in stream.getvalue()


def test_print_formatted_error_defaults_to_stderr(capsys):
    print_formatted_error(NeuroLangException("boom"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: boom" in captured.err
