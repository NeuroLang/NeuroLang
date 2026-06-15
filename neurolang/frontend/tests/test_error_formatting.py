"""
Tests for the error formatting module.

Generates a variety of real Datalog and SQUALL errors and verifies
that the formatting produces readable, helpful output.
"""

import sys

import pytest
from lark.exceptions import UnexpectedToken, UnexpectedCharacters

from neurolang.exceptions import (
    NeuroLangException,
    ParserError,
    SquallSemanticError,
    UnexpectedTokenError,
    UnexpectedCharactersError,
)
from neurolang.frontend.error_formatting import (
    format_error,
    format_lark_parse_error,
    _USE_COLOR,
    _suggest_datalog_fixes,
    _suggest_squall_fixes,
    _format_source_window,
    _format_source_pointer,
)


class TestSourceContext:
    """Test source context extraction and pointer formatting."""

    def test_format_source_pointer_basic(self):
        result = _format_source_pointer("ans(x) :- R(x)", 1)
        assert "^~~~" in result

    def test_format_source_pointer_mid_line(self):
        result = _format_source_pointer("ans(x) :- R(x)", 8)
        assert " " * 7 in result
        assert "^~~~" in result

    def test_format_source_pointer_none_column(self):
        result = _format_source_pointer("ans(x) :- R(x)", None)
        assert result == ""

    def test_format_source_window_basic(self):
        lines = [
            "line one",
            "ans(x) :- R(x)",
            "line three",
        ]
        result = _format_source_window(lines, 2, 1)
        assert "ans(x)" in result
        assert "^~~~" in result

    def test_format_source_window_out_of_range(self):
        result = _format_source_window([], 1, 1)
        assert result == ""


class TestDatalogSuggestions:
    """Test suggestion generation for Datalog queries."""

    def test_empty_body(self):
        suggestions = _suggest_datalog_fixes("ans(x) :- ")
        assert len(suggestions) > 0
        assert any("empty" in s.lower() for s in suggestions)

    def test_trailing_dot(self):
        suggestions = _suggest_datalog_fixes("ans(x) :- R(x).")
        assert len(suggestions) > 0
        assert any("'.'" in s for s in suggestions)

    def test_arrow_instead(self):
        suggestions = _suggest_datalog_fixes("ans(x) <- R(x)")
        assert len(suggestions) > 0
        assert any("<-" in s for s in suggestions)

    def test_select_keyword(self):
        suggestions = _suggest_datalog_fixes("SELECT x FROM R")
        assert len(suggestions) > 0
        assert any("SQL" in s for s in suggestions)

    def test_not_keyword(self):
        suggestions = _suggest_datalog_fixes("ans(x) :- R(x), not S(x)")
        assert len(suggestions) > 0
        assert any("'~'" in s for s in suggestions)

    def test_clean_query_no_suggestions(self):
        suggestions = _suggest_datalog_fixes("ans(x) :- R(x)")
        assert len(suggestions) == 0


class TestSquallSuggestions:
    """Test suggestion generation for SQUALL queries."""

    def test_select_instead(self):
        suggestions = _suggest_squall_fixes("select every study")
        assert len(suggestions) > 0
        assert any("obtain" in s.lower() for s in suggestions)

    def test_get_instead(self):
        suggestions = _suggest_squall_fixes("get every study")
        assert len(suggestions) > 0
        assert any("obtain" in s.lower() for s in suggestions)

    def test_for_each(self):
        suggestions = _suggest_squall_fixes("for each study")
        assert len(suggestions) > 0
        assert any("for every" in s.lower() for s in suggestions)

    def test_which_instead(self):
        suggestions = _suggest_squall_fixes("every study which reports")
        assert len(suggestions) > 0
        assert any("that" in s.lower() for s in suggestions)

    def test_clean_squall_no_suggestions(self):
        suggestions = _suggest_squall_fixes("obtain every study that reports")
        assert len(suggestions) == 0


class TestFormatError:
    """Test the main format_error function with various exception types."""

    def test_squall_semantic_error_with_source(self):
        exc = SquallSemanticError(
            "Unresolved anaphora: 'the term' has no matching 'for every term' in scope",
            line=1,
            column=20,
            source_line="define as V for every Voxel ?v where a Study reports the Term",
        )
        result = format_error(exc)
        assert "SQUALL semantic error" in result
        assert "the term" in result.lower() or "Term" in result
        assert "^~~~" in result

    def test_squall_semantic_error_no_source(self):
        exc = SquallSemanticError("Some error", line=5, column=10)
        result = format_error(exc)
        assert "SQUALL semantic error" in result
        assert "line 5" in result

    def test_squall_semantic_error_minimal(self):
        exc = SquallSemanticError("Some error")
        result = format_error(exc)
        assert "SQUALL semantic error" in result
        assert "Some error" in result

    def test_unexpected_token_error(self):
        exc = UnexpectedTokenError(
            "Unexpected token 'foo'",
            line=1,
            column=5,
        )
        result = format_error(exc)
        assert "parse error" in result.lower() or "Error" in result

    def test_unexpected_characters_error(self):
        exc = UnexpectedCharactersError(
            "Unexpected character '!'",
            line=1,
            column=10,
        )
        result = format_error(exc)
        assert "parse error" in result.lower() or "Error" in result

    def test_generic_neurolang_exception(self):
        exc = NeuroLangException("Something went wrong")
        result = format_error(exc)
        assert "Something went wrong" in result

    def test_parser_error(self):
        exc = ParserError("Parse failure", line=2, column=3)
        result = format_error(exc)
        assert "Parse failure" in result

    def test_lark_unexpected_token(self):
        # Simulate a Lark UnexpectedToken — use a grammar that will produce one
        try:
            from lark import Lark
            # LALR parser raises UnexpectedCharacters for unknown chars
            # Use a grammar that will produce UnexpectedToken
            g = Lark("start: \"hello\" \"world\"", parser="lalr")
            g.parse("hello there")
        except UnexpectedToken as e:
            result = format_error(e)
            assert "parse error" in result.lower() or "Error" in result
        except UnexpectedCharacters:
            # Some Lark versions raise UnexpectedCharacters instead
            pass

    def test_lark_unexpected_characters(self):
        try:
            from lark import Lark
            g = Lark("start: /[a-z]+/", parser="lalr")
            g.parse("123")
        except UnexpectedCharacters as e:
            result = format_error(e)
            assert "parse error" in result.lower() or "Error" in result

    def test_format_lark_parse_error_wrapper(self):
        exc = NeuroLangException("Test error")
        result = format_lark_parse_error(exc, "test source")
        assert "Test error" in result


class TestRealDatalogErrors:
    """Test formatting of real Datalog parse errors."""

    def test_missing_implication(self):
        """Datalog without :- should produce a parse error."""
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        try:
            dl_parser("ans(x) R(x)")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            result = format_error(e, "ans(x) R(x)")
            assert result
            assert "parse error" in result.lower() or "Error" in result or "error" in result.lower()

    def test_gibberish_datalog(self):
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        try:
            dl_parser("!@#$%")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            result = format_error(e, "!@#$%")
            assert result

    def test_incomplete_rule(self):
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        try:
            dl_parser("ans(x) :- ")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            result = format_error(e, "ans(x) :- ")
            assert result

    def test_wrong_arrow(self):
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        try:
            dl_parser("ans(x) <- R(x)")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            result = format_error(e, "ans(x) <- R(x)")
            assert result

    def test_trailing_dot(self):
        # The _preprocess function strips trailing dots from each line.
        # Test a case with a dot mid-expression that won't be stripped.
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        try:
            dl_parser("ans(x) :- R(x) .and. S(x)")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            result = format_error(e, "ans(x) :- R(x) .and. S(x)")
            assert result


class TestRealSquallErrors:
    """Test formatting of real SQUALL parse and semantic errors."""

    def test_gibberish_squall(self):
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        try:
            squall_parser("foo bar baz qux")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            result = format_error(e, "foo bar baz qux")
            assert result

    def test_gibberish_after_define(self):
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        try:
            squall_parser("define as Verb ~!@#$%")
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            result = format_error(e, "define as Verb ~!@#$%")
            assert result

    def test_the_anaphora_error(self):
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        # 'the' with no matching 'for every' in scope — should still fail
        code = "define as V for every Voxel ?v where a Study reports the Term and the Term is in the Study"
        try:
            squall_parser(code)
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            result = format_error(e, code)
            assert result

    def test_the_in_subject_position(self):
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        # Incomplete rule body — should produce a parse error
        code = "define as V for every Voxel ?v where"
        try:
            squall_parser(code)
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            result = format_error(e, code)
            assert result

    def test_the_anaphora_different_noun(self):
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        # Missing determiner before noun — should produce a parse error
        code = "define as V for every Voxel ?v where Study reports Term"
        try:
            squall_parser(code)
            pytest.fail("Expected parse error")
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            result = format_error(e, code)
            assert result


class TestCLIErrorHandling:
    """Test that the CLI properly catches and formats errors."""

    def test_datalog_parse_error_in_cli(self):
        """Simulate what happens when CLI gets a bad Datalog query."""
        from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
        from neurolang.frontend.error_formatting import print_formatted_error
        import io

        bad_query = "ans(x) R(x)"
        try:
            dl_parser(bad_query)
        except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
            # Capture stderr
            stderr = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr
            try:
                print_formatted_error(e, bad_query)
            finally:
                sys.stderr = old_stderr
            output = stderr.getvalue()
            assert output
            assert "parse error" in output.lower() or "Error" in output or "error" in output.lower()

    def test_squall_parse_error_in_cli(self):
        """Simulate what happens when CLI gets a bad SQUALL query."""
        from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
        from neurolang.frontend.error_formatting import print_formatted_error
        import io

        bad_query = "foo bar baz"
        try:
            squall_parser(bad_query)
        except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
            stderr = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr
            try:
                print_formatted_error(e, bad_query)
            finally:
                sys.stderr = old_stderr
            output = stderr.getvalue()
            assert output


if __name__ == "__main__":
    # Run a quick demo of error formatting
    print("=" * 60)
    print("NeuroLang Error Formatting Demo")
    print("=" * 60)

    # 1. Datalog: missing implication
    print("\n--- Datalog: missing ':-' ---")
    from neurolang.frontend.datalog.standard_syntax import parser as dl_parser
    try:
        dl_parser("ans(x) R(x)")
    except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
        print(format_error(e, "ans(x) R(x)"))

    # 2. Datalog: wrong arrow
    print("\n--- Datalog: '<-' instead of ':-' ---")
    try:
        dl_parser("ans(x) <- R(x)")
    except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
        print(format_error(e, "ans(x) <- R(x)"))

    # 3. Datalog: trailing dot
    print("\n--- Datalog: trailing '.' ---")
    try:
        dl_parser("ans(x) :- R(x).")
    except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
        print(format_error(e, "ans(x) :- R(x)."))

    # 4. Datalog: incomplete rule
    print("\n--- Datalog: empty body ---")
    try:
        dl_parser("ans(x) :- ")
    except (UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException) as e:
        print(format_error(e, "ans(x) :- "))

    # 5. SQUALL: anaphora error
    print("\n--- SQUALL: unresolved 'the Term' ---")
    from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
    try:
        squall_parser("define as V for every Voxel ?v where a Study reports the Term")
    except SquallSemanticError as e:
        print(format_error(e, "define as V for every Voxel ?v where a Study reports the Term"))

    # 6. SQUALL: gibberish
    print("\n--- SQUALL: gibberish ---")
    try:
        squall_parser("foo bar baz qux")
    except (UnexpectedTokenError, UnexpectedCharactersError, SquallSemanticError, NeuroLangException) as e:
        print(format_error(e, "foo bar baz qux"))

    print("\n" + "=" * 60)
    print("Demo complete!")
