"""Tests for structured error reporting in NeuroLang exceptions."""

import pickle

import pytest

from neurolang.exceptions import (
    CouldNotTranslateConjunctionException,
    ForbiddenDisjunctionError,
    ForbiddenExistentialError,
    ForbiddenRecursivityError,
    ForbiddenUnstratifiedAggregation,
    InvalidCommandExpression,
    NegativeFormulaNotNamedRelationException,
    NegativeFormulaNotSafeRangeException,
    NeuroLangException,
    NoValidChaseClassForStratumException,
    NotConjunctiveExpression,
    NotConjunctiveExpressionNegation,
    NotConjunctiveExpressionNestedPredicates,
    NotInFONegE,
    ParserError,
    ProjectionOverMissingColumnsError,
    ProtectedKeywordError,
    RelationalAlgebraNotImplementedError,
    SquallSemanticError,
    SymbolNotFoundError,
    UnsupportedProgramError,
    UnsupportedQueryError,
    UnsupportedSolverError,
    WrongArgumentsInPredicateError,
)
from neurolang.datalog.exceptions import (
    AggregatedVariableReplacedByConstantError,
    BoundAggregationApplicationError,
    InvalidMagicSetError,
    NegationInMagicSetsRewriteError,
    NoConstantPredicateFoundError,
    NonConjunctiveAntecedentInMagicSetsError,
)
from neurolang.probabilistic.exceptions import (
    DistributionDoesNotSumToOneError,
    ForbiddenConditionalQueryNonConjunctive,
    ForbiddenConditionalQueryNoProb,
    MalformedProbabilisticTupleError,
    NotEasilyShatterableError,
    NotHierarchicalQueryException,
    RepeatedTuplesInProbabilisticRelationError,
    UncomparableDistributionsError,
    UnsupportedProbabilisticQueryError,
)
from neurolang.utils.error_enrichment import enrich_exception, format_user_error


class TestNeuroLangException:
    """Tests for the base NeuroLangException enhancements."""

    def test_base_suggestion(self):
        exc = NeuroLangException("test error")
        assert exc.suggestion == "Check the query for errors."

    def test_error_summary_shape(self):
        exc = NeuroLangException("test error")
        summary = exc.error_summary()
        assert isinstance(summary, dict)
        assert "short_message" in summary
        assert "detail" in summary
        assert "suggestion" in summary
        assert "exception_type" in summary
        assert "line" in summary
        assert "column" in summary
        assert "source_line" in summary

    def test_error_summary_values(self):
        exc = NeuroLangException("test error")
        summary = exc.error_summary()
        assert summary["short_message"] == "test error"
        assert "Base class" in summary["detail"]
        assert summary["suggestion"] == "Check the query for errors."
        assert summary["exception_type"] == "NeuroLangException"
        assert summary["line"] is None
        assert summary["column"] is None
        assert summary["source_line"] is None

    def test_error_summary_docstring_detail(self):
        """Detail should come from __doc__ when available."""
        exc = SymbolNotFoundError("unknown symbol")
        summary = exc.error_summary()
        assert "symbol" in summary["detail"].lower()
        # docstring wraps at 79 chars creating indentation artifacts
        assert "previously" in summary["detail"]
        assert "defined" in summary["detail"]

    def test_empty_message_error_summary(self):
        """When str(exc) is empty, short_message falls back to class name."""
        exc = NeuroLangException()
        summary = exc.error_summary()
        assert summary["short_message"] == "NeuroLangException"

    def test_short_message_is_first_line(self):
        """short_message should be the first line of the exception string."""
        exc = NeuroLangException("line one\nline two\nline three")
        summary = exc.error_summary()
        assert summary["short_message"] == "line one"

    def test_line_column_source_line_defaults(self):
        exc = NeuroLangException()
        assert exc.line is None
        assert exc.column is None
        assert exc.source_line is None

    def test_pickle_roundtrip(self):
        """NeuroLangException should be picklable (for serialization)."""
        exc = NeuroLangException("test error")
        exc.line = 1
        exc.column = 5
        exc.source_line = "test query"
        restored = pickle.loads(pickle.dumps(exc))
        assert str(restored) == "test error"
        assert restored.line == 1
        assert restored.column == 5


class TestConcreteSuggestions:
    """Test that each exception class has a meaningful suggestion."""

    def test_symbol_not_found(self):
        exc = SymbolNotFoundError("unknown")
        assert "defined" in exc.suggestion.lower()
        assert "spelling" in exc.suggestion.lower()

    def test_wrong_arguments(self):
        exc = WrongArgumentsInPredicateError()
        assert "arguments" in exc.suggestion.lower()

    def test_forbidden_disjunction(self):
        exc = ForbiddenDisjunctionError()
        assert "disjunction" in exc.suggestion.lower()

    def test_forbidden_recursivity(self):
        exc = ForbiddenRecursivityError()
        assert "circular" in exc.suggestion.lower()

    def test_forbidden_unstratified_aggregation(self):
        exc = ForbiddenUnstratifiedAggregation()
        assert "stratum" in exc.suggestion.lower()

    def test_forbidden_existential(self):
        exc = ForbiddenExistentialError()
        assert "existential" in exc.suggestion.lower()

    def test_protected_keyword(self):
        exc = ProtectedKeywordError()
        assert "keyword" in exc.suggestion.lower()

    def test_not_conjunctive_expression(self):
        exc = NotConjunctiveExpression()
        assert "conjunction" in exc.suggestion.lower()

    def test_not_conjunctive_negation(self):
        exc = NotConjunctiveExpressionNegation()
        assert "negation" in exc.suggestion.lower()
        assert "aggregate" in exc.suggestion.lower()

    def test_not_conjunctive_nested(self):
        exc = NotConjunctiveExpressionNestedPredicates()
        assert "nested" in exc.suggestion.lower()

    def test_negative_formula_not_safe_range(self):
        exc = NegativeFormulaNotSafeRangeException("p(x)")
        assert "non-negated" in exc.suggestion.lower()

    def test_negative_formula_not_named(self):
        exc = NegativeFormulaNotNamedRelationException("p(x)")
        assert "non-negated" in exc.suggestion.lower()

    def test_projection_over_missing_columns(self):
        exc = ProjectionOverMissingColumnsError()
        assert "signature" in exc.suggestion.lower()

    def test_relational_algebra_not_implemented(self):
        exc = RelationalAlgebraNotImplementedError()
        assert "malformed" in exc.suggestion.lower()

    def test_no_valid_chase_class(self):
        exc = NoValidChaseClassForStratumException()
        assert "solver" in exc.suggestion.lower()

    def test_unsupported_program(self):
        exc = UnsupportedProgramError()
        assert "not supported" in exc.suggestion.lower()

    def test_unsupported_query(self):
        exc = UnsupportedQueryError()
        assert "not supported" in exc.suggestion.lower()

    def test_unsupported_solver(self):
        exc = UnsupportedSolverError()
        assert "not supported" in exc.suggestion.lower()

    def test_not_in_foneg(self):
        exc = NotInFONegE()
        assert "negation" in exc.suggestion.lower()

    def test_invalid_command_expression(self):
        exc = InvalidCommandExpression()
        assert "syntax" in exc.suggestion.lower()

    def test_squall_semantic_error(self):
        exc = SquallSemanticError("test")
        assert "squall" in exc.suggestion.lower()

    def test_invalid_magic_set(self):
        exc = InvalidMagicSetError()
        assert "magic" in exc.suggestion.lower()

    def test_bound_aggregation(self):
        exc = BoundAggregationApplicationError()
        assert "aggregate" in exc.suggestion.lower()

    def test_negation_in_magic_sets(self):
        exc = NegationInMagicSetsRewriteError()
        assert "negation" in exc.suggestion.lower()

    def test_non_conjunctive_antecedent(self):
        exc = NonConjunctiveAntecedentInMagicSetsError()
        assert "conjunctive" in exc.suggestion.lower()

    def test_no_constant_predicate(self):
        exc = NoConstantPredicateFoundError()
        assert "constant" in exc.suggestion.lower()

    def test_aggregated_variable_replaced(self):
        exc = AggregatedVariableReplacedByConstantError()
        assert exc.suggestion  # inherits from InvalidMagicSetError


class TestProbabilisticSuggestions:
    """Test probabilistic exception suggestions."""

    def test_distribution_sum(self):
        exc = DistributionDoesNotSumToOneError()
        assert "sum" in exc.suggestion.lower()

    def test_malformed_tuple(self):
        exc = MalformedProbabilisticTupleError()
        assert "tuple" in exc.suggestion.lower()

    def test_not_hierarchical(self):
        exc = NotHierarchicalQueryException()
        assert "hierarchical" in exc.suggestion.lower()

    def test_uncomparable_distributions(self):
        exc = UncomparableDistributionsError()
        assert "distribution" in exc.suggestion.lower()

    def test_not_easily_shatterable(self):
        exc = NotEasilyShatterableError()
        assert "shatter" in exc.suggestion.lower()

    def test_unsupported_probabilistic_query(self):
        exc = UnsupportedProbabilisticQueryError()
        assert "not supported" in exc.suggestion.lower()

    def test_forbidden_conditional_no_prob(self):
        exc = ForbiddenConditionalQueryNoProb()
        assert "probabilistic" in exc.suggestion.lower()

    def test_forbidden_conditional_non_conjunctive(self):
        exc = ForbiddenConditionalQueryNonConjunctive()
        assert "conjunctive" in exc.suggestion.lower()

    def test_repeated_tuples(self):
        exc = RepeatedTuplesInProbabilisticRelationError(2, 5, "dup")
        assert "tuple" in exc.suggestion.lower()


class TestParserError:
    """ParserError already has line/column — ensure compatibility."""

    def test_parser_error_line_column(self):
        exc = ParserError("parse error", line=5, column=10)
        assert exc.line == 5
        assert exc.column == 10
        summary = exc.error_summary()
        assert summary["line"] == 5
        assert summary["column"] == 10

    def test_parser_error_inherits_suggestion(self):
        exc = ParserError("parse error")
        assert exc.suggestion == "Check the query for errors."

    def test_unexpected_token_error(self):
        exc = ParserError("unexpected token")
        assert exc.suggestion == "Check the query for errors."


class TestEnrichException:
    """Tests for the enrich_exception utility."""

    def test_enrich_neurolang_exception(self):
        exc = SymbolNotFoundError("foo")
        result = enrich_exception(
            exc,
            query_text="ans(x) :- foo(x)",
            engine_type="datalog",
            line=1,
            column=5,
            source_line="ans(x) :- foo(x)",
        )
        assert result is exc  # same object
        assert result.line == 1
        assert result.column == 5
        assert result.source_line == "ans(x) :- foo(x)"
        assert result.query_text == "ans(x) :- foo(x)"  # type: ignore
        assert result.engine_type == "datalog"  # type: ignore

    def test_enrich_preserves_existing_context(self):
        """If the exception already has line/col, enrich_exception should
        not overwrite them."""
        exc = ParserError("parse error", line=5, column=10)
        enrich_exception(exc, line=999, column=999)
        assert exc.line == 5  # unchanged
        assert exc.column == 10  # unchanged

    def test_enrich_non_neurolang_exception(self):
        """Regular exceptions should still get query_text and engine_type."""
        exc = ValueError("some error")
        result = enrich_exception(
            exc,
            query_text="bad query",
            engine_type="datalog",
        )
        assert result is exc
        assert result.query_text == "bad query"  # type: ignore
        assert result.engine_type == "datalog"  # type: ignore

    def test_format_user_error_with_error_summary(self):
        exc = SymbolNotFoundError("foo")
        msg = format_user_error(exc)
        assert "foo" in msg
        assert "Suggestion:" in msg

    def test_format_user_error_fallback(self):
        exc = ValueError("plain error")
        msg = format_user_error(exc)
        assert msg == "plain error"

    def test_enrich_squall_semantic_error(self):
        exc = SquallSemanticError(
            "test message",
            line=3,
            column=7,
            source_line="obtain every Person.",
        )
        enrich_exception(exc, query_text="obtain every Person.")
        assert exc.line == 3
        assert exc.column == 7
        assert exc.source_line == "obtain every Person."


class TestErrorSummaryForSpecialCases:
    """Exceptions with custom __init__ that also set attributes."""

    def test_could_not_translate_conjunction(self):
        exc = CouldNotTranslateConjunctionException("some formula")
        summary = exc.error_summary()
        assert exc.output == "some formula"
        assert "conjunctive" in exc.suggestion.lower()
        assert "Could not translate conjunction" in summary["short_message"]

    def test_negative_formula_not_safe_range_with_formula(self):
        exc = NegativeFormulaNotSafeRangeException("p(x)")
        summary = exc.error_summary()
        assert exc.formula == "p(x)"
        assert "non-negated" in exc.suggestion.lower()

    def test_negative_formula_not_named_relation_with_formula(self):
        exc = NegativeFormulaNotNamedRelationException("p(x)")
        summary = exc.error_summary()
        assert exc.formula == "p(x)"
        assert "non-negated" in exc.suggestion.lower()

    def test_repeated_tuples_with_counts(self):
        exc = RepeatedTuplesInProbabilisticRelationError(
            5, 100, "5 repeated tuples"
        )
        assert exc.n_repeated_tuples == 5
        assert exc.n_tuples == 100


class TestPickleRoundtrip:
    """Ensure all key exception classes can be pickled."""

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: WrongArgumentsInPredicateError(),
            lambda: ForbiddenDisjunctionError(),
            lambda: ForbiddenRecursivityError(),
            lambda: ForbiddenUnstratifiedAggregation(),
            lambda: SymbolNotFoundError("test"),
            lambda: ProtectedKeywordError(),
            lambda: NoValidChaseClassForStratumException(),
            lambda: CouldNotTranslateConjunctionException("test"),
            lambda: NegativeFormulaNotSafeRangeException("p(x)"),
            lambda: NegativeFormulaNotNamedRelationException("p(x)"),
            lambda: SquallSemanticError("test"),
            lambda: ParserError("test", line=1, column=2),
            lambda: DistributionDoesNotSumToOneError(),
            lambda: InvalidMagicSetError(),
            lambda: UnsupportedProbabilisticQueryError(),
        ],
    )
    def test_pickle(self, exc_factory):
        exc = exc_factory()
        restored = pickle.loads(pickle.dumps(exc))
        assert type(restored) is type(exc)
        assert restored.suggestion == exc.suggestion
