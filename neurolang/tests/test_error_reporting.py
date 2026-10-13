"""Tests for structured error reporting in NeuroLang exceptions."""

import pickle

import pytest

from neurolang.exceptions import (
    CouldNotTranslateConjunctionException,
    ForbiddenDisjunctionError,
    ForbiddenRecursivityError,
    ForbiddenUnstratifiedAggregation,
    NegativeFormulaNotNamedRelationException,
    NegativeFormulaNotSafeRangeException,
    NeuroLangException,
    NoValidChaseClassForStratumException,
    NotConjunctiveExpressionNestedPredicates,
    ParserError,
    ProtectedKeywordError,
    SquallSemanticError,
    SymbolNotFoundError,
    WrongArgumentsInPredicateError,
)
from neurolang.datalog.exceptions import InvalidMagicSetError
from neurolang.probabilistic.exceptions import (
    DistributionDoesNotSumToOneError,
    RepeatedTuplesInProbabilisticRelationError,
    UnsupportedProbabilisticQueryError,
)


class TestNeuroLangException:
    """Tests for the base NeuroLangException enhancements."""

    def test_base_has_no_suggestion(self):
        assert NeuroLangException("test error").suggestion is None


class TestConcreteSuggestions:
    """Test that each exception class has a meaningful suggestion."""

    def test_symbol_not_found(self):
        exc = SymbolNotFoundError("unknown")
        assert "spelling" in exc.suggestion.lower()

    def test_wrong_arguments(self):
        exc = WrongArgumentsInPredicateError()
        assert "arguments" in exc.suggestion.lower()

    def test_forbidden_disjunction(self):
        exc = ForbiddenDisjunctionError()
        assert "single rule" in exc.suggestion.lower()

    def test_forbidden_recursivity(self):
        exc = ForbiddenRecursivityError()
        assert "cycle" in exc.suggestion.lower()

    def test_forbidden_unstratified_aggregation(self):
        exc = ForbiddenUnstratifiedAggregation()
        assert "aggregation" in exc.suggestion.lower()

    def test_protected_keyword(self):
        exc = ProtectedKeywordError()
        assert "keyword" in exc.suggestion.lower()

    def test_not_conjunctive_nested(self):
        exc = NotConjunctiveExpressionNestedPredicates()
        assert "separate rule" in exc.suggestion.lower()

    def test_negative_formula_not_safe_range(self):
        exc = NegativeFormulaNotSafeRangeException("p(x)")
        assert "non-negated" in exc.suggestion.lower()

    def test_negative_formula_not_named(self):
        exc = NegativeFormulaNotNamedRelationException("p(x)")
        assert "named" in exc.suggestion.lower()


class TestParserError:
    """ParserError already has line/column — ensure compatibility."""

    def test_parser_error_line_column(self):
        exc = ParserError("parse error", line=5, column=10)
        assert exc.line == 5
        assert exc.column == 10


class TestSpecialCases:
    """Exceptions with custom __init__ that also set attributes."""

    def test_could_not_translate_conjunction(self):
        exc = CouldNotTranslateConjunctionException("some formula")
        assert exc.output == "some formula"
        assert "conjunctive" in exc.suggestion.lower()
        assert str(exc).startswith("Could not translate conjunction")

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
