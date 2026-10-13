"""Tests for structured error reporting in NeuroLang exceptions."""

import pickle  # nosec B403 - round-trips exceptions built by the test

import pytest

from ..exceptions import (
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
from ..datalog.exceptions import InvalidMagicSetError
from ..probabilistic.exceptions import (
    DistributionDoesNotSumToOneError,
    RepeatedTuplesInProbabilisticRelationError,
    UnsupportedProbabilisticQueryError,
)


def test_base_exception_has_no_suggestion():
    assert NeuroLangException("test error").suggestion is None


@pytest.mark.parametrize(
    "exc, keyword",
    [
        (SymbolNotFoundError("unknown"), "spelling"),
        (WrongArgumentsInPredicateError(), "number of arguments"),
        (ForbiddenDisjunctionError(), "single rule"),
        (ForbiddenRecursivityError(), "cycle"),
        (ForbiddenUnstratifiedAggregation(), "aggregation"),
        (ProtectedKeywordError(), "reserved"),
        (NotConjunctiveExpressionNestedPredicates(), "separate rule"),
        (CouldNotTranslateConjunctionException("f"), "normal form"),
        (NegativeFormulaNotSafeRangeException("p(x)"), "non-negated"),
        (NegativeFormulaNotNamedRelationException("p(x)"), "own rule"),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v,
)
def test_actionable_suggestion(exc, keyword):
    assert keyword in exc.suggestion


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
            WrongArgumentsInPredicateError,
            ForbiddenDisjunctionError,
            ForbiddenRecursivityError,
            ForbiddenUnstratifiedAggregation,
            lambda: SymbolNotFoundError("test"),
            ProtectedKeywordError,
            NoValidChaseClassForStratumException,
            lambda: CouldNotTranslateConjunctionException("test"),
            lambda: NegativeFormulaNotSafeRangeException("p(x)"),
            lambda: NegativeFormulaNotNamedRelationException("p(x)"),
            lambda: SquallSemanticError("test"),
            lambda: ParserError("test", line=1, column=2),
            DistributionDoesNotSumToOneError,
            InvalidMagicSetError,
            UnsupportedProbabilisticQueryError,
        ],
    )
    def test_pickle(self, exc_factory):
        exc = exc_factory()
        restored = pickle.loads(pickle.dumps(exc))  # nosec B301 - data created here
        assert type(restored) is type(exc)
        assert restored.suggestion == exc.suggestion
