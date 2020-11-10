from neurolang.datalog.negation import is_conjunctive_negation

from ..exceptions import (
    NeuroLangException,
    UnexpectedExpressionError,
    UnsupportedQueryError
)


class DistributionDoesNotSumToOneError(NeuroLangException):
    pass


class MalformedProbabilisticTupleError(NeuroLangException):
    pass


class NotHierarchicalQueryException(NeuroLangException):
    pass


class UncomparableDistributionsError(NeuroLangException):
    pass


class NotEasilyShatterableError(UnexpectedExpressionError):
    pass


class UnsupportedProbabilisticQueryError(UnsupportedQueryError):
    pass


class ForbiddenConditionalQueryNoProb(UnsupportedQueryError):
    pass


class ForbiddenConditionalQueryNonConjunctive(UnsupportedQueryError):
    pass
