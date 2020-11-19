from neurolang.datalog.negation import is_conjunctive_negation

from ..exceptions import (
    NeuroLangException,
    UnexpectedExpressionError,
    UnsupportedQueryError,
    UnsupportedSolverError
)


class DistributionDoesNotSumToOneError(NeuroLangException):
    pass


class MalformedProbabilisticTupleError(NeuroLangException):
    pass


class NotHierarchicalQueryException(NeuroLangException):
    pass


class UncomparableDistributionsError(NeuroLangException):
    pass


class NotEasilyShatterableError(UnsupportedSolverError):
    pass


class UnsupportedProbabilisticQueryError(UnsupportedQueryError):
    pass


class ForbiddenConditionalQueryNoProb(UnsupportedQueryError):
    pass


class ForbiddenConditionalQueryNonConjunctive(UnsupportedQueryError):
    pass
