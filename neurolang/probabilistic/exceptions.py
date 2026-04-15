from ..exceptions import (
    NeuroLangException,
    UnsupportedQueryError,
    UnsupportedSolverError,
)


class DistributionDoesNotSumToOneError(NeuroLangException):
    pass


class MalformedProbabilisticTupleError(NeuroLangException):
    pass


class NotHierarchicalQueryException(UnsupportedSolverError):
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


class RepeatedTuplesInProbabilisticRelationError(
    MalformedProbabilisticTupleError
):
    def __init__(self, n_repeated_tuples, n_tuples, message):
        self.n_repeated_tuples = n_repeated_tuples
        self.n_tuples = n_tuples
        self.message = message
        super().__init__(self.message)
