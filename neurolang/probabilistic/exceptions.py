from ..exceptions import (
    NeuroLangException,
    UnsupportedQueryError,
    UnsupportedSolverError,
)


class DistributionDoesNotSumToOneError(NeuroLangException):
    _suggestion = (
        "Probability distribution values must sum to 1."
    )


class MalformedProbabilisticTupleError(NeuroLangException):
    _suggestion = (
        "Check the format of probabilistic tuples."
    )


class NotHierarchicalQueryException(UnsupportedSolverError):
    _suggestion = (
        "The query is not in hierarchical form."
    )


class UncomparableDistributionsError(NeuroLangException):
    _suggestion = (
        "Cannot compare these probability distributions."
    )


class NotEasilyShatterableError(UnsupportedSolverError):
    _suggestion = (
        "The probabilistic program could not be shattered."
    )


class UnsupportedProbabilisticQueryError(UnsupportedQueryError):
    _suggestion = (
        "Probabilistic query type not supported."
    )


class ForbiddenConditionalQueryNoProb(UnsupportedQueryError):
    _suggestion = (
        "Conditional query requires a probabilistic predicate."
    )


class ForbiddenConditionalQueryNonConjunctive(UnsupportedQueryError):
    _suggestion = (
        "Conditional query must be conjunctive."
    )


class RepeatedTuplesInProbabilisticRelationError(
    MalformedProbabilisticTupleError
):
    _suggestion = (
        "Check the format of probabilistic tuples. "
        "Duplicate tuples are not allowed."
    )

    def __init__(self, n_repeated_tuples, n_tuples, message):
        self.n_repeated_tuples = n_repeated_tuples
        self.n_tuples = n_tuples
        self.message = message
        super().__init__(self.message)
