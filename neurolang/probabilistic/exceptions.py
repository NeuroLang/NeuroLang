from ..exceptions import NeuroLangException, UnexpectedExpressionError


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
