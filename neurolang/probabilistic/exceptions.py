from ..exceptions import NeuroLangException


class DistributionDoesNotSumToOneError(NeuroLangException):
    pass


class MalformedProbabilisticTupleError(NeuroLangException):
    pass


class NotHierarchicalQueryException(NeuroLangException):
    pass


class UncomparableDistributionsError(NeuroLangException):
    pass
