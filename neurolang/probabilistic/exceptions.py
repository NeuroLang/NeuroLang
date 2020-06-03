from ..exceptions import NeuroLangException


class DistributionDoesNotSumToOneError(NeuroLangException):
    pass


class MalformedProbabilisticTupleError(NeuroLangException):
    pass


class UncomparableDistributionsError(NeuroLangException):
    pass
