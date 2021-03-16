class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    pass


class UnexpectedExpressionError(NeuroLangException):
    pass


class NeuroLangNotImplementedError(NeuroLangException):
    pass


class ForbiddenExpressionError(NeuroLangException):
    pass


class ForbiddenDisjunctionError(ForbiddenExpressionError):
    pass


class ForbiddenExistentialError(ForbiddenExpressionError):
    pass


class RelationalAlgebraError(NeuroLangException):
    pass


class ProjectionOverMissingColumnsError(RelationalAlgebraError):
    pass


class RelationalAlgebraNotImplementedError(
    RelationalAlgebraError, NotImplementedError
):
    pass


class ForbiddenBuiltinError(ForbiddenExpressionError):
    pass


class NeuroLangFrontendException(NeuroLangException):
    pass


class SymbolNotFoundError(NeuroLangException):
    pass


class RuleNotFoundError(NeuroLangException):
    pass


class UnsupportedProgramError(NeuroLangException):
    pass


class UnsupportedQueryError(NeuroLangException):
    pass


class UnsupportedSolverError(NeuroLangException):
    pass


class ProtectedKeywordError(NeuroLangException):
    pass


class ForbiddenRecursivityError(UnsupportedProgramError):
    pass


class ForbiddenUnstratifiedAggregation(UnsupportedProgramError):
    pass


class WrongArgumentsInPredicateError(NeuroLangException):
    pass


class TranslateToNamedRAException(NeuroLangException):
    pass


class NoValidChaseClassForStratumException(NeuroLangException):
    pass


class CouldNotTranslateConjunctionException(TranslateToNamedRAException):
    def __init__(self, output):
        super().__init__(f"Could not translate conjunction: {output}")
        self.output = output


class NegativeFormulaNotSafeRangeException(TranslateToNamedRAException):
    def __init__(self, formula):
        super().__init__(f"Negative predicate {formula} is not safe range")
        self.formula = formula


class NegativeFormulaNotNamedRelationException(TranslateToNamedRAException):
    def __init__(self, formula):
        super().__init__(f"Negative formula {formula} is not a named relation")
        self.formula = formula


class NonLiftableException(NeuroLangException):
    pass
