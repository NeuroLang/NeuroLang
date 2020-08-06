class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    pass


class NeuroLangFrontendException(NeuroLangException):
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


class ProtectedKeywordError(NeuroLangException):
    pass
