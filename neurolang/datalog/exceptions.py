from ..exceptions import ForbiddenExpressionError


class AggregatedVariableReplacedByConstantError(ForbiddenExpressionError):
    pass


class BoundAggregationApplicationError(ForbiddenExpressionError):
    pass


class NegationInMagicSetsRewriteError(ForbiddenExpressionError):
    pass


class NonConjunctiveAntecedentInMagicSetsError(ForbiddenExpressionError):
    pass