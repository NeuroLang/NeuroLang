from ..exceptions import ForbiddenExpressionError


class AggregatedVariableReplacedByConstantError(ForbiddenExpressionError):
    pass

class BoundAggregationApplicationError(ForbiddenExpressionError):
    pass
