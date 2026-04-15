from ..exceptions import ForbiddenExpressionError


class AggregatedVariableReplacedByConstantError(ForbiddenExpressionError):
    pass


class InvalidMagicSetError(ForbiddenExpressionError):
    """
    Generic class for Magic Sets errors. Errors in Magic Sets typically signal
    that the magic sets algorithm cannot be applied to the current datalog
    program. Subclasses should specify the reason why the algorithm cannot be
    applied, for instance if negations are present in the code.
    """
    pass


class BoundAggregationApplicationError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if an argument which is an aggregate
    function is bound.

    Example
    -------
    Q(x, count(y)) :- P(x, y)
    ans(x) :- Q(x, 3)
    """
    pass


class NegationInMagicSetsRewriteError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if negations are present in the code.
    """
    pass


class NonConjunctiveAntecedentInMagicSetsError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if one of the rules has a non
    conjunctive antecedent.
    """
    pass


class NoConstantPredicateFoundError(InvalidMagicSetError):
    """
    Magic Sets algorithm only works if there is at least one predicate in the
    code with a constant as an argument.
    """
    pass