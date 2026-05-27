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

    _suggestion = (
        "The magic sets rewrite could not be applied. "
        "Ensure the program is conjunctive and has at least one "
        "constant argument."
    )


class BoundAggregationApplicationError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if an argument which is an aggregate
    function is bound.

    Examples
    -------
    Q(x, count(y)) :- P(x, y)
    ans(x) :- Q(x, 3)
    """

    _suggestion = (
        "Magic sets cannot be applied when an aggregate function "
        "argument is bound to a constant."
    )


class NegationInMagicSetsRewriteError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if negations are present in the code.
    """

    _suggestion = (
        "Magic sets cannot be applied to programs with negation."
    )


class NonConjunctiveAntecedentInMagicSetsError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if one of the rules has a non
    conjunctive antecedent.
    """

    _suggestion = (
        "All rule antecedents must be conjunctive for magic sets."
    )


class NoConstantPredicateFoundError(InvalidMagicSetError):
    """
    Magic Sets algorithm only works if there is at least one predicate in the
    code with a constant as an argument.
    """

    _suggestion = (
        "Magic sets requires at least one predicate with a constant "
        "argument."
    )