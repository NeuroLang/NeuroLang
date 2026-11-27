from nl_expr.exceptions import ForbiddenExpressionError


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


class PredicateArityInMagicSetsError(InvalidMagicSetError):
    """
    A rule uses an intensional predicate with a number of arguments other
    than the one of its definition. The positions Magic Sets binds, and
    those of an aggregation it leaves free, are those of the definition,
    so the program is solved unrewritten, as without Magic Sets.

    Examples
    -------
    Q(x, count(y)) :- P(x, y)
    ans(x) :- R(a, x), Q(x)
    """


class UnstratifiedMagicSetsRewriteError(InvalidMagicSetError):
    """
    The program of the Magic Sets rewrite has an aggregation, or a
    negation, in a cycle of its dependencies: it would not be stratified,
    though the program that was rewritten is. The magic predicate of a
    predicate gets the bindings of the literals before it in the body, which
    can depend on the aggregate that the predicate is read by.
    """


class BoundAggregationApplicationError(InvalidMagicSetError):
    """
    Magic Sets algorithm does not work if an argument which is an aggregate
    function is bound.

    Examples
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


class NewRecursionInMagicSetsRewriteError(InvalidMagicSetError):
    """
    The program of the Magic Sets rewrite has a cycle of predicates none
    of which is recursive in the program that was rewritten. Every call
    site of a predicate with one adornment shares its adorned and magic
    predicates: ``p^bb`` gets the bindings of a rule that uses it through
    ``magic_p^bb``, whose rule reads them from atoms which may be bound by
    ``p^bb``'s own. Evaluating a non-recursive program in one recursive
    stratum costs more than the bindings save, so the program is solved
    unrewritten.
    """
