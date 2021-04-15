"""
This module exposes utility functions for tests on relational algebra
expressions.

It should not be used for any other purpose than testing.

"""
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Definition, Expression
from ...relational_algebra import (
    NAryRelationalAlgebraOperation,
    RelationalAlgebraOperation,
)

__all__ = [
    "ra_exp_commutative_equal",
]


class RelationalAlgebraCommutativeComparison(Definition):
    """
    Comparison between two expressions that uses the commutativity property of
    some relational algebra operations such as natural join.

    Parameters
    ----------
    first : RelationalAlgebraOperation
        First set.

    second : RelationalAlgebraOperation
        Second set.

    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return "Compare\n\t{}\nwith\n\t{}".format(
            repr(self.first), repr(self.second)
        )


class RelationalAlgebraCommutativeComparator(ExpressionWalker):
    """
    Compare relational algebra expressions using the commutativity property of
    some operators such as natural join.

    """

    @add_match(
        RelationalAlgebraCommutativeComparison(
            NAryRelationalAlgebraOperation, NAryRelationalAlgebraOperation
        )
    )
    def nary_relational_algebra_operators(self, comp):
        """
        Compare two n-ary relational algebra operators by comparing their two
        sets of expressions.

        """
        if not isinstance(comp.first, type(comp.second)) or not isinstance(
            comp.second, type(comp.first)
        ):
            return False
        return self._compare_set_of_expressions(comp.first, comp.second)

    @add_match(
        RelationalAlgebraCommutativeComparison(
            RelationalAlgebraOperation, RelationalAlgebraOperation
        )
    )
    def expressions(self, comp):
        args1 = comp.first.unapply()
        args2 = comp.second.unapply()
        if len(args1) != len(args2):
            return False
        for arg1, arg2 in zip(args1, args2):
            if not self._args_equal(arg1, arg2):
                return False
        return True

    def _args_equal(self, arg1, arg2):
        if isinstance(arg1, Expression) and isinstance(arg2, Expression):
            if not self.walk(
                RelationalAlgebraCommutativeComparison(arg1, arg2)
            ):
                return False
        elif arg1 != arg2:
            return False
        return True

    def _compare_set_of_expressions(self, first, second):
        return all(
            any(
                self.walk(RelationalAlgebraCommutativeComparison(r1, r2))
                for r2 in second.relations
            )
            for r1 in first.relations
        )


def ra_exp_commutative_equal(exp1, exp2):
    """
    Compare two expressions using the commutativity property of relational
    algebra operators.

    The two expressions do not need to be purely equal if the order of the
    expressions of a commutative operator is not the same in the two
    expressions.

    Apart from commutative operators, the comparison between the two
    expressions remains the same as the equality comparison.

    Parameters
    ----------
    exp1 : RelationalAlgebraOperation
        First expression.

    exp2 : RelationalAlgebraOperation
        Second expression.

    """
    if not isinstance(exp1, RelationalAlgebraOperation) or not isinstance(
        exp2, RelationalAlgebraOperation
    ):
        raise ValueError("Can only compare relational algebra expressions")
    return RelationalAlgebraCommutativeComparator().walk(
        RelationalAlgebraCommutativeComparison(exp1, exp2)
    )
