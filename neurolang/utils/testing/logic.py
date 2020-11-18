"""
This module exposes utility functions for tests on logic expressions.

It should not be used for any other purpose than testing.

"""
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Definition, Expression
from ...logic import NaryLogicOperator

__all__ = [
    "logic_exp_commutative_equal",
]


class LogicCommutativeComparison(Definition):
    """
    Comparison between two expressions that uses the commutativity property of
    some logic operators such as conjunctions and disjunctions.

    Parameters
    ----------
    first : Expression
        First expression.

    second : Expression
        Second expression.

    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return "Compare\n\t{}\nwith\n\t{}".format(
            repr(self.first), repr(self.second)
        )


class LogicCommutativeComparator(ExpressionWalker):
    """
    Compare logic expressions using the commutativity property of some logic
    operators such as conjunctions and disjunctions.

    """

    @add_match(
        LogicCommutativeComparison(NaryLogicOperator, NaryLogicOperator)
    )
    def nary_logic_operators(self, comp):
        """
        Compare two n-ary logic operators by comparing their two sets of
        formulas.

        """
        if not isinstance(comp.first, type(comp.second)) or not isinstance(
            comp.second, type(comp.first)
        ):
            return False
        return self._compare_set_of_formulas(comp.first, comp.second)

    @add_match(LogicCommutativeComparison(Expression, Expression))
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
            if not self.walk(LogicCommutativeComparison(arg1, arg2)):
                return False
        elif arg1 != arg2:
            return False
        return True

    def _compare_set_of_formulas(self, first, second):
        return all(
            any(
                self.walk(LogicCommutativeComparison(f1, f2))
                for f2 in second.formulas
            )
            for f1 in first.formulas
        )


def logic_exp_commutative_equal(exp1, exp2):
    """
    Compare two expressions using the commutativity property of logic
    operators.

    The two expressions do not need to be purely equal if the order of the
    formulas of a commutative logic operator is not the same in the two
    expressions.

    Apart from commutative logic operators, the comparison between the two
    expressions remains the same as the equality comparison.

    Parameters
    ----------
    exp1 : Expression
        First expression.

    exp2 : Expression
        Second expression.

    """
    if not isinstance(exp1, Expression) or not isinstance(exp2, Expression):
        raise ValueError("Can only compare expressions")
    return LogicCommutativeComparator().walk(
        LogicCommutativeComparison(exp1, exp2)
    )
