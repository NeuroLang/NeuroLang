from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import Definition, Expression
from . import NaryLogicOperator


class LogicalComparison(Definition):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return "Compare\n\t{}\nwith\n\t{}".format(
            repr(self.first), repr(self.second)
        )


class ExpressionComparator(ExpressionWalker):
    @add_match(LogicalComparison(NaryLogicOperator, NaryLogicOperator))
    def nary_logic_operators(self, comp):
        if not isinstance(comp.first, type(comp.second)) or not isinstance(
            comp.second, type(comp.first)
        ):
            return False
        return self._compare_set_of_formulas(comp.first, comp.second)

    @add_match(LogicalComparison(Expression, Expression))
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
            if not self.walk(LogicalComparison(arg1, arg2)):
                return False
        elif arg1 != arg2:
            return False
        return True

    def _compare_set_of_formulas(self, first, second):
        return all(
            any(self.walk(LogicalComparison(f1, f2)) for f2 in second.formulas)
            for f1 in first.formulas
        )


def logic_exps_equal(exp1, exp2):
    if not isinstance(exp1, Expression) or not isinstance(exp2, Expression):
        raise ValueError("Can only compare expressions")
    return ExpressionComparator().walk(LogicalComparison(exp1, exp2))
