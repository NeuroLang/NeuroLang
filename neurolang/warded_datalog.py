from .expressions import (
    Symbol, Constant, ExpressionBlock,
    FunctionApplication,
)
from .expression_walker import (
    PatternWalker, add_match
)
from .solver_datalog_naive import (
    Implication, Fact
)

class WardedDatalog(PatternWalker):

    def __init__(self):
        self.can_be_dangerous = []


    @add_match(ExpressionBlock)
    def warded_expression_block(self, expression):
        for rule in expression.expressions:
            self.walk(rule)

        cwd = CheckWardedDatalog(self.can_be_dangerous)
        cwd.walk(expression)


    @add_match(FunctionApplication(Constant, ...))
    def warded_function_constant(self, expression):
        args = set()
        for arg in expression.args:
            temp = self.walk(arg)
            args = args.union(temp)

        return args


    @add_match(FunctionApplication)
    def warded_function_application(self, expression):
        symbols = set()
        for arg in expression.args:
            symbol = self.walk(arg)
            symbols.add(symbol)

        return symbols


    @add_match(Fact)
    def warded_fact(self, expression):
        symbols = set()
        for arg in expression.fact.args:
            symbol = self.walk(arg)
            symbols.add(symbol)

        return symbols


    @add_match(Implication)
    def warded_implication(self, expression):
        antecedent = self.walk(expression.antecedent)
        consequent = self.walk(expression.consequent)

        free_vars = antecedent.symmetric_difference(consequent)

        for var in free_vars:
            if var in consequent:
                self.can_be_dangerous.append(var)


    @add_match(Symbol)
    def warded_symbol(self, expression):
        return expression.name


    @add_match(Constant)
    def warded_constant(self, expression):
        pass


class CheckWardedDatalog(PatternWalker):

    def __init__(self, can_be_dangerous):
        self.can_be_dangerous = can_be_dangerous


    @add_match(ExpressionBlock)
    def warded_expression_block(self, expression):
        for rule in expression.expressions:
            warded = self.walk(rule)
            if not warded:
                return False
        else:
            return True