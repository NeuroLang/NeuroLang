from . import expression_walker as ew
from .expressions import (Constant, Statement, Symbol, Unknown, Expression)


class Evaluate(Expression):
    """Executes the evaluation of a symbol of expresion and returns
    the result

    Parameters
    ----------
    expression : Expression
        expression or symbol to be evaluated
    """

    def __init__(self, expression):
        self.expression = expression


class LazyCodeEvaluationMixin(ew.PatternWalker):
    @ew.add_match(Statement)
    def statement(self, expression):
        if expression.lhs.type is Unknown:
            expression.lhs.change_type(expression.rhs)
        self.symbol_table[expression.lhs] = expression.rhs
        return expression.lhs

    @ew.add_match(Symbol)
    def symbol(self, expression):
        return expression

    @ew.add_match(Evaluate(Symbol))
    def execute_symbol(self, expression):
        symbol = expression.expression
        value = self.symbol_table[symbol]
        if value != symbol:
            value = self.walk(Evaluate(value))
            expression.change_type(value.type)
            self.symbol_table[symbol] = value
        return value

    @ew.add_match(Evaluate(Constant))
    def execute_constant(self, expression):
        return expression.expression

    @ew.add_match(Evaluate(Expression))
    def execute(self, expression):
        args = tuple(
            self.walk(Evaluate(arg)) for arg in expression.expression.unapply()
        )
        return self.walk(expression.expression.apply(*args))

    @ew.add_match(Evaluate)
    def execute_tuple(self, expression):
        args = tuple(self.walk(Evaluate(arg)) for arg in expression.expression)
        return args
