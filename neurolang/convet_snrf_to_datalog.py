from operator import and_, or_, invert
from uuid import uuid4
from typing import Callable

from .expression_walker import PatternWalker, add_match
from .expressions import (
    FunctionApplication, Symbol,
    Constant, Statement, ExpressionBlock,
    ExistentialPredicate
)
from .utils import OrderedSet


__all__ = ['ConvertSNRFToDatalogWalker']


class ConvertSNRFToDatalogWalker(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        datalog = tuple()
        consequents = OrderedSet()
        variables = OrderedSet()

        for a in expression.args:
            consequent, eb = self.walk(a)
            consequents.add(consequent)
            variables |= consequent.args
            datalog += eb.expressions

        types = [v.type for v in variables]

        functor = Constant[Callable[[bool] * len(consequents), bool]](and_)

        new_expression = Statement(
            Symbol[Callable[types, bool]](str(uuid4()))(*variables),
            FunctionApplication(functor, tuple(consequents))
        )

        datalog += (new_expression,)
        return new_expression.lhs, ExpressionBlock(datalog)

    @add_match(FunctionApplication(Constant(or_), ...))
    def disjunction(self, expression):
        datalog = tuple()
        consequents = OrderedSet()
        variables = OrderedSet()

        for a in expression.args:
            consequent, eb = self.walk(a)
            consequents.add(consequent)
            variables |= consequent.args
            datalog += eb.expressions

        types = [v.type for v in variables]

        new_predicate = Symbol[Callable[types, bool]](str(uuid4()))(*variables)
        new_expressions = tuple(
            Statement(new_predicate, consequent)
            for consequent in consequents
        )

        datalog += new_expressions
        return new_predicate, ExpressionBlock(datalog)

    @add_match(FunctionApplication(Constant(invert), FunctionApplication))
    def negation(self, expression):
        raise NotImplemented()

    @add_match(
        FunctionApplication,
        lambda exp: not any(
            isinstance(a, FunctionApplication)
            for a in exp.args
        )
    )
    def function_application(self, expression):
        return expression, ExpressionBlock(tuple())

    @add_match(ExistentialPredicate)
    def existential(self, expression):
        head = expression.head
        if not isinstance(head, tuple):
            head = (head,)

        lhs, datalog = self.walk(expression.body)

        new_args = []
        new_types = []
        for a in lhs.args:
            if a in head:
                continue
            new_args.append(a)
            new_types.append(a.type)

        new_lhs_functor = Symbol[Callable[new_types, bool]](str(uuid4()))
        new_lhs = new_lhs_functor(*new_args)

        datalog = ExpressionBlock(
            datalog.expressions + (
                Statement(new_lhs, lhs),
            )
        )

        return new_lhs, datalog
