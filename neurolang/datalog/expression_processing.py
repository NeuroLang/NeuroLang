"""
Utilities to process intermediate representations of
Datalog programs.
"""

from operator import and_, invert, or_, xor

from ..expression_walker import PatternWalker, add_match, expression_iterator
from ..expressions import (Constant, ExpressionBlock, FunctionApplication,
                           NeuroLangException, Quantifier, Symbol)
from ..utils import OrderedSet
from .expressions import Implication


class ExtractDatalogFreeVariablesWalker(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        fvs = OrderedSet()
        for arg in expression.args:
            fvs |= self.walk(arg)
        return fvs

    @add_match(FunctionApplication)
    def extract_variables_fa(self, expression):
        args = expression.args

        variables = OrderedSet()
        for a in args:
            if isinstance(a, Symbol):
                variables.add(a)
            elif isinstance(a, Constant):
                pass
            else:
                raise NeuroLangException('Not a Datalog function application')
        return variables

    @add_match(Quantifier)
    def extract_variables_q(self, expression):
        return self.walk(expression.body) - expression.head._symbols

    @add_match(Implication)
    def extract_variables_s(self, expression):
        return (
            self.walk(expression.antecedent) -
            self.walk(expression.consequent)
        )

    @add_match(ExpressionBlock)
    def extract_variables_eb(self, expression):
        res = set()
        for exp in expression.expressions:
            res |= self.walk(exp)

        return res

    @add_match(...)
    def _(self, expression):
        return OrderedSet()


def extract_datalog_free_variables(expression):
    '''extract variables from expression knowing that it's in Datalog format'''
    efvw = ExtractDatalogFreeVariablesWalker()
    return efvw.walk(expression)


def is_conjunctive_expression(expression):
    return all(
        not isinstance(exp, FunctionApplication) or
        (
            isinstance(exp, FunctionApplication) and
            (
                (
                    isinstance(exp.functor, Constant) and
                    exp.functor.value is and_
                ) or all(
                    not isinstance(arg, FunctionApplication)
                    for arg in exp.args
                )
            )
        )
        for _, exp in expression_iterator(expression)
    )


def is_conjunctive_expression_with_nested_predicates(expression):
    stack = [expression]
    while stack:
        exp = stack.pop()
        if isinstance(exp, FunctionApplication):
            if isinstance(exp.functor, Constant):
                if exp.functor is and_:
                    stack += exp.args
                    continue
                elif exp.functor in (or_, invert, xor):
                    return False
            stack += [
                arg for arg in exp.args
                if isinstance(arg, FunctionApplication)
            ]
    return True


class ExtractDatalogPredicates(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        res = OrderedSet()
        for arg in expression.args:
            res |= self.walk(arg)
        return res

    @add_match(FunctionApplication)
    def extract_predicates_fa(self, expression):
        return OrderedSet([expression])

    @add_match(ExpressionBlock)
    def expression_block(self, expression):
        res = OrderedSet()
        for exp in expression.expressions:
            res |= self.walk(exp)
        return res


def extract_datalog_predicates(expression):
    """
    extract predicates from expression
    knowing that it's in Datalog format
    """
    edp = ExtractDatalogPredicates()
    return edp.walk(expression)
