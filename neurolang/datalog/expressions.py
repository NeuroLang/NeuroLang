"""
Expressions for the intermediate representation of a
Datalog program.
"""

from typing import Any

from ..expression_walker import add_match
from ..expressions import Constant, ExpressionBlock
from ..logic import (Conjunction, Implication, Negation, UnaryLogicOperator,
                     Union)
from ..logic import expression_processing as ep

__all__ = [
    "Fact", "TranslateToLogic", "Implication", "Conjunction", "Union",
    "Negation", "NULL", "UNDEFINED", "UnaryLogicOperator"
]


class Fact(Implication):
    def __init__(self, consequent):
        super().__init__(consequent, Constant(True))

    @property
    def fact(self):
        return self.consequent

    def __repr__(self):
        return 'Fact{{{} \u2190 {}}}'.format(
            repr(self.consequent), True
        )


class Undefined(Constant):
    def __repr__(self):
        return 'UNDEFINED'


class NullConstant(Constant):
    def __repr__(self):
        return 'NULL'


UNDEFINED = Undefined(None)
NULL = NullConstant[Any](None)


class TranslateToLogic(ep.TranslateToLogic):
    @add_match(
        Implication(..., Constant(True)),
        lambda x: not isinstance(x, Fact)
    )
    def translate_true_implication(self, implication):
        return self.walk(Fact(implication.consequent))

    @add_match(ExpressionBlock)
    def build_conjunction_from_expression_block(self, expression_block):
        formulas = tuple()
        for expression in expression_block.expressions:
            new_exp = self.walk(expression)
            formulas += (new_exp,)
        return self.walk(Union(formulas))
