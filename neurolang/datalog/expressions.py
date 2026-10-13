"""
Expressions for the intermediate representation of a
Datalog program.
"""

from typing import Any

from ..config import config
from ..expression_walker import add_match
from ..expressions import (
    Constant, ExpressionBlock, FunctionApplication, Symbol
)
from ..logic import (Conjunction, Implication, Negation, UnaryLogicOperator,
                     Union)
from ..logic import expression_processing as ep

__all__ = [
    "Fact", "TranslateToLogic", "Implication", "Conjunction", "Union",
    "Negation", "NULL", "UNDEFINED", "UnaryLogicOperator",
    "AdornedSymbol", "predicate_identity",
]


def predicate_identity(expr):
    """Return a full identity string for a predicate functor.

    For `AdornedSymbol`s this returns the adorned string representation
    (e.g. ``P^bf``), which distinguishes predicates that share a base
    name. For plain `Symbol`s this returns ``expr.name``.

    This helper must only be used where the pipeline needs to distinguish
    distinct predicate symbols. It must not be used for EDB/IDB *lookup*
    by base name (e.g. in magic-sets), where ``.name`` is the intended key.
    """
    if isinstance(expr, AdornedSymbol):
        return str(expr)
    return expr.name


class AdornedSymbol(Symbol):
    def __init__(self, expression, adornment, number):
        self.expression = expression
        self.adornment = adornment
        self.number = number
        self.is_fresh = False

    @property
    def name(self):
        if isinstance(self.expression, Symbol):
            return self.expression.name
        else:
            raise NotImplementedError()

    def __eq__(self, other):
        return (
            hash(self) == hash(other)
            and isinstance(other, AdornedSymbol)
            and self.unapply() == other.unapply()
        )

    def __hash__(self):
        return hash((self.expression, self.adornment, self.number))

    def __str__(self) -> str:
        if isinstance(self.expression, Symbol):
            rep = self.expression.name
        elif isinstance(self.expression, Constant):
            rep = self.expression.value
        else:
            raise NotImplementedError()

        if len(self.adornment) > 0:
            superindex = f'^{self.adornment}'
        else:
            superindex = ''

        if self.number is not None:
            subindex = f'_{self.number}'
        else:
            subindex = ''
        return f'{rep}{superindex}{subindex}'

    def __repr__(self):
        if config.expression_type_printing():
            return (
                f'S{{{self}: '
                f'{self.__type_repr__}}}'
            )
        else:
            return f'S{{{self}}}'


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


class AggregationApplication(FunctionApplication):
    def __repr__(self):
        r = u'\u03BB{{<{}>: {}}}'.format(self.functor, self.__type_repr__)
        if self.args is ...:
            r += '(...)'
        elif self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args)
                + ')'
                )
        return r


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
