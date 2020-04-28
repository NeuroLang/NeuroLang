"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets and has support for constraints.
"""

from expression_walker import ExpressionWalker, add_match
from expressions import Expression

from ..logic import LogicOperator, NaryLogicOperator
from .basic_representation import DatalogProgram


class RightImplication(LogicOperator):
    """
    This class defines implications to the right. They are used to define
    constraints in datalog programs. The functionality is the same as
    that of an implication, but with body and head inverted in position.
    """

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self._symbols = consequent._symbols | antecedent._symbols

    def __repr__(self):
        return "RightImplication{{{} \u2192 {}}}".format(
            repr(self.antecedent), repr(self.consequent)
        )


class DatalogConstraintsProgramMixin(ExpressionWalker):

    protected_keywords = set({"__constraints__"})

    @add_match(NaryLogicOperator)
    def add_nary_constraint(self, expression):
        for term in expression.formulas:
            self.walk(term)

    @add_match(LogicOperator)
    def add_logic_constraint(self, expression):
        if (
            isinstance(expression, RightImplication)
            and "__constraints__" in self.symbol_table
        ):
            constrains = self.symbol_table["__constraints__"]
            constrains = Union((constrains.formulas + (expression,)))
        elif isinstance(expression, RightImplication):
            self.symbol_table["__constraints__"] = Union((expression))

    def constraints(self):
        return self.symbol_table["__constraints__"]


class DatalogConstraintsProgram(
    DatalogConstraintsProgramMixin, DatalogProgram
):
    pass
