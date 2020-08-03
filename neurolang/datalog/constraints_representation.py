"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets and has support for constraints.
"""

from ..expression_walker import ExpressionWalker, add_match
from ..logic import LogicOperator, NaryLogicOperator, Union
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


class DatalogConstraintsMixin(ExpressionWalker):
    protected_keywords = {"__constraints__"}

    @add_match(NaryLogicOperator)
    def add_nary_constraint(self, expression):
        for formula in expression.formulas:
            self.walk(formula)

    @add_match(LogicOperator)
    def add_logic_constraint(self, expression):
        if (
            isinstance(expression, RightImplication)
            and "__constraints__" in self.symbol_table
        ):
            constrains = self.symbol_table["__constraints__"]
            constrains = Union((constrains.formulas + (expression,)))
            self.symbol_table["__constraints__"] = constrains
        elif isinstance(expression, RightImplication):
            self.symbol_table["__constraints__"] = Union((expression,))

    def constraints(self):
        return self.symbol_table.get("__constraints__", Union(()))


class DatalogConstraintsProgram(DatalogProgram, DatalogConstraintsMixin):
    pass
