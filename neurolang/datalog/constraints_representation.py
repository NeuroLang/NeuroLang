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
    categorized_constraints = {}

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
        '''Function that returns the constraints contained
        in the Datalog program.

        Returns
        -------
        Union of formulas containing all constraints loaded into the program.
        '''
        return self.symbol_table.get("__constraints__", Union(()))

    def set_constraints(self, categorized_constraints):
        '''This function receives a dictionary with the contraints organized
        according to the functor of its consequent and is in charge of setting
        it both in the symbol table and in the global variable
        `categorized_contraints`, useful for XRewriter optimizations.

        Parameters
        ----------
        categorized_constraints : dict
            Dictionary of constraints where each key is
            the functor of the consequent of the rules
            and the values are lists of contraints with
            each rule associated to the corresponding functor.
        '''
        if isinstance(categorized_constraints, dict):
            cons = [b for a in list(categorized_constraints.values()) for b in a]
            self.symbol_table["__constraints__"] = Union(cons)

            self.categorized_constraints = categorized_constraints

    def get_constraints(self):
        '''Returns the contraints in a dictionary, where the key is the functor
        of the consequent of each of the rules.

        Returns
        -------
        Dictionary containing all constraints loaded in the datalog program,
        indexed according to the functor of the rule consequent.
        '''
        if not self.categorized_constraints:
            constraints = self.symbol_table.get("__constraints__", Union(()))
            for f in constraints.formulas:
                self._categorize_constraints(f)

        return self.categorized_constraints

    def _categorize_constraints(self, sigma):
        '''Function in charge of sorting the constraints in a dictionary
        using the consequent functor as an index.

        This categorization is useful to obtain the constraints in the
        way they are needed for the rewriting algorithm.

        Parameters
        ----------
        sigma : RightImplicationn
            constraint to be categorized.
        '''
        sigma_functor = sigma.consequent.functor.name
        if (
            self.categorized_constraints
            and sigma_functor in self.categorized_constraints
        ):
            cons_set = self.categorized_constraints[sigma_functor]
            if sigma not in cons_set:
                cons_set.add(sigma)
                self.categorized_constraints[sigma_functor] = cons_set
        else:
            self.categorized_constraints[sigma_functor] = set([sigma])


class DatalogConstraintsProgram(DatalogProgram, DatalogConstraintsMixin):
    pass
