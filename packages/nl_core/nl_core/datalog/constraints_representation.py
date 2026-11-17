"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intentional, and builtin
sets and has support for constraints.
"""

from typing import Iterable

from .expression_processing import extract_logic_atoms
from .basic_representation import DatalogProgramMixin
from nl_expr.expression_walker import (
    ExpressionWalker,
    PatternWalker,
    add_match,
)
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import Implication, LogicOperator, Union

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


class DatalogConstraintsMixin(PatternWalker):
    protected_keywords = {"__constraints__"}
    categorized_constraints = {}
    existential_rules = {}

    @add_match(RightImplication)
    def add_logic_constraint(self, expression):
        sym_constraints = Symbol("__constraints__")
        if (
            isinstance(expression, RightImplication)
            and sym_constraints in self.symbol_table
        ):
            constrains = self.symbol_table[sym_constraints]
            constrains = Union((constrains.formulas + (expression,)))
            self.symbol_table[sym_constraints] = constrains
        elif isinstance(expression, RightImplication):
            self.symbol_table[sym_constraints] = Union((expression,))

        return expression

    def constraints(self):
        '''Function that returns the constraints contained
        in the Datalog program.

        Returns
        -------
        Union of formulas containing all constraints loaded into the program.
        '''
        sym_constraints = Symbol("__constraints__")
        return self.symbol_table.get(sym_constraints, Union(()))

    def add_constraints(self, categorized_constraints):
        '''Add constraints, organised by the functor of their consequent,
        to those of the program: both to the symbol table and to
        `categorized_constraints`, useful for XRewriter optimizations.

        Parameters
        ----------
        categorized_constraints : dict
            Dictionary of constraints where each key is
            the functor of the consequent of the rules
            and the values are lists of constraints with
            each rule associated to the corresponding functor.
        '''
        constraints = {
            functor: set(formulas)
            for functor, formulas in self.get_constraints().items()
        }
        for functor, formulas in categorized_constraints.items():
            constraints.setdefault(functor, set()).update(formulas)
        self.symbol_table[Symbol("__constraints__")] = Union(
            tuple(c for formulas in constraints.values() for c in formulas)
        )
        self.categorized_constraints = constraints

    def get_constraints(self):
        '''Returns the constraints in a dictionary, where the key is the functor
        of the consequent of each of the rules.

        Returns
        -------
        Dictionary containing all constraints loaded in the datalog program,
        indexed according to the functor of the rule consequent.
        '''
        sym_constraints = Symbol("__constraints__")
        if not self.categorized_constraints:
            constraints = self.symbol_table.get(sym_constraints, Union(()))
            for f in constraints.formulas:
                self._categorize_constraints(f)

        return self.categorized_constraints

    def add_existential_rules(self, rules):
        """Add existential rules, by entity, to those of the program."""
        merged = {
            entity: list(entity_rules)
            for entity, entity_rules in self.existential_rules.items()
        }
        for entity, entity_rules in rules.items():
            merged.setdefault(entity, []).extend(entity_rules)
        self.existential_rules = merged

    def _categorize_constraints(self, sigma):
        '''Function in charge of sorting the constraints in a dictionary
        using the consequent functor as an index.

        This indexing is useful to obtain the constraints in the
        way they are needed for the rewriting algorithm.

        Parameters
        ----------
        sigma : RightImplication
            constraint to be categorized.
        '''
        sigma_functor = sigma.consequent.functor.name
        # A new dictionary, not the class's, shared by all the programs.
        self.categorized_constraints = {
            **self.categorized_constraints,
            sigma_functor: (
                self.categorized_constraints.get(sigma_functor, set())
                | {sigma}
            ),
        }


def iter_implications(expression):
    """
    Yield the implications of ``expression``.

    ``expression`` is an implication, a union of implications, or anything
    else, such as None or a constant, which has none.
    """
    if isinstance(expression, Implication):
        yield expression
    elif isinstance(expression, Union):
        yield from expression.formulas


def get_list_of_intensional_rules(program):
    """
    Return the list of the rules of the intensional database of ``program``.

    The constraints of ``program``, its right implications, are left out.

    Parameters
    ----------
    program : DatalogProgram
        Program from which we want to obtain the list of intensional rules.

    Returns
    -------
    list
        list of intensional rules
    """
    return [
        rule
        for exp in program.intensional_database().values()
        for rule in iter_implications(exp)
        if not isinstance(rule, RightImplication)
    ]


def reachable_code(query, datalog):
    """Produces the code reachable by a query

    Parameters
    ----------
    query : Implication
        Rule to figure out the reachable program from
    datalog : DatalogProgram
        datalog instance containing the EDB and IDB.

    Returns
    -------
    ExpressionBlock
        Code needed to solve the query.
    """
    if not isinstance(query, Iterable):
        query = [query]

    reachable_code = []
    idb = datalog.intensional_database()
    to_reach = [q.consequent.functor for q in query]
    reached = set()
    seen_rules = set()
    while to_reach:
        p = to_reach.pop()
        reached.add(p)
        rules = idb[p]
        if isinstance(rules, Constant):
            continue
        for rule in rules.formulas:
            if rule in seen_rules:
                continue
            seen_rules.add(rule)
            # A constraint is not code to solve, but the predicates of
            # its antecedent are: their rules give the values it applies to.
            if not isinstance(rule, RightImplication):
                reachable_code.append(rule)
            for predicate in extract_logic_atoms(rule.antecedent):
                functor = predicate.functor
                if functor not in reached and functor in idb:
                    to_reach.append(functor)

    return Union(reachable_code[::-1])

class DatalogConstraintsProgram(
    DatalogConstraintsMixin,
    DatalogProgramMixin,
    ExpressionWalker
):
    pass