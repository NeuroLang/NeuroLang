"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intentional, and builtin
sets and has support for constraints.
"""

from typing import Iterable, Tuple

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
    """
    The constraints of a program, and its existential rules, are in its
    symbol table, under ``__constraints__`` and ``__existential_rules__``:
    they follow its scopes, as its rules and facts do.
    """

    protected_keywords = {"__constraints__", "__existential_rules__"}

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
        to those of the program.

        Parameters
        ----------
        categorized_constraints : dict
            Dictionary of constraints where each key is
            the functor of the consequent of the rules
            and the values are lists of constraints with
            each rule associated to the corresponding functor.
        '''
        known = set(self.constraints().formulas)
        added = tuple(
            c for formulas in categorized_constraints.values()
            for c in formulas if c not in known
        )
        self.symbol_table[Symbol("__constraints__")] = Union(
            self.constraints().formulas + added
        )

    @property
    def categorized_constraints(self):
        '''The constraints of the program in a dictionary, where the key
        is the functor of the consequent of each of them, useful for
        XRewriter optimizations.'''
        constraints = self.symbol_table.get(Symbol("__constraints__"))
        cached = self.__dict__.get("_categorized_constraints")
        if cached is None or cached[0] is not constraints:
            categorized = {}
            for sigma in getattr(constraints, "formulas", ()):
                categorized.setdefault(
                    sigma.consequent.functor.name, set()
                ).add(sigma)
            cached = self._categorized_constraints = (
                constraints, categorized
            )
        return cached[1]

    def get_constraints(self):
        '''Returns the constraints in a dictionary, where the key is the functor
        of the consequent of each of the rules.

        Returns
        -------
        Dictionary containing all constraints loaded in the datalog program,
        indexed according to the functor of the rule consequent.
        '''
        return self.categorized_constraints

    @property
    def existential_rules(self):
        """The existential rules of the program, by entity."""
        rules = self.symbol_table.get(Symbol("__existential_rules__"))
        return {} if rules is None else dict(rules.value)

    def add_existential_rules(self, rules):
        """Add existential rules, by entity, to those of the program."""
        merged = {
            entity: tuple(entity_rules)
            for entity, entity_rules in self.existential_rules.items()
        }
        for entity, entity_rules in rules.items():
            merged[entity] = merged.get(entity, ()) + tuple(entity_rules)
        self.symbol_table[Symbol("__existential_rules__")] = Constant[Tuple](
            tuple(merged.items()), auto_infer_type=False, verify_type=False
        )


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