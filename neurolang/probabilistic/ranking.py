from collections import defaultdict
from itertools import product

from ..expression_walker import ReplaceExpressionWalker, add_match
from ..expressions import FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, LogicOperator, Quantifier
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import LogicExpressionWalker
from ..relational_algebra import (
    Projection,
    Selection,
    eq_,
    int2columnint_constant
)
from .transforms import convert_ucq_to_ccq


def argsort_symbols(seq):
    return sorted(range(len(seq)), key=lambda i: seq[i].name)


class VerifyQueryIsRanked(LogicExpressionWalker):
    @add_match(Conjunction)
    def verify_conjunction(self, expression):
        if any(not self.walk(formula) for formula in expression.formulas):
            return False

        free_variables = extract_logic_free_variables(expression)

        if len(free_variables) < 2:
            return True

        argument_precedence = {v: set() for v in free_variables}
        atom_order = {}
        for atom in extract_logic_atoms(expression):
            if atom.functor not in atom_order:
                atom_order[atom.functor] = argsort_symbols(atom.args)
            args = tuple(atom.args[i] for i in atom_order[atom.functor])
            for i, arg in enumerate(args[1:], 1):
                if arg in argument_precedence:
                    argument_precedence[arg].update(args[:i])

        for arg1, arg2 in product(argument_precedence, argument_precedence):
            if (
                arg1 in argument_precedence[arg2] and
                arg2 in argument_precedence[arg1]
            ):
                return False
        return True

    @add_match(Disjunction)
    def verify_disjunction(self, expression):
        return self.verify_conjunction(expression)

    @add_match(Quantifier)
    def verify_quantifier(self, expression):
        return self.walk(expression.body)

    @add_match(LogicOperator)
    def logic_operator(self, expression):
        fvs = True
        for arg in expression.unapply():
            fvs &= self.walk(arg)
        return fvs

    @add_match(FunctionApplication)
    def verify_function_application(self, expression):
        return True


def verify_that_the_query_is_ranked(query):
    if any(
        not isinstance(atom.functor, Symbol)
        for atom in extract_logic_atoms(query)
    ):
        return False
    query = convert_ucq_to_ccq(query, transformation='DNF')

    return VerifyQueryIsRanked().walk(query)


def partially_rank_query(query, symbol_table):
    atoms_to_rank, atom_argument_quantity = \
        _process_atoms_with_repeatd_arguments(query)

    for symbol, instances_to_rank in atoms_to_rank.items():
        if len(instances_to_rank) != 1:
            continue
        arguments_to_rank = instances_to_rank.pop()
        if all(len(repeats) == 0 for _, repeats in arguments_to_rank):
            continue
        expression, arguments_to_eliminate = _build_ra_expression(
            atom_argument_quantity, symbol, arguments_to_rank
        )
        new_symbol = Symbol.fresh()
        symbol_table[new_symbol] = expression
        replacements = {
            atom: new_symbol(*(
                arg for i, arg in enumerate(atom.args)
                if i not in arguments_to_eliminate
            )
            )
            for atom in extract_logic_atoms(query)
            if atom.functor == symbol
        }
        query = ReplaceExpressionWalker(replacements).walk(query)
    return query


def _build_ra_expression(atom_argument_quantity, symbol, arguments_to_rank):
    expression = symbol
    arguments_to_eliminate = set()
    for arg, repeats in arguments_to_rank:
        arg = int2columnint_constant(arg)
        for repeat in repeats:
            expression = Selection(
                    expression,
                    eq_(arg, int2columnint_constant(repeat))
                )
        arguments_to_eliminate.update(repeats)
    attributes = tuple(
            int2columnint_constant(i)
            for i in range(atom_argument_quantity[symbol])
            if i not in arguments_to_eliminate
        )
    expression = Projection(expression, attributes)
    return expression, arguments_to_eliminate


def _process_atoms_with_repeatd_arguments(query):
    atoms_to_rank = defaultdict(set)
    atom_argument_quantity = dict()
    for atom in extract_logic_atoms(query):
        if not isinstance(atom.functor, Symbol):
            continue
        atom_argument_quantity[atom.functor] = len(atom.args)
        arg_counts = defaultdict(list)
        for i, arg in enumerate(atom.args):
            arg_counts[arg].append(i)
        arg_counts_tuple = tuple(
            (v[0], tuple(v[1:])) for v in arg_counts.values()
        )
        atoms_to_rank[atom.functor].add(arg_counts_tuple)
    return atoms_to_rank, atom_argument_quantity
