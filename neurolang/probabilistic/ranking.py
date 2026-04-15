from collections import defaultdict
from itertools import product

from ..expression_walker import ReplaceExpressionWalker
from ..expressions import Symbol
from ..logic.expression_processing import extract_logic_atoms
from ..logic.transformations import convert_to_pnf_with_dnf_matrix
from ..relational_algebra import (
    Projection,
    Selection,
    eq_,
    int2columnint_constant
)


def verify_that_the_query_is_ranked(query):
    argument_precedence = defaultdict(set)
    query = convert_to_pnf_with_dnf_matrix(query)

    for atom in extract_logic_atoms(query):
        if not isinstance(atom.functor, Symbol):
            return False
        for i, arg in enumerate(atom.args[1:], 1):
            argument_precedence[arg].update(atom.args[:i])

    for arg1, arg2 in product(argument_precedence, argument_precedence):
        if (
            arg1 in argument_precedence[arg2] and
            arg2 in argument_precedence[arg1]
        ):
            return False
    return True


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
