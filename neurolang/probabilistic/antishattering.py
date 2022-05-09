from itertools import product

from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionWalker,
    add_match,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, Implication
from ..logic.expression_processing import extract_logic_atoms
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation,
)
from .transforms import convert_rule_to_ucq, convert_ucq_to_ccq


class ReplaceFunctionApplicationArgsWalker(ExpressionWalker):
    def __init__(self, symbol_replacements):
        self.symbol_replacements = symbol_replacements

    @add_match(FunctionApplication)
    def replace_args(self, symbol):
        new_args = tuple()
        old_vars = self.symbol_replacements.keys()
        for arg in symbol.args:
            if arg in old_vars:
                new_args += (self.symbol_replacements[arg],)
            else:
                new_args += (arg,)

        return symbol.functor(*new_args)


def pchoice_constants_as_head_variables(query, cpl_program):
    """
    First step of the antishattering strategy.
    Given an implication, body constants that are associated with
    probabilistic choices are removed and replaced by variables
    that are also included in the consequent.

    Parameters
    ----------
    query : Implication
        Query to be transformed.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    Implication
        Implication with the constants of the pchoices
        replaced by variables, which are also added in the consequent.
    dic
        Dictionary mapping replaced constants to their new
        replacement variables.
    """
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, query.antecedent
    )
    query_ucq = convert_rule_to_ucq(query)
    query_ucq = convert_ucq_to_ccq(query_ucq, transformation="DNF")

    parameter_variable = {}
    args_amount = {}

    for clause in query_ucq.formulas:
        for atom in extract_logic_atoms(clause):
            if not isinstance(
                atom, Constant
            ) and is_atom_a_probabilistic_choice_relation(atom, symbol_table):
                for i, arg in enumerate(atom.args):
                    if isinstance(arg, Constant):
                        if atom not in parameter_variable:
                            parameter_variable[atom] = set([(i, arg)])
                            args_amount[atom.functor] = len(
                                [
                                    arg
                                    for arg in atom.args
                                    if isinstance(arg, Constant)
                                ]
                            )
                        else:
                            tmp = parameter_variable[atom]
                            tmp.add((i, arg))
                            parameter_variable[atom] = tmp

    new_atoms = tuple()
    rewritten_atoms = {}
    for atom, var_pos in parameter_variable.items():
        n_args = args_amount[atom.functor]
        filtered = [
            clean_tuple(elem)
            for elem in product(var_pos, repeat=n_args)
            if ordered_atoms(elem)
        ]
        new_atom = Symbol.fresh()
        cpl_program.add_extensional_predicate_from_tuples(new_atom, filtered)
        atom_new_vars = [Symbol.fresh() for _ in range(len(filtered[0]))]

        new_atoms += tuple([new_atom(*atom_new_vars)])
        rewritten_atoms[atom] = ReplaceFunctionApplicationArgsWalker(
            dict(zip(*filtered, atom_new_vars))
        ).walk(atom)

    query = ReplaceExpressionWalker(rewritten_atoms).walk(query)

    query = Implication(
        query.consequent, Conjunction((*new_atoms, query.antecedent))
    )

    return query


def ordered_atoms(atoms):
    original_len = len(atoms)
    # assumes order, should be modified
    new_atoms = [atom for pos, atom in enumerate(atoms) if pos == atom[0]]
    return len(new_atoms) == original_len


def clean_tuple(atoms):
    return tuple([atom[1].value for atom in atoms])
