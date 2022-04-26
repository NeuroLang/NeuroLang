from operator import eq

from ..expression_walker import (PatternWalker, ReplaceExpressionWalker,
                                 add_match)
from ..expressions import Constant, Symbol
from ..logic import Implication
from ..logic.expression_processing import extract_logic_atoms
from ..relational_algebra import (Projection, RelationalAlgebraOperation,
                                  Selection, str2columnstr_constant)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation)


class ProjectionSelectionByPChoiceConstant(PatternWalker):

    def __init__(self, constants_by_formula_dict):
        self.constants_by_formula_dict = constants_by_formula_dict

    @add_match(RelationalAlgebraOperation)
    def match_projection(self, raoperation):
        operation = raoperation.relation
        prov_columns = raoperation.provenance_column
        symbols_as_columns = [
            str2columnstr_constant(symbol.name) for symbol
            in self.constants_by_formula_dict.values()
        ]
        non_proyected_vars = set(symbols_as_columns).union(set([prov_columns]))
        proyected_vars = raoperation.columns() - non_proyected_vars
        eq_ = Constant(eq)
        for constant, fresh_var in self.constants_by_formula_dict.items():
            operation = Selection(
                operation,
                eq_(str2columnstr_constant(fresh_var.name), constant)
            )

        proyected = tuple([prov_columns])
        if len(proyected_vars) > 0:
            for pv in proyected_vars:
                proyected = proyected + (Constant(pv), Constant(pv))

        operation = Projection(operation, tuple(proyected))

        return ProvenanceAlgebraSet(operation, prov_columns)

def _pchoice_constants_as_head_variables(query, cpl_program):
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, query.antecedent
    )
    constants_by_formula = {}
    for atom in extract_logic_atoms(query.antecedent):
        if is_atom_a_probabilistic_choice_relation(atom, symbol_table):
            replacements = {arg:Symbol.fresh() for arg in atom.args if isinstance(arg, Constant)}
            constants_by_formula = {**replacements, **constants_by_formula}

    query = ReplaceExpressionWalker(constants_by_formula).walk(query)
    new_args = query.consequent.args + tuple(constants_by_formula.values())
    new_consequent = query.consequent.functor(*new_args)

    query = Implication(new_consequent, query.antecedent)

    return query, constants_by_formula

def _selfjoins_in_pchoices(rule_dnf, symbol_table):
    for formula in rule_dnf.formulas:
        pchoice_functors = []
        for atom in extract_logic_atoms(formula):
            if is_atom_a_probabilistic_choice_relation(atom, symbol_table):
                if atom.functor in pchoice_functors:
                    return True
                else:
                    pchoice_functors.append(atom.functor)

    return False
