from collections import defaultdict
from operator import eq, or_

from neurolang.logic.unification import most_general_unifier

from ..exceptions import NonLiftableException

from ..expression_walker import (
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
)
from ..expressions import Constant, Symbol
from ..logic import Conjunction, Disjunction, Implication
from ..logic.expression_processing import extract_logic_atoms
from ..logic.transformations import MakeExistentialsImplicit
from ..relational_algebra import Projection, Selection, str2columnstr_constant
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation,
)
from .transforms import (
    convert_rule_to_ucq,
    convert_to_dnf_ucq,
    convert_ucq_to_ccq,
)


class ProjectionSelectionByPChoiceConstant(PatternWalker):
    """
    Given a dictionary of {constants: symbols}, this walker is responsible for
    reintroducing the constants in pchoices that were replaced by variables
    to conclude the anti-shattering strategy.
    """

    def __init__(self, constants_by_formula_dict):
        self.constants_by_formula_dict = constants_by_formula_dict

    @add_match(ProvenanceAlgebraSet)
    def match_projection(self, raoperation):
        operation = raoperation.relation
        prov_columns = raoperation.provenance_column
        _or = Constant(or_)
        _eq = Constant(eq)
        symbols_as_columns = []
        for functor, constants_set in self.constants_by_formula_dict.items():
            eq_ops = [
                _eq(str2columnstr_constant(functor.name), constant)
                for constant in constants_set
            ]
            if len(eq_ops) > 1:
                operation = Selection(operation, _or(*eq_ops),)
            else:
                operation = Selection(operation, eq_ops[0],)
            symbols_as_columns.append(str2columnstr_constant(functor.name))
        non_proyected_vars = set(symbols_as_columns).union(set([prov_columns]))
        proyected_vars = raoperation.columns() - non_proyected_vars

        proyected = tuple([prov_columns])
        if len(proyected_vars) > 0:
            for pv in proyected_vars:
                proyected = proyected + tuple([pv])

        operation = Projection(operation, proyected)

        return ProvenanceAlgebraSet(operation, prov_columns)


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

    query_ucq = remove_selfjoins_between_pchoices(query_ucq, symbol_table)

    parameter_variable = defaultdict(Symbol.fresh)

    constants_by_formula = defaultdict(set)
    query_formulas = tuple()
    for clause in query_ucq.formulas:
        constants_by_formula_clause = {}
        # for atom in extract_logic_atoms(clause):
        for atom in clause.formulas:
            if not isinstance(
                atom, Constant
            ) and is_atom_a_probabilistic_choice_relation(atom, symbol_table):
                replacements = {
                    arg: parameter_variable[(atom.functor, i)]
                    for i, arg in enumerate(atom.args)
                    if isinstance(arg, Constant)
                }
                constants_by_formula_clause = {
                    **replacements,
                    **constants_by_formula_clause,
                }
                atom = ReplaceExpressionWalker(
                    constants_by_formula_clause
                ).walk(atom)
            query_formulas += (atom,)
        for k, v in constants_by_formula_clause.items():
            constants_by_formula[v].add(k)

    new_args = query.consequent.args + tuple(constants_by_formula)
    new_consequent = query.consequent.functor(*new_args)

    query_formulas = convert_ucq_to_ccq(
        Conjunction(query_formulas), transformation="CNF"
    )

    query = Implication(
        new_consequent, MakeExistentialsImplicit().walk(query_formulas)
    )

    return query, constants_by_formula


def remove_selfjoins_between_pchoices(rule_dnf, symbol_table):
    """
    Verify the existence of selfjoins between pchoices.

    Parameters
    ----------
    rule_dnf : bool
        Rule to analyze in DNF form.
    symbol_table : Mapping
        Mapping from symbols to probabilistic
        or deterministic sets to solve the query.

    Returns
    -------
    bool
        True if there is a join between pchoices
        in the same conjunction

    """

    disj_formulas = []
    for formula in rule_dnf.formulas:
        conj_formulas = tuple()
        for i1, atom1 in enumerate(extract_logic_atoms(formula)):
            if is_atom_a_probabilistic_choice_relation(atom1, symbol_table):
                for i2, atom2 in enumerate(extract_logic_atoms(formula)):
                    if i1 >= i2 or atom1.functor != atom2.functor:
                        continue

                    mgu = most_general_unifier(atom1, atom2)
                    if mgu is None:
                        conj_formulas += (Constant(False),)
                    else:
                        conj_formulas += (atom1,)  # + igualdad
            else:
                conj_formulas += (atom1,)

        disj_formulas.append(Conjunction(conj_formulas))

    return Disjunction(tuple(disj_formulas))
