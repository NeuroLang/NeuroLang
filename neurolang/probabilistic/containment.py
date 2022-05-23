import logging

from neurolang.expression_walker import expression_iterator

from ..datalog import Fact, chase
from ..datalog.negation import DatalogProgramNegation
from ..expressions import Constant, Symbol
from ..logic import Disjunction, Implication, Negation, Union
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    MakeExistentialsImplicit,
    PushExistentialsDown,
    RemoveUniversalPredicates,
    convert_to_pnf_with_dnf_matrix
)

LOG = logging.getLogger(__name__)


__all__ = ['is_contained_rule', 'is_contained']


def canonical_database_program(rule):
    '''
    Given a datalog rule R(X):-S_1(X,Y),...,S_N(X,Y) where
    X and Y are sets of symbols, produce the datalog program corresponding
    to the facts
    S_1(A,B).
    S_2(A,B).
    .
    .
    .
    S_N(A,B).
    where A and B are mappings from the symbols in X and Y to constants.
    '''
    consequent, antecedent = rule.unapply()
    return Union(tuple(
        Fact(freeze_atom(atom))
        for atom in extract_logic_atoms(antecedent)
    )), freeze_atom(consequent)


def freeze_atom(atom):
    '''
    Converts all symbols of the atom
    to constants with a string representing
    the symbol.
    '''
    args = (
        Constant(arg.name) if isinstance(arg, Symbol) else arg
        for arg in atom.args
    )
    return atom.functor(*args)


def is_contained_rule(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 Datalog rules without constants, computes wether
    q1←q2.
    '''
    s = Symbol.fresh()
    q1 = Implication(
        s(*q1.consequent.args), q1.antecedent
    )
    q2 = Implication(
        s(*q2.consequent.args), q2.antecedent
    )
    d_q2, frozen_head = canonical_database_program(q2)
    dp = DatalogProgramNegation()
    for f in d_q2.formulas:
        dp.walk(f)
    dp.walk(q1)
    solution = chase.Chase(dp).build_chase_solution()
    return frozen_head in solution.as_set()


def is_contained(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive positive ∃ logic queries,
    without constants, computes wether q1←q2.

    caveat: the free variables of q1 and q2 should be
    the same, else the queries are judge as not contained.
    '''
    s = Symbol.fresh()
    programs = []
    if (
        extract_logic_free_variables(q1) !=
        extract_logic_free_variables(q2)
    ):
        LOG.debug("The free variables are different")
        return False

    if any(
        isinstance(e, Negation)
        for _, e in expression_iterator(Union((q1, q2)))
    ):
        return False

    for query in (q1, q2):
        program = convert_pos_logic_query_to_datalog_rules(query, s)
        LOG.debug("Canonical program for %s", query)
        programs.append(program)

    for q2_ in programs[1]:
        for q1_ in programs[0]:
            if is_contained_rule(q1_, q2_):
                break
        else:
            return False
    return True


def convert_pos_logic_query_to_datalog_rules(query, head):
    '''
    Converts a positive ∃ logic query without constants
    to a list of datalog rules.
    '''
    ruq = RemoveUniversalPredicates()
    mei = MakeExistentialsImplicit()
    ped = PushExistentialsDown()
    q_args = set(extract_logic_free_variables(query))
    antecedent = ped.walk(ruq.walk(convert_to_pnf_with_dnf_matrix(query)))
    if isinstance(antecedent, Disjunction):
        program = antecedent.formulas
    else:
        program = (antecedent,)
    program = [
        Implication(
            head(*(q_args & extract_logic_free_variables(formula))),
            mei.walk(formula)
        )
        for formula in program
    ]
    return program
