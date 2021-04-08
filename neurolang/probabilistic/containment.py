from ..datalog import DatalogProgram, Fact, chase
from ..expressions import Constant, Symbol
from ..logic import Disjunction, Implication, Union
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    MakeExistentialsImplicit,
    convert_to_pnf_with_dnf_matrix
)


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
    args = (
        Constant(s.name)
        for s in atom.args
    )
    return atom.functor(*args)


def is_contained_rule(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive Datalog queries, computes wether
    q1←q2.
    '''
    s = Symbol.fresh()
    q1 = Implication(
        s(*q1.consequent.args), q1.antecedent
    )
    q2 = Implication(
        s(*q2.consequent.args), q2.antecedent
    )
    d_q1, frozen_head = canonical_database_program(q1)
    dp = DatalogProgram()
    for f in d_q1.formulas:
        dp.walk(f)
    dp.walk(q2)
    solution = chase.Chase(dp).build_chase_solution()
    contained = (
        frozen_head.functor in solution and
        (
            tuple(a.value for a in frozen_head.args)
            in solution[frozen_head.functor].value.unwrap()
        )
    )
    return contained


def is_contained(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive Datalog queries, computes wether
    q1←q2.
    '''
    s = Symbol.fresh()
    if not isinstance(q1, Implication):
        q1 = Implication(
            s(*extract_logic_free_variables(q1)),
            q1
        )
    if not isinstance(q2, Implication):
        q2 = Implication(
            s(*extract_logic_free_variables(q2)),
            q2
        )
    args1 = set(q1.consequent.args)
    args2 = set(q2.consequent.args)
    mei = MakeExistentialsImplicit()
    antecedent1 = mei.walk(convert_to_pnf_with_dnf_matrix(q1.antecedent))
    antecedent2 = mei.walk(convert_to_pnf_with_dnf_matrix(q2.antecedent))
    if not isinstance(antecedent1, Disjunction):
        antecedent1 = Disjunction((antecedent1,))
    if not isinstance(antecedent2, Disjunction):
        antecedent2 = Disjunction((antecedent2,))

    return all(
        any(
            is_contained_rule(
                Implication(
                    s(*(args1 & extract_logic_free_variables(q1_))), q1_
                ),
                Implication(
                    s(*(args2 & extract_logic_free_variables(q2_))), q2_
                )
            )
            for q1_ in antecedent1.formulas
        )
        for q2_ in antecedent2.formulas
    )
