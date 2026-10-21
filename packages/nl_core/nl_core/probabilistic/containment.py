import logging

from ..datalog import Fact, chase
from ..datalog.negation import DatalogProgramNegation
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    Implication,
    Negation,
    Union,
)
from nl_expr.logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables,
)
from nl_expr.logic.transformations import (
    MakeExistentialsImplicit,
    PushExistentialsDown,
    RemoveUniversalPredicates,
    convert_to_pnf_with_dnf_matrix,
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


class NegatedSymbol(Symbol):
    """
    The relation of the negated atoms ¬R(x) of the relation symbol
    *expression* R, a relation of its own, that no atom R(x) matches.
    """
    def __init__(self, expression):
        self.expression = expression
        self.is_fresh = False

    @property
    def name(self):
        return self.expression.name

    def __eq__(self, other):
        return (
            isinstance(other, NegatedSymbol)
            and self.expression == other.expression
        )

    def __hash__(self):
        return hash((NegatedSymbol, self.expression))

    def __str__(self):
        return f"¬{self.expression.name}"

    __repr__ = __str__


def negation_as_atom(formula):
    """
    The atom ¬R(x) of a negated atom is the atom R̄(x) of the relation
    `NegatedSymbol(R)`; any other formula is as it is.
    """
    if isinstance(formula, Negation) and isinstance(
        formula.formula.functor, Symbol
    ):
        atom = formula.formula
        return NegatedSymbol(atom.functor)(*atom.args)
    return formula


def negations_as_atoms(rule):
    '''
    The *rule* with each negated atom ¬R(x) of its body replaced by the
    atom R̄(x) of the relation `NegatedSymbol(R)`.

    Containment is tested by the frozen body of a query, whose atoms are
    facts, and by the evaluation of the other query on them. A negated
    atom is not a fact, and not being in the facts is not the negation
    holding. With R̄ a relation of its own, a negated atom of a query is
    matched by a negated atom of the other, as the homomorphisms of the
    containment of positive queries do: the test stays sound (a
    containment found holds), and is not complete.
    '''
    antecedent = rule.antecedent
    if isinstance(antecedent, Conjunction):
        antecedent = Conjunction(tuple(
            negation_as_atom(formula) for formula in antecedent.formulas
        ))
    else:
        antecedent = negation_as_atom(antecedent)
    return Implication(rule.consequent, antecedent)


def is_contained_rule(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 Datalog rules without constants, computes whether
    q1←q2. A negated atom is matched by the same negated atom
    (see :func:`negations_as_atoms`).
    '''
    q1, q2 = negations_as_atoms(q1), negations_as_atoms(q2)
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
    if len(frozen_head.args) == 0:
        # A nullary fact is derived when its relation is not empty; its
        # membership in the facts of the solution is not tested, as their
        # type differs from that of the frozen head.
        relation = solution.get(frozen_head.functor)
        return relation is not None and not relation.value.is_empty()
    return frozen_head in solution.as_set()


def is_contained(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive positive ∃ logic queries,
    without constants, computes whether q1←q2.

    The free variables are treated as constants. Each disjunct of
    q2 must imply a disjunct of q1. A disjunct d2 implies a disjunct
    d1 if the free variables of d1 are included in those of d2 and
    ∃(fv(d2) - fv(d1)) d2 implies d1. This test is sound, and complete
    when the queries have the same free variables in every disjunct.
    '''
    s = Symbol.fresh()
    programs = []
    for query in (q1, q2):
        program = convert_pos_logic_query_to_datalog_rules(query, s)
        LOG.debug("Canonical program for %s", query)
        programs.append(program)

    for q2_ in programs[1]:
        for q1_ in programs[0]:
            if _disjunct_implies(q2_, q1_):
                break
        else:
            return False
    return True


def _disjunct_implies(q2_, q1_):
    head_args = q1_.consequent.args
    if not set(head_args) <= set(q2_.consequent.args):
        return False
    q2_ = Implication(q2_.consequent.functor(*head_args), q2_.antecedent)
    return is_contained_rule(q1_, q2_)


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
    # The head arguments are sorted so that the rules of two queries
    # with the same free variables have the same head.
    program = [
        Implication(
            head(*sorted(
                q_args & extract_logic_free_variables(formula),
                key=lambda variable: variable.name
            )),
            mei.walk(formula)
        )
        for formula in program
    ]
    return program
