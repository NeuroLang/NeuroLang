"""
Translation from First Order Logic to Horn clauses.

This module defines the HornClause expression class and algorithms to translate
First Order Logic expressions to a union of Horn clauses.  Between the
algorithms here are implemented a translation from FOL sentences to an
equivalent expression in safe range normal form, how to obtain the set of range
restricted variables of an expression, how to check if the expression is safe
range and, if so, how to translate the expression to a union of horn clauses.
Furthermore the resulting horn clauses can be translated to a datalog
expression block to use in a solver. The theory behind this is mostly taken
from the chapter 4 of [1]_.

.. [1] Abiteboul, Hull, and Vianu, Foundations of Databases: The Logical Level.
"""
import operator
from functools import reduce
from typing import Callable

from ..datalog.expressions import Fact
from ..datalog.negation import is_conjunctive_negation
from ..exceptions import NeuroLangException
from ..expression_walker import ChainedWalker, PatternWalker, add_match
from ..expressions import ExpressionBlock
from ..type_system import Unknown, is_leq_informative
from . import (
    FALSE,
    TRUE,
    Conjunction,
    Constant,
    Disjunction,
    ExistentialPredicate,
    FunctionApplication,
    Implication,
    LogicOperator,
    Negation,
    Quantifier,
    Symbol,
    Union
)
from .expression_processing import extract_logic_free_variables
from .transformations import (
    DesambiguateQuantifiedVariables,
    EliminateImplications,
    LogicExpressionWalker,
    MoveNegationsToAtoms,
    RemoveUniversalPredicates
)


class NeuroLangTranslateToHornClauseException(NeuroLangException):
    pass


class HornClause(LogicOperator):
    """
    Expression of the form `P(X) \u2190 (Q(X) \u22C0 ... \u22C0 S(X))`
    """

    def __init__(self, head, body):
        self._validate(head, body)
        self.head = head
        self.body = body
        self._symbols = head._symbols | body._symbols

    def _validate(self, head, body):
        if not self._is_valid_as_head(head):
            raise NeuroLangException(
                f"Head must be a literal or FALSE, {head} given"
            )

        if not self._is_valid_as_body(body):
            raise NeuroLangException(
                f"Body must be a Conjunction of literals, a single"
                " literal or TRUE, {body} given"
            )

    def _is_valid_as_head(self, exp):
        return exp == FALSE or self._is_literal(exp)

    def _is_valid_as_body(self, exp):
        return (
            exp == TRUE
            or self._is_literal(exp)
            or (
                isinstance(exp, Conjunction)
                and all(self._is_literal(l) for l in exp.formulas)
            )
        )

    def _is_literal(self, exp):
        return (
            isinstance(exp, FunctionApplication)
            or isinstance(exp, Symbol)
            or isinstance(exp, Constant)
            or (
                isinstance(exp, Negation)
                and (
                    isinstance(exp.formula, FunctionApplication)
                    or isinstance(exp.formula, Symbol)
                    or isinstance(exp.formula, Constant)
                )
            )
        )

    def __repr__(self):
        r = "HornClause{"
        if self.head is not None:
            r += repr(self.head)
            if self.body is not None:
                r += " :- "
        else:
            r += "?- "

        if self.body != TRUE:
            r += repr(self.body)
        r += ".}"
        return r


class HornFact(HornClause):
    """
    Expression of the form `P(X) \u2190 TRUE`
    """

    def __init__(self, head):
        if head == FALSE:
            raise NeuroLangException("A HornFact can not have FALSE as head.")
        super().__init__(head, TRUE)

    def __repr__(self):
        r = "HornFact{"
        if self.head is not None:
            r += repr(self.head)
        r += ".}"
        return r


class MoveNegationsToAtomsOrExistentialQuantifiers(MoveNegationsToAtoms):
    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        h = quantifier.head
        b = self.walk(quantifier.body)
        exp = Negation(ExistentialPredicate(h, b))
        if b != quantifier.body:
            exp = self.walk(exp)
        return exp


def convert_to_srnf(e):
    """
    Convert an expression to safe range normal form.

    Safe range normal form is an equivalent expression
    but where the there are no implications or universal
    quantifiers and also the negations are only applied
    over atoms or existential quantifiers.
    """
    w = ChainedWalker(
        DesambiguateQuantifiedVariables,
        EliminateImplications,
        RemoveUniversalPredicates,
        MoveNegationsToAtomsOrExistentialQuantifiers,
    )
    return w.walk(e)


def is_safe_range(expression):
    """
    Return true if an expression is safe range.

    This function receives an expression in safe range
    normal form and returns true if all its free variables
    are range restricted.
    """
    try:
        return extract_logic_free_variables(
            expression
        ) == range_restricted_variables(expression)
    except NeuroLangException:
        return False


def range_restricted_variables(e):
    return RangeRestrictedVariables().walk(e)


class RangeRestrictedVariables(LogicExpressionWalker):
    """
    The set of variables which are range restricted in a expression.

    The range restricted variables are defined recursively as follows:
    - For atoms: The variables occurring in the arguments.
    - For negation: The empty set.
    - For existential predicates: The restricted variables of the body without
      the head.
    - For conjunction: The union of the restricted variables of the involved
      expressions.
    - For disjunction: The intersection of the restricted variables of the
      involved expressions.

    Also, if an existentially quantified variable is not restricted in the body
    of its predicate then the expression does not have a restricted range. This
    also counts for existentials inside negations.
    """

    @add_match(FunctionApplication)
    def function(self, exp):
        return set([a for a in exp.args if isinstance(a, Symbol)])

    @add_match(Negation)
    def negation(self, exp):
        self.walk(exp.formula)
        return set()

    @add_match(Disjunction)
    def disjunction(self, exp):
        return set.intersection(*self.walk(exp.formulas))

    @add_match(Conjunction)
    def conjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(ExistentialPredicate)
    def existential(self, exp):
        r = self.walk(exp.body)
        if exp.head in r:
            return r - {exp.head}
        # This better could be a return value
        raise NeuroLangException(
            "Some quantified variable is not range restricted"
        )


def convert_srnf_to_horn_clauses(head, expression):
    """
    Converts a safe range query into an union of horn clauses.

    Given a query represented by an _answer_ head and a range restricted
    expression in safe range normal form, returns an union of horn clauses in
    which the result set given to head is equivalent. Also, it is required that
    all the variables appear free in the body.

    The algorithm is implemented using a stack of queries to process, which
    starts with the query given as parameter. In each iteration, a query is
    processed and the result is a Horn clause with a remainder of auxiliar
    queries which appear in the clause. The clause is then added to the result
    and the remainder queries are added to the stack.

    The cases where auxiliar queries are needed are the negation over an
    existential predicate and the disjunction. For the disjunction an auxiliar
    query is introduced for each formula in the disjunction, but all with the
    same head. For a negated existential another query is added because
    negation can only be applied over atoms.

    Care must be taken to ensure that the auxiliar queries are also range
    restricted. To do so, a set of the positive atoms which appear in the
    parent expressions is added alongside each query in the stack. Those
    atoms are precisely the ones that restrict the range of the variables. If
    is the case that a query is not range restricted then some of those atoms
    are added to the body of the query to ensure the range restriction. This
    restricts the result set of the auxiliar queries but does not alter the
    equivalence for the overall query because the range of those variables was
    already restricted in some of its parents.
    """

    if not is_safe_range(expression):
        raise NeuroLangTranslateToHornClauseException(
            "Expression is not safe range: {}".format(expression)
        )
    if not set(head.args) <= range_restricted_variables(expression):
        raise NeuroLangTranslateToHornClauseException(
            "Variables in head ({}) must be present in body ({})".format(
                head, expression
            )
        )

    stack = [(head, expression, set())]
    processed = []

    while stack:
        head, exp, restrictive_atoms = stack.pop()
        body, remainder = ConvertSRNFToHornClause().walk(exp)
        body = _restrict_variables(head, body, restrictive_atoms)
        restrictive_atoms |= _restrictive_atoms(body)
        remainder = [r + (restrictive_atoms,) for r in remainder]
        processed.append(_to_horn_clause(head, body))
        stack += remainder

    return Union(tuple(reversed(processed)))


def _to_horn_clause(head, body):
    if len(body) == 0:
        r = HornFact(head)
    elif len(body) == 1:
        r = HornClause(head, body[0])
    else:
        r = HornClause(head, Conjunction(body))
    return r


def _restrict_variables(head, body, restrictive_atoms):
    while not set(head.args).issubset(_restricted_variables(body)):
        uv = set(head.args) - _restricted_variables(body)
        new_atoms = _choose_restriction_atoms(uv, restrictive_atoms, head)
        body = new_atoms + body
    return body


def _choose_restriction_atoms(unrestricted_variables, available_atoms, head):
    x = list(unrestricted_variables)[0]
    valid_choices = [
        (a, (set(a.args) - set(head.args)))
        for a in available_atoms
        if (x in a.args) and (a != head)
    ]
    valid_choices = sorted(valid_choices, key=lambda t: t[1])
    a, _variables_not_in_head = valid_choices[0]
    return (a,)


def _restricted_variables(body):
    r = set()
    for a in _restrictive_atoms(body):
        r |= _atom_variables(a)
    return r


def _restrictive_atoms(atoms):
    return set(a for a in atoms if _is_restriction(a))


def _is_restriction(atom):
    if isinstance(atom, FunctionApplication):
        if is_leq_informative(atom.functor.type, Unknown):
            return True
        elif atom.functor == operator.eq and any(
            isinstance(a, Constant) for a in atom.args
        ):
            return True
        return not is_leq_informative(atom.functor.type, Callable)
    return False


def _atom_variables(atom):
    return set(s for s in atom.args if isinstance(s, Symbol))


class ConvertSRNFToHornClause(PatternWalker):
    """
    Converts a expression in safe range normal form into the atoms of a horn
    clause and a list of auxiliar queries remaining to be processed.

    See `convert_srnf_to_horn_clauses`.
    """

    @add_match(Conjunction)
    def conjunction(self, exp):
        bodies, remainders = zip(*map(self.walk, exp.formulas))
        return (
            tuple(
                sorted(reduce(operator.add, bodies), key=self._negation_order)
            ),
            reduce(operator.add, remainders),
        )

    def _negation_order(self, exp):
        return 1 if isinstance(exp, Negation) else 0

    @add_match(Disjunction)
    def disjunction(self, exp):
        nh = self._new_head_for(exp)
        return (nh,), [(nh, f) for f in exp.formulas]

    @add_match(ExistentialPredicate)
    def existential(self, exp):
        return self.walk(exp.body)

    @add_match(Negation(FunctionApplication))
    def negated_atom(self, exp):
        return (exp,), []

    @add_match(Negation(ExistentialPredicate))
    def negated_existential(self, exp):
        nh = self._new_head_for(exp.formula)
        return (Negation(nh),), [(nh, exp.formula)]

    @add_match(FunctionApplication)
    def atom(self, exp):
        return (exp,), []

    @add_match(...)
    def unknown(self, exp):
        raise NeuroLangTranslateToHornClauseException(
            "Expression not in safe range normal form: {}".format(exp)
        )

    def _new_head_for(self, exp):
        fv = extract_logic_free_variables(exp)
        S = Symbol.fresh()
        return S(*tuple(fv))


def translate_horn_clauses_to_datalog(horn_clauses):
    """
    Straightforward translation from Horn clauses a datalog expression block.
    """
    return TranslateHornClausesToDatalog().walk(horn_clauses)


class TranslateHornClausesToDatalog(LogicExpressionWalker):
    @add_match(Union)
    def union(self, exp):
        return ExpressionBlock(self.walk(exp.formulas))

    @add_match(Quantifier)
    def quantifier(self, exp):
        return self.walk(exp.body)

    @add_match(HornFact)
    def horn_fact(self, exp):
        return Fact(exp.head)

    @add_match(HornClause)
    def horn_rule(self, exp):
        return Implication(exp.head, self.walk(exp.body))


def fol_query_to_datalog_program(head, exp):
    """
    Returns a datalog program for a given query in first order logic.

    Given a query represented by an _answer_ head and an expression in first
    order logic, converts the expression to safe range normal form and then, if
    the query is safe range, returns an ExpressionBlock with the equivalent
    program in datalog. Throw a NeuroLangTranslateToHornClauseException
    otherwise.
    """
    exp = convert_to_srnf(exp)
    horn_clauses = convert_srnf_to_horn_clauses(head, exp)
    program = translate_horn_clauses_to_datalog(horn_clauses)
    return program


class Fol2DatalogTranslationException(NeuroLangException):
    pass


class Fol2DatalogMixin(PatternWalker):
    @add_match(
        Implication, lambda imp: not is_conjunctive_negation(imp.antecedent)
    )
    def translate_implication(self, imp):
        try:
            program = fol_query_to_datalog_program(
                imp.consequent, imp.antecedent
            )
        except NeuroLangException as e:
            raise Fol2DatalogTranslationException from e
        return self.walk(program)
