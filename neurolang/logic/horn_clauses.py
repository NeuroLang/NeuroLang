from . import (
    Implication,
    Conjunction,
    Disjunction,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
)
from ..expressions import Constant, Symbol, FunctionApplication
from ..expression_walker import ExpressionWalker, add_match


class EliminateImplications(ExpressionWalker):
    """
    Removes the implication ocurrences of an expression.

    """

    @add_match(Implication(..., ...))
    def remove_implication(self, implication):
        c = self.walk(implication.consequent)
        a = self.walk(implication.antecedent)
        return Disjunction((c, Negation(a)))


class MoveNegationsToAtoms(ExpressionWalker):
    """
    Moves the negations the furthest possible to the atoms.
    Assumes that there are no implications in the expression.
    """

    @add_match(Negation(UniversalPredicate(..., ...)))
    def negated_universal(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return ExistentialPredicate(x, p)

    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return UniversalPredicate(x, p)

    @add_match(Negation(Conjunction(...)))
    def negated_conjunction(self, negation):
        conj = negation.formula
        formulas = map(lambda e: self.walk(Negation(e)), conj.formulas)
        return Disjunction(tuple(formulas))

    @add_match(Negation(Disjunction(...)))
    def negated_disjunction(self, negation):
        disj = negation.formula
        formulas = map(lambda e: self.walk(Negation(e)), disj.formulas)
        return Conjunction(tuple(formulas))

    @add_match(Negation(Negation(...)))
    def negated_negation(self, negation):
        return negation.formula.formula


class Skolemize(ExpressionWalker):
    """
    Replaces the existential quantifiers and introduces
    Skolem constants for quantified variables.
    """

    def __init__(self):
        self.used_symbols = []
        self.mapping = []
        self.universally_quantified_variables = []

    def fresh_skolem_constant(self):
        c = Symbol.fresh()
        c.skolem_constant = True
        self.used_symbols.append(c)
        return c

    @add_match(UniversalPredicate)
    def universal_quantifier(self, expression):
        self.universally_quantified_variables.append(expression.head)
        new_body = self.walk(expression.body)
        self.universally_quantified_variables.remove(expression.head)
        return UniversalPredicate(expression.head, new_body)

    @add_match(ExistentialPredicate)
    def existential_quantifier(self, expression):
        c = self.fresh_skolem_constant()
        if self.universally_quantified_variables:
            c = c(*self.universally_quantified_variables)
        self.mapping.append((expression.head, c))
        new_body = self.walk(expression.body)
        self.mapping.remove((expression.head, c))
        return new_body

    @add_match(Symbol)
    def _match_symbol(self, symbol):
        for s, r in reversed(self.mapping):
            if s == symbol:
                return r
        return symbol

    @add_match(
        FunctionApplication, lambda e: hasattr(e.functor, "skolem_constant")
    )
    def _match_function_application(self, f):
        return f
