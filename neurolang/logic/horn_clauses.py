
from . import (Implication, Conjunction, Disjunction, Negation,
               UniversalPredicate, ExistentialPredicate)
from ..expression_walker import IdentityWalker, add_match


class EliminateImplications(IdentityWalker):
    '''
    Removes the implication ocurrences of an expression.

    '''
    @add_match(Implication(..., ...))
    def remove_implication(self, implication):
        c = self.walk(implication.consequent)
        a = self.walk(implication.antecedent)
        return Disjunction((c, Negation(a)))


class MoveNegationsToAtoms(IdentityWalker):
    '''
    Moves the negations the furthest possible to the atoms.
    Assumes that there are no implications in the expression.
    '''
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
        return Disjunction(tuple(map(
            lambda e: self.walk(Negation(e)),
            conj.formulas
        )))

    @add_match(Negation(Disjunction(...)))
    def negated_disjunction(self, negation):
        disj = negation.formula
        return Conjunction(tuple(map(
            lambda e: self.walk(Negation(e)),
            disj.formulas
        )))

    @add_match(Negation(Negation(...)))
    def negated_negation(self, negation):
        return negation.formula.formula
