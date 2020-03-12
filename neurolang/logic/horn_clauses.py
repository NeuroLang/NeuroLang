from . import (
    Implication,
    Union,
    Conjunction,
    Disjunction,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
    LogicOperator,
    Quantifier,
)
from ..expressions import Constant, Symbol, FunctionApplication
from ..expression_walker import (
    add_match,
    ExpressionWalker,
    ChainedWalker,
    ReplaceSymbolWalker,
)


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


class MoveQuantifiersUp(ExpressionWalker):
    """
    Moves the quantifiers up in order to format the expression
    in prenex normal form. Assumes the expression contains no implications
    and the variables of the quantifiers are not repeated.
    """

    @add_match(Negation(Quantifier))
    def negated_universal(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return quantifier.apply(x, p)

    @add_match(
        Disjunction,
        lambda e: any(isinstance(f, Quantifier) for f in e.formulas),
    )
    def disjunction_with_quantifiers(self, expression):
        quantifiers = []
        formulas = []
        for f in expression.formulas:
            if isinstance(f, Quantifier):
                quantifiers.append(f)
                formulas.append(f.body)
            else:
                formulas.append(f)
        exp = Disjunction(tuple(formulas))
        for q in reversed(quantifiers):
            exp = q.apply(q.head, exp)
        return self.walk(exp)

    @add_match(
        Conjunction,
        lambda e: any(isinstance(f, Quantifier) for f in e.formulas),
    )
    def conjunction_with_quantifiers(self, expression):
        quantifiers = []
        formulas = []
        for f in expression.formulas:
            if isinstance(f, Quantifier):
                quantifiers.append(f)
                formulas.append(f.body)
            else:
                formulas.append(f)
        exp = Conjunction(tuple(formulas))
        for q in reversed(quantifiers):
            exp = q.apply(q.head, exp)
        return self.walk(exp)


class DesambiguateQuantifiedVariables(ExpressionWalker):
    """
    Replaces each quantified variale to a fresh one.
    """

    @add_match(Union)
    def match_union(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Disjunction)
    def match_dijunction(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Conjunction)
    def match_conjunction(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Negation)
    def match_negation(self, expression):
        return expression.apply(self.walk(expression.formula))

    @add_match(Quantifier)
    def match_quantifier(self, expression):
        expression.body = self.walk(expression.body)
        return ReplaceSymbolWalker({expression.head: Symbol.fresh()}).walk(
            expression
        )


class Skolemize(ExpressionWalker):
    """
    Replaces the existential quantifiers and introduces
    Skolem constants for quantified variables. Assumes that
    the expression contains no implications and the quantified
    variables are unique.
    """

    def __init__(self):
        self.used_symbols = []
        self.mapping = []
        self.universally_quantified_variables = []

    def fresh_skolem_constant(self):
        # Replace this by custom subclass of FunctionApplication
        c = Symbol.fresh()
        c.skolem_constant = True
        self.used_symbols.append(c)
        return c

    @add_match(Union)
    def match_union(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Disjunction)
    def match_dijunction(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Conjunction)
    def match_conjunction(self, expression):
        return expression.apply(tuple(map(self.walk, expression.formulas)))

    @add_match(Negation)
    def match_negation(self, expression):
        return expression.apply(self.walk(expression.formula))

    @add_match(UniversalPredicate)
    def universal_quantifier(self, expression):
        self.universally_quantified_variables.append(expression.head)
        new_body = self.walk(expression.body)
        self.universally_quantified_variables.pop()
        return UniversalPredicate(expression.head, new_body)

    @add_match(ExistentialPredicate)
    def existential_quantifier(self, expression):
        c = self.fresh_skolem_constant()
        if self.universally_quantified_variables:
            c = c(*self.universally_quantified_variables)
        self.mapping.append((expression.head, c))
        new_body = self.walk(expression.body)
        self.mapping.pop()
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


class RemoveUniversalPredicates(ExpressionWalker):
    """
    Removes the universal predicates and leaves free the bound variables.
    Assumes that the quantified variables are unique.
    """

    @add_match(UniversalPredicate)
    def match_universal(self, expression):
        return self.walk(expression.body)


class DistributeDisjunctions(ExpressionWalker):
    @add_match(Disjunction, lambda e: len(e.formulas) > 2)
    def match_split(self, expression):
        head, *rest = expression.formulas
        new_exp = Disjunction((head, Disjunction(tuple(rest))))
        return self.walk(new_exp)

    @add_match(Disjunction((..., Conjunction)))
    def match_rotate(self, expression):
        q, c = expression.formulas
        return self.walk(
            Conjunction(tuple(map(lambda p: Disjunction((q, p)), c.formulas)))
        )

    @add_match(Disjunction((Conjunction, ...)))
    def match_distribute(self, expression):
        c, q = expression.formulas
        return self.walk(
            Conjunction(tuple(map(lambda p: Disjunction((p, q)), c.formulas)))
        )


class CollapseDisjunctions(ExpressionWalker):
    @add_match(
        Disjunction,
        lambda e: any(isinstance(f, Disjunction) for f in e.formulas),
    )
    def match_disjunction(self, e):
        new_arg = []
        for f in map(self.walk, e.formulas):
            if isinstance(f, Disjunction):
                new_arg.extend(f.formulas)
            else:
                new_arg.append(f)
        return Disjunction(tuple(new_arg))


class CollapseConjunctions(ExpressionWalker):
    @add_match(
        Conjunction,
        lambda e: any(isinstance(f, Conjunction) for f in e.formulas),
    )
    def match_conjunction(self, e):
        new_arg = []
        for f in map(self.walk, e.formulas):
            if isinstance(f, Conjunction):
                new_arg.extend(f.formulas)
            else:
                new_arg.append(f)
        return Conjunction(tuple(new_arg))


def convert_to_pnf_with_cnf_matrix(expression):
    walker = ChainedWalker(
        EliminateImplications,
        MoveNegationsToAtoms,
        DesambiguateQuantifiedVariables,
        MoveQuantifiersUp,
        DistributeDisjunctions,
        CollapseDisjunctions,
        CollapseConjunctions,
    )
    return walker.walk(expression)


class HornClause(LogicOperator):
    """Expression of the form `P(X) :- Q(X), S(X).`"""

    def __init__(self, head, body):
        self.head = head
        self.body = body

        self._symbols = head.symbols or set()
        if body is not None:
            for l in body:
                self._symbols |= l._symbols

    def __repr__(self):
        r = 'HornClause{'
        if self.head is not None:
            r += repr(self.head)
            if self.body is not None:
                r += ' :- '
        else:
            r += '?- '

        if self.body is not None:
            r += ', '.join(repr(l) for l in self.body)
        r += '.}'
        return r
