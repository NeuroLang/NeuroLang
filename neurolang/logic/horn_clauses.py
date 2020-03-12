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
from ..exceptions import NeuroLangException
from ..expressions import Symbol, FunctionApplication
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
        return expression.apply(self.walk(expression.formulas))

    @add_match(Disjunction)
    def match_dijunction(self, expression):
        return expression.apply(self.walk(expression.formulas))

    @add_match(Conjunction)
    def match_conjunction(self, expression):
        return expression.apply(self.walk(expression.formulas))

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
        return expression.apply(self.walk(expression.formulas))

    @add_match(Disjunction)
    def match_dijunction(self, expression):
        return expression.apply(self.walk(expression.formulas))

    @add_match(Conjunction)
    def match_conjunction(self, expression):
        return expression.apply(self.walk(expression.formulas))

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
    """
    Expression of the form `P(X) \u2190 (Q(X) \u22C0 ... \u22C0 S(X))`
    """

    def __init__(self, head, body=None):
        self._validate(head, body)
        self.head = head
        self.body = body
        self._symbols = head._symbols or set()
        if body is not None:
            for l in body:
                self._symbols |= l._symbols

    def _validate(self, head, body):
        if not (head is None or self._is_literal(head)):
            raise NeuroLangException(
                f"Head must be a literal or None, {head} given"
            )
        if not head and not body:
            raise NeuroLangException(f"Head and body can not both be empty")
        if not (
            body is None
            or (
                isinstance(body, tuple)
                and all(self._is_literal(l) for l in body)
            )
        ):
            raise NeuroLangException(
                f"Body must be a tuple or None, {body} given"
            )

    def _is_literal(self, exp):
        return isinstance(exp, FunctionApplication) or isinstance(exp, Symbol)

    def __repr__(self):
        r = "HornClause{"
        if self.head is not None:
            r += repr(self.head)
            if self.body is not None:
                r += " :- "
        else:
            r += "?- "

        if self.body is not None:
            r += ", ".join(repr(l) for l in self.body)
        r += ".}"
        return r


def convert_to_horn_clauses(expression):
    expression = convert_to_pnf_with_cnf_matrix(expression)
    innermost_quantifier = None
    matrix = expression
    while isinstance(matrix, Quantifier):
        innermost_quantifier = matrix
        matrix = matrix.body

    if isinstance(matrix, Conjunction):
        clauses = Union(tuple(map(_build_clause, matrix.formulas)))
    else:
        clauses = Union((_build_clause(matrix),))

    if innermost_quantifier:
        innermost_quantifier.body = clauses
        return expression
    return clauses


def _build_clause(exp):
    if isinstance(exp, Disjunction):
        head, body = _extract_head_and_body(exp)
        return HornClause(head, body)

    elif isinstance(exp, Negation):
        return HornClause(None, (exp,))
    return HornClause(exp, None)


def _extract_head_and_body(exp):
    positive = [f for f in exp.formulas if not isinstance(f, Negation)]
    if len(positive) > 1:
        raise NeuroLangException(
            f"{exp} contains more than one positive literal"
        )
    negative = [f.formula for f in exp.formulas if isinstance(f, Negation)]
    return (
        positive[0] if positive else None,
        tuple(negative) if negative else None,
    )
