from functools import reduce
from operator import add
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
from ..expressions import Symbol, Constant, FunctionApplication
from ..expression_walker import (
    add_match,
    PatternWalker,
    ChainedWalker,
    ReplaceSymbolWalker,
)


class LogicExpressionWalker(PatternWalker):
    @add_match(Quantifier)
    def walk_quantifier(self, expression):
        return expression.apply(
            self.walk(expression.head), self.walk(expression.body)
        )

    @add_match(Constant)
    def walk_constant(self, expression):
        return expression

    @add_match(Symbol)
    def walk_symbol(self, expression):
        return expression

    @add_match(Negation)
    def walk_negation(self, expression):
        return Negation(self.walk(expression.formula))

    @add_match(FunctionApplication)
    def walk_function(self, expression):
        return FunctionApplication(
            expression.functor, tuple(map(self.walk, expression.args))
        )

    @add_match(Union)
    def walk_union(self, expression):
        return expression.apply(self.walk(expression.formulas))

    @add_match(Disjunction)
    def walk_disjunction(self, expression):
        return expression.apply(map(self.walk, expression.formulas))

    @add_match(Conjunction)
    def walk_conjunction(self, expression):
        return expression.apply(map(self.walk, expression.formulas))


class EliminateImplications(LogicExpressionWalker):
    """
    Removes the implication ocurrences of an expression.
    """

    @add_match(Implication(..., ...))
    def remove_implication(self, implication):
        c = self.walk(implication.consequent)
        a = self.walk(implication.antecedent)
        return Disjunction((c, Negation(a)))


class MoveNegationsToAtoms(LogicExpressionWalker):
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


class MoveQuantifiersUp(LogicExpressionWalker):
    """
    Moves the quantifiers up in order to format the expression
    in prenex normal form. Assumes the expression contains no implications
    and the variables of the quantifiers are not repeated.
    """

    @add_match(Negation(UniversalPredicate(..., ...)))
    def negated_universal(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        return self.walk(ExistentialPredicate(x, Negation(quantifier.body)))

    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        return self.walk(UniversalPredicate(x, Negation(quantifier.body)))

    @add_match(Disjunction)
    def disjunction_with_quantifiers(self, expression):
        expression = self.walk_disjunction(expression)
        if not any(isinstance(f, Quantifier) for f in expression.formulas):
            return expression

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

    @add_match(Conjunction)
    def conjunction_with_quantifiers(self, expression):
        expression = self.walk_conjunction(expression)
        if not any(isinstance(f, Quantifier) for f in expression.formulas):
            return expression

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


class DesambiguateQuantifiedVariables(LogicExpressionWalker):
    """
    Replaces each quantified variale to a fresh one.
    """

    @add_match(Implication)
    def match_implication(self, expression):
        return expression.apply(
            self.walk(expression.consequent), self.walk(expression.antecedent)
        )

    @add_match(Union)
    def match_union(self, expression):
        return expression.apply(self.walk(expression.formulas))

    @add_match(Disjunction)
    def match_disjunction(self, expression):
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


class Skolemize(LogicExpressionWalker):
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
    def match_disjunction(self, expression):
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


class DistributeDisjunctions(LogicExpressionWalker):
    @add_match(Disjunction, lambda e: len(e.formulas) > 2)
    def match_split(self, expression):
        head, *rest = expression.formulas
        rest = self.walk(Disjunction(tuple(rest)))
        new_exp = Disjunction((head, rest))
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


class CollapseDisjunctions(LogicExpressionWalker):
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


class CollapseConjunctions(LogicExpressionWalker):
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
        # DesambiguateQuantifiedVariables,
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
                f"Body must be a tuple of literals or None, {body} given"
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


class RemoveUniversalPredicates(LogicExpressionWalker):
    """
    Removes the universal predicates and leaves free the bound variables.
    Assumes that the quantified variables are unique.
    """

    @add_match(UniversalPredicate)
    def match_universal(self, expression):
        return Negation(
            ExistentialPredicate(
                expression.head, Negation(self.walk(expression.body))
            )
        )


class MoveNegationsToAtomsOrExistentialQuantifiers(LogicExpressionWalker):
    @add_match(Implication)
    def match_implication(self, expression):
        return expression.apply(
            self.walk(expression.consequent), self.walk(expression.antecedent)
        )

    @add_match(Negation(UniversalPredicate(..., ...)))
    def negated_universal(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return ExistentialPredicate(x, p)

    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        h = quantifier.head
        b = self.walk(quantifier.body)
        return Negation(ExistentialPredicate(h, b))

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


def convert_to_srnf(e):
    # e = DesambiguateQuantifiedVariables().walk(e)
    e = EliminateImplications().walk(e)
    e = RemoveUniversalPredicates().walk(e)
    e = MoveNegationsToAtomsOrExistentialQuantifiers().walk(e)
    return e


def is_safe_range(expression):
    try:
        return free_variables(expression) == range_restricted_variables(
            expression
        )
    except NeuroLangException:
        return False


def range_restricted_variables(e):
    return RangeRestrictedVariables().walk(e)


class RangeRestrictedVariables(LogicExpressionWalker):
    @add_match(FunctionApplication)
    def match_function(self, exp):
        return set([a for a in exp.args if isinstance(a, Symbol)])

    @add_match(Negation)
    def match_negation(self, exp):
        return set()

    @add_match(Disjunction)
    def match_disjunction(self, exp):
        return set.intersection(*self.walk(exp.formulas))

    @add_match(Conjunction)
    def match_conjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(UniversalPredicate)
    def match_universal(self, exp):
        return set()

    @add_match(ExistentialPredicate)
    def match_existential(self, exp):
        r = self.walk(exp.body)
        if exp.head in r:
            return r - {exp.head}
        # This better could be a return value
        raise NeuroLangException(
            "Some quantified variable is not range restricted"
        )


def free_variables(expression):
    return FreeVariables().walk(expression)


class FreeVariables(PatternWalker):
    @add_match(FunctionApplication)
    def match_function(self, exp):
        return set([a for a in exp.args if isinstance(a, Symbol)])

    @add_match(Negation)
    def match_negation(self, exp):
        return self.walk(exp.formula)

    @add_match(Disjunction)
    def match_disjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(Implication)
    def match_implication(self, exp):
        return set.union(self.walk(exp.antecedent), self.walk(exp.consequent))

    @add_match(Conjunction)
    def match_conjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(Quantifier)
    def match_quantifier(self, exp):
        return self.walk(exp.body) - {exp.head}


def convert_srnf_to_horn_clauses(head, expression):
    queue = [(head, expression)]
    processed = []

    while queue:
        head, exp = queue.pop()
        body, remainder = _process(exp)
        processed.append(HornClause(head, body))
        queue = remainder + queue

    return Union(tuple(reversed(processed)))


def _new_head_for(exp):
    fv = free_variables(exp)
    fv = sorted(fv, key=lambda s: s.name)
    S = Symbol.fresh()
    return S(*tuple(fv))


# Rewrite this as a walker
def _process(exp):
    if isinstance(exp, Conjunction):
        bodies, remainders = zip(*map(_process, exp.formulas))
        return (
            reduce(add, bodies),
            reduce(add, remainders),
        )

    if isinstance(exp, Disjunction):
        nh = _new_head_for(exp)
        return (nh,), [(nh, f) for f in exp.formulas]

    if isinstance(exp, ExistentialPredicate):
        return _process(exp.body)

    if isinstance(exp, Negation):
        if isinstance(exp.formula, FunctionApplication):
            return (exp,), []
        if isinstance(exp.formula, ExistentialPredicate):
            nh = _new_head_for(exp.formula)
            return (Negation(nh),), [(nh, exp.formula)]

    if isinstance(exp, FunctionApplication):
        return (exp,), []

    raise NeuroLangException(
        "Expression not in safe range normal form: {}".format(exp)
    )
