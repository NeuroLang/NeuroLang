from . import (
    Symbol,
    Constant,
    FunctionApplication,
    Implication,
    Union,
    Conjunction,
    Disjunction,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
    Quantifier,
)
from ..logic.expression_processing import ExtractFreeVariablesWalker
from ..expression_walker import (
    ExpressionWalker,
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
        return expression.apply(self.walk(expression.formula))

    @add_match(FunctionApplication)
    def walk_function(self, expression):
        return expression.apply(
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
        return self.walk(Disjunction((c, Negation(a))))


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
        return self.walk(ExistentialPredicate(x, p))

    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return self.walk(UniversalPredicate(x, p))

    @add_match(Negation(Conjunction(...)))
    def negated_conjunction(self, negation):
        conj = negation.formula
        formulas = map(lambda e: self.walk(Negation(e)), conj.formulas)
        return self.walk(Disjunction(tuple(formulas)))

    @add_match(Negation(Disjunction(...)))
    def negated_disjunction(self, negation):
        disj = negation.formula
        formulas = map(lambda e: self.walk(Negation(e)), disj.formulas)
        return self.walk(Conjunction(tuple(formulas)))

    @add_match(Negation(Negation(...)))
    def negated_negation(self, negation):
        return self.walk(negation.formula.formula)


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

    @add_match(
        Disjunction,
        lambda exp: any(isinstance(f, Quantifier) for f in exp.formulas),
    )
    def disjunction_with_quantifiers(self, expression):
        expression = self.walk_disjunction(expression)
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
        lambda exp: any(isinstance(f, Quantifier) for f in exp.formulas),
    )
    def conjunction_with_quantifiers(self, expression):
        expression = self.walk_conjunction(expression)
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
    def implication(self, expression):
        fs = self.walk(expression.consequent), self.walk(expression.antecedent)
        self._desambiguate(fs)
        return expression.apply(*fs)

    @add_match(Union)
    def union(self, expression):
        fs = self.walk(expression.formulas)
        self._desambiguate(fs)
        return expression.apply(fs)

    @add_match(Disjunction)
    def disjunction(self, expression):
        fs = self.walk(expression.formulas)
        self._desambiguate(fs)
        return expression.apply(fs)

    @add_match(Conjunction)
    def conjunction(self, expression):
        fs = self.walk(expression.formulas)
        self._desambiguate(fs)
        return expression.apply(fs)

    @add_match(Negation)
    def negation(self, expression):
        return expression.apply(self.walk(expression.formula))

    @add_match(Quantifier)
    def quantifier(self, expression):
        expression.body = self.walk(expression.body)
        uq = UsedQuantifiers().walk(expression.body)
        for q in uq:
            if q.head == expression.head:
                self.rename_quantifier(q)
        return expression

    def rename_quantifier(self, q):
        ns = Symbol.fresh()
        q.body = ReplaceFreeSymbolWalker({q.head: ns}).walk(q.body)
        q.head = ns

    def _desambiguate(self, l):
        used_variables = set()
        for f in l:
            uq = UsedQuantifiers().walk(f)
            for q in self._conflicted_quantifiers(used_variables, uq):
                self.rename_quantifier(q)
            used_variables |= self._bound_variables(uq)

    def _conflicted_quantifiers(self, used_variables, quantifiers):
        bv = self._bound_variables(quantifiers)
        repeated = bv & used_variables
        return [q for q in quantifiers if q.head in repeated]

    def _bound_variables(self, quantifiers):
        return set(map(lambda q: q.head, quantifiers))


class ReplaceFreeSymbolWalker(ReplaceSymbolWalker):
    @add_match(Quantifier)
    def stop_if_bound(self, expression):
        s = expression.head
        r = self.symbol_replacements.pop(s, None)
        expression.body = self.walk(expression.body)
        if r:
            self.symbol_replacements[s] = r
        return expression


class UsedQuantifiers(PatternWalker):
    @add_match(FunctionApplication)
    def function(self, exp):
        return set()

    @add_match(Symbol)
    def symbol(self, exp):
        return set()

    @add_match(Negation)
    def negation(self, exp):
        return self.walk(exp.formula)

    @add_match(Disjunction)
    def disjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(Implication)
    def implication(self, exp):
        return set.union(self.walk(exp.antecedent), self.walk(exp.consequent))

    @add_match(Conjunction)
    def conjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(Quantifier)
    def quantifier(self, exp):
        return {exp} | self.walk(exp.body)


class DistributeDisjunctions(LogicExpressionWalker):
    @add_match(Disjunction, lambda e: len(e.formulas) > 2)
    def split(self, expression):
        head, *rest = expression.formulas
        rest = self.walk(Disjunction(tuple(rest)))
        new_exp = Disjunction((head, rest))
        return self.walk(new_exp)

    @add_match(Disjunction((..., Conjunction)))
    def rotate(self, expression):
        q, c = expression.formulas
        return self.walk(
            Conjunction(tuple(map(lambda p: Disjunction((q, p)), c.formulas)))
        )

    @add_match(Disjunction((Conjunction, ...)))
    def distribute(self, expression):
        c, q = expression.formulas
        return self.walk(
            Conjunction(tuple(map(lambda p: Disjunction((p, q)), c.formulas)))
        )


class CollapseDisjunctionsMixin(PatternWalker):
    @add_match(
        Disjunction,
        lambda e: any(isinstance(f, Disjunction) for f in e.formulas),
    )
    def disjunction(self, e):
        new_arg = []
        for f in map(self.walk, e.formulas):
            if isinstance(f, Disjunction):
                new_arg.extend(f.formulas)
            else:
                new_arg.append(f)
        return self.walk(Disjunction(tuple(new_arg)))


class CollapseDisjunctions(CollapseDisjunctionsMixin, ExpressionWalker):
    pass


class CollapseConjunctionsMixin(PatternWalker):
    @add_match(
        Conjunction,
        lambda e: any(isinstance(f, Conjunction) for f in e.formulas),
    )
    def conjunction(self, e):
        new_arg = []
        for f in map(self.walk, e.formulas):
            if isinstance(f, Conjunction):
                new_arg.extend(f.formulas)
            else:
                new_arg.append(f)
        return self.walk(Conjunction(tuple(new_arg)))


class CollapseConjunctions(CollapseConjunctionsMixin, ExpressionWalker):
    pass


class RemoveUniversalPredicates(LogicExpressionWalker):
    """
    Changes the universal predicates by equivalent expressions
    using existential quantifiers.
    """

    @add_match(UniversalPredicate)
    def universal(self, expression):
        return self.walk(
            Negation(
                ExistentialPredicate(
                    expression.head, Negation(self.walk(expression.body))
                )
            )
        )


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


class ExtractFOLFreeVariables(ExtractFreeVariablesWalker):
    @add_match(Implication)
    def extract_variables_s(self, exp):
        return self.walk(exp.consequent) | self.walk(exp.antecedent)


class DistributeUniversalQuantifiers(PatternWalker):
    @add_match(UniversalPredicate(..., Conjunction))
    def distribute_universal_quantifier(self, uq):
        return self.walk(
            Conjunction(
                tuple(map(self._apply_quantifier(uq.head), uq.body.formulas))
            )
        )

    def _apply_quantifier(self, var):
        def foo(exp):
            fv = ExtractFOLFreeVariables().walk(exp)
            if var in fv:
                exp = UniversalPredicate(var, exp)
            return exp

        return foo


class DistributeImplicationsWithConjunctiveHeads(PatternWalker):
    @add_match(Implication(Conjunction, ...))
    def distribute_implication_with_conjunctive_head(self, impl):
        return self.walk(
            Conjunction(
                tuple(
                    Implication(h, impl.antecedent)
                    for h in impl.consequent.formulas
                )
            )
        )
