from ..exceptions import NotInFONegE
from ..expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    IdentityWalker,
    PatternWalker,
    ReplaceSymbolWalker,
    add_match
)
from ..logic.expression_processing import (
    ExtractFreeVariablesWalker,
    extract_logic_free_variables
)
from . import (
    Conjunction,
    Constant,
    Disjunction,
    ExistentialPredicate,
    FunctionApplication,
    Implication,
    NaryLogicOperator,
    Negation,
    Quantifier,
    Symbol,
    UnaryLogicOperator,
    Union,
    UniversalPredicate
)


class LogicExpressionWalker(PatternWalker):
    @add_match(Quantifier)
    def walk_quantifier(self, expression):
        new_head = self.walk(expression.head)
        new_body = self.walk(expression.body)
        if new_head != expression.head or new_body != expression.body:
            return self.walk(expression.apply(new_head, new_body))
        else:
            return expression

    @add_match(Constant)
    def walk_constant(self, expression):
        return expression

    @add_match(Symbol)
    def walk_symbol(self, expression):
        return expression

    @add_match(UnaryLogicOperator)
    def walk_negation(self, expression):
        formula = self.walk(expression.formula)
        if formula != expression.formula:
            return self.walk(expression.apply(self.walk(expression.formula)))
        else:
            return expression

    @add_match(FunctionApplication)
    def walk_function(self, expression):
        new_args = tuple(map(self.walk, expression.args))
        if new_args != expression.args:
            return self.walk(expression.apply(
                    expression.functor, new_args
                ))
        else:
            return expression

    @add_match(NaryLogicOperator)
    def walk_nary(self, expression):
        new_formulas = tuple(self.walk(expression.formulas))
        if any(nf != f for nf, f in zip(new_formulas, expression.formulas)):
            return self.walk(expression.apply(new_formulas))
        else:
            return expression


class EliminateImplications(LogicExpressionWalker):
    """
    Removes the implication ocurrences of an expression.
    """

    @add_match(Implication(..., ...))
    def remove_implication(self, implication):
        c = self.walk(implication.consequent)
        a = self.walk(implication.antecedent)
        return self.walk(Disjunction((c, Negation(a))))


class MoveNegationsToAtomsSimpleOperationsMixin(PatternWalker):
    """
    Moves the negations the furthest possible to the atoms. On
    conjunction, disjunction, and negation operations.
    """
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


class MoveNegationsToAtoms(
    MoveNegationsToAtomsSimpleOperationsMixin, LogicExpressionWalker
):
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


class FONegELogicExpression(LogicExpressionWalker):
    """
    Walk a logic expression verifying that it is only in
    FO Neg Existential.
    """
    @add_match(UniversalPredicate)
    def abort_universal_predicate(self, expression):
        raise NotInFONegE(
            f"Forumla {expression} not in FO Existental Negation"
        )

    @add_match(Implication)
    def abort_implication(self, expression):
        raise NotInFONegE(
            f"Forumla {expression} not in FO Existental Negation"
        )


class MoveNegationsToAtomsInFONegE(
    MoveNegationsToAtomsSimpleOperationsMixin, FONegELogicExpression
):
    """
    Moves the negations the furthest possible to the atoms.
    Assumes that there are no implications in the expression.
    Expressions assumed to be in FO with existential and negation
    only.
    """

    @add_match(Negation(UniversalPredicate(..., ...)))
    def negated_universal(self, negation):
        quantifier = negation.formula
        x = quantifier.head
        p = self.walk(Negation(quantifier.body))
        return self.walk(ExistentialPredicate(x, p))


class FactorQuantifiersMixin(PatternWalker):
    """
    Factor quantifiers up on conjunctions and disjunctions.
    """
    @add_match(
        Disjunction,
        lambda exp: any(isinstance(f, Quantifier) for f in exp.formulas),
    )
    def disjunction_with_quantifiers(self, expression):
        expression = Disjunction(tuple(
            self.walk(formula)
            for formula in expression.formulas
        ))
        formulas = []
        quantifiers = []
        variables = set()
        for f in expression.formulas:
            if isinstance(f, Quantifier):
                if f.head in variables:
                    f = ReplaceSymbolWalker(
                        {f.head: Symbol[f.type].fresh()}
                    ).walk(f)
                variables.add(f.head)
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
        expression = Conjunction(tuple(
            self.walk(formula)
            for formula in expression.formulas
        ))
        quantifiers = []
        formulas = []
        variables = set()
        for f in expression.formulas:
            if isinstance(f, Quantifier):
                if f.head in variables:
                    f = ReplaceSymbolWalker(
                        {f.head: Symbol[f.type].fresh()}
                    ).walk(f)
                variables.add(f.head)
                quantifiers.append(f)
                formulas.append(f.body)
            else:
                formulas.append(f)
        exp = Conjunction(tuple(formulas))
        for q in reversed(quantifiers):
            exp = q.apply(q.head, exp)
        return self.walk(exp)


class MoveQuantifiersUp(FactorQuantifiersMixin, LogicExpressionWalker):
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


class MoveQuantifiersUpFONegE(FactorQuantifiersMixin, FONegELogicExpression):
    """
    Moves the quantifiers up in order to format the expression
    in prenex normal form. Assumes the expression contains no implications,
    that the variables of the quantifiers are not repeated, and that
    the query is in FO with Negation Existential as only quantifier.
    """
    pass


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

    def _desambiguate(self, formulas):
        used_variables = set()
        for uv in extract_logic_free_variables(formulas):
            used_variables |= uv
        for f in formulas:
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


class DistributeConjunctions(LogicExpressionWalker):
    @add_match(Conjunction, lambda e: len(e.formulas) > 2)
    def split(self, expression):
        head, *rest = expression.formulas
        rest = self.walk(Conjunction(tuple(rest)))
        new_exp = Conjunction((head, rest))
        return self.walk(new_exp)

    @add_match(Conjunction((..., Disjunction)))
    def rotate(self, expression):
        q, c = expression.formulas
        return self.walk(
            Disjunction(tuple(map(lambda p: Conjunction((q, p)), c.formulas)))
        )

    @add_match(Conjunction((Disjunction, ...)))
    def distribute(self, expression):
        c, q = expression.formulas
        return self.walk(
            Disjunction(tuple(map(lambda p: Conjunction((p, q)), c.formulas)))
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


class MakeExistentialsImplicit(LogicExpressionWalker):
    @add_match(ExistentialPredicate)
    def existential(self, expression):
        return self.walk(expression.body)


class RemoveExistentialOnVariables(LogicExpressionWalker):
    def __init__(self, variables_to_eliminate):
        self._variables_to_eliminate = variables_to_eliminate

    @add_match(ExistentialPredicate)
    def existential(self, expression):
        body = self.walk(expression.body)
        if expression.head in self._variables_to_eliminate:
            return body
        else:
            return ExistentialPredicate(expression.head, body)


def convert_to_pnf_with_cnf_matrix(expression):
    walker = ChainedWalker(
        EliminateImplications,
        MoveNegationsToAtoms,
        DesambiguateQuantifiedVariables,
        MoveQuantifiersUp,
        RemoveTrivialOperations,
        DistributeDisjunctions,
        CollapseDisjunctions,
        CollapseConjunctions,
    )
    return walker.walk(expression)


def convert_to_pnf_with_dnf_matrix(expression):
    walker = ChainedWalker(
        EliminateImplications,
        MoveNegationsToAtoms,
        DesambiguateQuantifiedVariables,
        MoveQuantifiersUp,
        RemoveTrivialOperations,
        DistributeConjunctions,
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


class RemoveTrivialOperations(LogicExpressionWalker):
    @add_match(Implication)
    def implication(self, expression):
        return Implication(
            expression.consequent,
            self.walk(expression.antecedent)
        )

    @add_match(NaryLogicOperator, lambda e: len(e.formulas) == 1)
    def remove_single(self, expression):
        return self.walk(expression.formulas[0])

    @add_match(Negation(Negation(...)))
    def remove_double_negation(self, expression):
        return self.walk(expression.formula.formula)


class PushExistentialsDown(
    CollapseConjunctionsMixin, CollapseDisjunctionsMixin,
    LogicExpressionWalker
):
    @add_match(
        ExistentialPredicate(..., NaryLogicOperator),
        lambda e: len(e.body.formulas) == 1
    )
    def push_eliminate_trivial_operation(self, expression):
        return self.walk(
            ExistentialPredicate(expression.head, expression.body.formulas[0])
        )

    @add_match(ExistentialPredicate(..., Disjunction))
    def push_existential_down_disjunction(self, expression):
        variable = expression.head
        changed = False
        new_formulas = tuple()
        for formula in expression.body.formulas:
            if variable in extract_logic_free_variables(formula):
                changed = True
                formula = self.walk(ExistentialPredicate(variable, formula))
            new_formulas += (formula,)
        if changed:
            res = self.walk(Disjunction(new_formulas))
        else:
            res = expression
        return res

    @add_match(
        ExistentialPredicate(..., Conjunction),
        lambda expression: any(
            isinstance(formula, Negation) and
            expression.head in extract_logic_free_variables(formula)
            for formula in expression.body.formulas
        )
    )
    def dont_push_when_it_can_be_unsafe(self, expression):
        variable = expression.head
        in_ = tuple()
        out_ = tuple()
        negative_logic_free_variables = set()
        for formula in expression.body.formulas:
            if isinstance(formula, Negation):
                negative_logic_free_variables |= extract_logic_free_variables(
                    formula
                )

        for formula in expression.body.formulas:
            if (
                negative_logic_free_variables &
                extract_logic_free_variables(formula)
            ):
                in_ += (formula,)
            else:
                out_ += (formula,)

        if len(out_) == 0:
            res = ExistentialPredicate(variable, self.walk(Conjunction(in_)))
        else:
            res = self.walk(
                Conjunction((
                    ExistentialPredicate(variable, Conjunction(in_)),
                    Conjunction(out_)
                ))
            )
        return res

    @add_match(ExistentialPredicate(..., Conjunction))
    def push_existential_down(self, expression):
        variable = expression.head
        in_ = tuple()
        out_ = tuple()
        for formula in expression.body.formulas:
            if variable in extract_logic_free_variables(formula):
                in_ += (formula,)
            else:
                out_ += (formula,)

        if len(out_) == 0:
            res = ExistentialPredicate(variable, self.walk(Conjunction(in_)))
        elif len(in_) == 0:
            res = self.walk(Conjunction(out_))
        if len(in_) > 0 and len(out_) > 0:
            res = self.walk(
                Conjunction(
                    (
                        ExistentialPredicate(variable, Conjunction(in_)),
                    ) + out_
                )
            )
        return res

    @add_match(ExistentialPredicate(..., ExistentialPredicate))
    def nested_existential(self, expression):
        outer_var = expression.head
        new_body = self.walk(expression.body)
        if new_body != expression.body:
            expression = self.walk(
                ExistentialPredicate(outer_var, new_body)
            )
        else:
            inner_var = expression.body.head
            inner_body = expression.body.body
            swapped_body = ExistentialPredicate(
                outer_var, inner_body
            )
            new_body = self.walk(swapped_body)
            if new_body != swapped_body:
                expression = self.walk(
                    ExistentialPredicate(inner_var, new_body)
                )
        return expression


class GuaranteeConjunction(IdentityWalker):
    @add_match(..., lambda e: not isinstance(e, Conjunction))
    def guarantee_conjunction(self, expression):
        return Conjunction((expression,))


class GuaranteeDisjunction(IdentityWalker):
    @add_match(..., lambda e: not isinstance(e, Disjunction))
    def guarantee_conjunction(self, expression):
        return Disjunction((expression,))


def nary_op_has_duplicated_formulas(nary_op: NaryLogicOperator) -> bool:
    seen = set()
    for formula in nary_op.formulas:
        if formula in seen:
            return True
        seen.add(formula)
    return False


class RemoveDuplicatedConjunctsDisjuncts(LogicExpressionWalker):
    @add_match(Disjunction, nary_op_has_duplicated_formulas)
    def disjunction(self, disjunction: Disjunction) -> Disjunction:
        return self.walk(self._nary_op_without_duplicates(disjunction))

    @add_match(Conjunction, nary_op_has_duplicated_formulas)
    def conjunction(self, conjunction: Conjunction) -> Conjunction:
        return self.walk(self._nary_op_without_duplicates(conjunction))

    @staticmethod
    def _nary_op_without_duplicates(
        nary_op: NaryLogicOperator,
    ) -> NaryLogicOperator:
        return nary_op.apply(tuple(set(nary_op.formulas)))


class IdentifyPureConjunctions(LogicExpressionWalker):
    @add_match(Conjunction, lambda e: CheckPureConjunction().walk(e))
    def pure_conjunction(self, conjunction):
        return [conjunction]

    @add_match(Conjunction)
    def conjunction(self, conjunction):
        conjunctions = []
        for f in conjunction.formulas:
            inner_f = self.walk(f)
            for inner_term in inner_f:
                if inner_term:
                    conjunctions.append(inner_term)

        return conjunctions

    @add_match(Disjunction)
    def disjunction(self, disjunction):
        res = []
        for f in disjunction.formulas:
            walked_f = self.walk(f)
            for inner_f in walked_f:
                if inner_f:
                    res.append(inner_f)

        return res

    @add_match(ExistentialPredicate)
    def existential_predicate(self, fa):
        body = self.walk(fa.body)
        if body:
            return [fa]

        return []

    @add_match(FunctionApplication)
    def f_app(self, fa):
        return []

    @add_match(Negation)
    def neg(self, expression):
        return self.walk(expression.formula)


class CheckPureConjunction(LogicExpressionWalker):
    @add_match(Conjunction)
    def conjunction(self, conjunction):
        for f in conjunction.formulas:
            if isinstance(f, Conjunction) or not self.walk(f):
                return False

        return True

    @add_match(FunctionApplication)
    def f_app(self, fa):
        return True

    @add_match(ExistentialPredicate)
    def existential_predicate(self, exist_predicate):
        if isinstance(exist_predicate.body, Conjunction):
            return False

        return self.walk(exist_predicate.body)

    @add_match(...)
    def default(self, expression):
        return False


class RemoveExistentialPredicates(LogicExpressionWalker):
    @add_match(ExistentialPredicate)
    def existential_predicate(self, existential_predicate):
        return self.walk(existential_predicate.body)


