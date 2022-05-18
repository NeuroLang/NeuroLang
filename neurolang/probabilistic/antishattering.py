from collections import Counter
from itertools import product
from operator import eq

from neurolang.logic.transformations import (
    CollapseConjunctions,
    MoveQuantifiersUp,
    PushExistentialsDown,
    RemoveTrivialOperations,
)

from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionWalker,
    add_match,
    expression_iterator,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import (
    FALSE,
    TRUE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Quantifier,
)
from ..logic.expression_processing import extract_logic_atoms
from ..logic.unification import (
    apply_substitution,
    compose_substitutions,
    most_general_unifier,
)
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation,
)


def _check_selfjoins(conjunction):
    a = Counter(
        (
            f.functor
            for f in conjunction.formulas
            if isinstance(f, FunctionApplication)
        )
    )
    return any(c > 1 for _, c in a.items())


class SelfjoinChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(
        Conjunction,
        lambda expression: any(
            isinstance(e, ExistentialPredicate)
            for _, e in expression_iterator(expression)
        ),
    )
    def match_conj(self, conjunction):
        expression = MoveQuantifiersUp().walk(conjunction)
        expression = CollapseConjunctions().walk(expression)
        expression = self.walk(expression)
        if expression != FALSE:
            expression = PushExistentialsDown().walk(expression)

        return expression

    @add_match(Conjunction, _check_selfjoins)
    def match_conjunction(self, conjunction):
        replacements = {}
        for i, f1 in enumerate(conjunction.formulas):
            if not (
                isinstance(f1, FunctionApplication)
                and is_atom_a_probabilistic_choice_relation(
                    f1, self.symbol_table
                )
            ):
                continue

            for f2 in conjunction.formulas[i + 1 :]:
                if (
                    isinstance(f2, FunctionApplication)
                    and f1.functor != f2.functor
                ):
                    continue
                mgu = most_general_unifier(f1, f2)
                if mgu is not None:
                    replacements = compose_substitutions(replacements, mgu[0])

        new_formulas = set(
            apply_substitution(f, replacements) for f in conjunction.formulas
        )

        sfc = Counter(
            (
                f.functor
                for f in new_formulas
                if isinstance(f, FunctionApplication)
                and is_atom_a_probabilistic_choice_relation(
                    f, self.symbol_table
                )
            )
        )
        if any(c > 1 for _, c in sfc.items()):
            return FALSE

        equalities = set(Constant(eq)(a, b) for a, b in replacements.items())

        new_formulas = tuple(new_formulas) + tuple(equalities)

        return Conjunction(new_formulas)


class EqualityVarsDetection(ExpressionWalker):
    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        vars = []
        for formula in conjunction.formulas:
            eq_vars = self.walk(formula)
            for lst in eq_vars:
                if lst:
                    vars.append(lst)

        return vars

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        vars = []
        res = self.walk(existential.body)
        for lst in res:
            if lst:
                vars.append(lst)

        return vars

    @add_match(FunctionApplication(Constant(eq), (..., ...)))
    def match_equality(self, equality):
        return [list(equality.args)]

    @add_match(...)
    def no_match(self, _):
        return [tuple()]


class NestedExistentialChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table, eq_vars):
        self.symbol_table = symbol_table
        self.eq_vars = eq_vars

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        evd = EqualityVarsDetection()
        forms = tuple()
        for formula in conjunction.formulas:
            eq_vars = evd.walk(formula)
            for vars_pair in eq_vars:
                if vars_pair in self.eq_vars:
                    forms += ReplaceNestedExistential(vars_pair).walk(formula)
                else:
                    forms += formula

        return Conjunction(forms)

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        evd = EqualityVarsDetection()
        eq_vars = evd.walk(existential.body)
        for vars_pair in eq_vars:
            if vars_pair in self.eq_vars:
                existential = ReplaceNestedExistential(vars_pair).walk(
                    existential
                )

        return existential


class ReplaceNestedExistential(ExpressionWalker):
    def __init__(self, eq_vars):
        self.eq_vars = eq_vars

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            forms += (self.walk(formula),)

        return Conjunction(forms)

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        if existential.head in self.eq_vars and isinstance(
            existential.body, ExistentialPredicate
        ):
            if existential.body.head in self.eq_vars:
                inner_formula = self.walk(existential.body.body)
                existential = ExistentialPredicate(
                    self.eq_vars[1], inner_formula
                )

        return existential

    @add_match(FunctionApplication(Constant[any](eq), ...))
    def match_equality(self, fa):
        for arg in fa.args:
            if arg not in self.eq_vars:
                return fa

        return ()

    @add_match(FunctionApplication)
    def match_function_application(self, fa):
        new_args = tuple()
        for arg in fa.args:
            if arg == self.eq_vars[0]:
                new_args += (self.eq_vars[1],)
            else:
                new_args += arg

        return fa.functor(*new_args)

    @add_match(...)
    def match_expression(self, expression):
        return expression
