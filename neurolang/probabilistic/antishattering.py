from collections import Counter
from operator import eq
from typing import AbstractSet

from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet

from ..expression_walker import (
    ExpressionWalker,
    add_match,
    expression_iterator,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import FALSE, TRUE, Conjunction, ExistentialPredicate, Quantifier
from ..logic.expression_processing import LogicSolver
from ..logic.transformations import (
    CollapseConjunctions,
    MoveQuantifiersUp,
    PushExistentialsDown,
)
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


def _check_equality(existential):
    equalities = [
        e
        for _, e in expression_iterator(existential)
        if isinstance(e, FunctionApplication) and e.functor == Constant(eq)
    ]

    return len(equalities) > 0


def _only_equality(existential):
    return isinstance(
        existential.body, FunctionApplication
    ) and existential.body.functor == Constant(eq)


def _equility_non_existential(conjunction):
    equalities = [
        formula
        for formula in conjunction.formulas
        if isinstance(formula, FunctionApplication)
        and formula.functor == Constant(eq)
        and any(isinstance(arg, Constant) for arg in formula.args)
    ]

    return len(equalities) > 0


class NestedExistentialChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    # Overlap with the next one, unify
    @add_match(Conjunction, _equility_non_existential)
    def match_conjuntion_non_existential(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            if isinstance(formula, ExistentialPredicate):
                forms += (self.walk(formula),)
            elif isinstance(
                formula, FunctionApplication
            ) and formula.functor == Constant(eq):
                consts = [
                    (arg,) for arg in formula.args if isinstance(arg, Constant)
                ]
                symbols = [
                    arg for arg in formula.args if isinstance(arg, Symbol)
                ]
                if len(consts) == 1:
                    new_symbol = Symbol.fresh()
                    new_set = WrappedRelationalAlgebraSet(iterable=consts)
                    type_ = new_set.row_type

                    constant = Constant[AbstractSet[type_]](
                        new_set, auto_infer_type=False, verify_type=False
                    )
                    constant_symbol = new_symbol.cast(constant.type)
                    self.symbol_table[constant_symbol] = constant
                    forms += (new_symbol(symbols[0]),)
                else:
                    forms += (formula,)
            else:
                forms += (formula,)

        expression = LogicQuantifiersSolver().walk(Conjunction(forms))
        return expression

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            if isinstance(formula, ExistentialPredicate):
                forms += (self.walk(formula),)
            else:
                forms += (formula,)

        expression = LogicQuantifiersSolver().walk(Conjunction(forms))
        return expression

    @add_match(ExistentialPredicate, _only_equality)
    def match_existential_equality(self, existential):
        return TRUE

    @add_match(ExistentialPredicate, _check_equality)
    def match_existential(self, existential):
        expression = MoveQuantifiersUp().walk(existential)
        expression = CollapseConjunctions().walk(expression)
        ext_vars = set()
        while hasattr(expression, "head"):
            ext_vars.add(expression.head)
            expression = expression.body

        pchoice_args = set()
        no_pchoice_args = set()
        for _, e in expression_iterator(expression.formulas):
            if isinstance(
                e, FunctionApplication
            ) and is_atom_a_probabilistic_choice_relation(
                e, self.symbol_table
            ):
                for arg in e.args:
                    pchoice_args.add(arg)

        for _, e in expression_iterator(expression.formulas):
            if (
                isinstance(e, FunctionApplication)
                and e.functor != Constant(eq)
                and not is_atom_a_probabilistic_choice_relation(
                    e, self.symbol_table
                )
            ):
                for arg in e.args:
                    no_pchoice_args.add(arg)

        only_pchoice_args = pchoice_args - no_pchoice_args
        if len(only_pchoice_args) == 0:
            only_ext_pchoice_args = ext_vars - no_pchoice_args
        else:
            only_ext_pchoice_args = only_pchoice_args.intersection(ext_vars)
        forms = tuple()
        remove_vars = set()
        for formula in expression.formulas:
            if isinstance(
                formula, FunctionApplication
            ) and formula.functor == Constant(eq):
                symbols, _ = self.get_symbols_constants_in_set(
                    formula.args, only_ext_pchoice_args
                )
                forms += (TRUE,)
                if len(symbols) >= 1:
                    remove_vars.add(formula.args[0])
            else:
                forms += (formula,)

        expression = Conjunction(forms)
        new_ext_vars = ext_vars - remove_vars
        for ext_var in new_ext_vars:
            expression = ExistentialPredicate(ext_var, expression)

        expression = PushExistentialsDown().walk(expression)
        expression = LogicQuantifiersSolver().walk(expression)
        return expression

    def get_symbols_constants_in_set(self, args, vars_set):
        consts = set()
        symbols = set()
        for arg in args:
            if isinstance(arg, Constant) and arg in vars_set:
                consts.add(arg)
            elif isinstance(arg, Symbol) and arg in vars_set:
                symbols.add(arg)

        return symbols, consts


class LogicQuantifiersSolver(LogicSolver):
    @add_match(Quantifier)
    def match_quantifier(self, quantifier):
        body = self.walk(quantifier.body)
        return ExistentialPredicate(quantifier.head, body)

    @add_match(FunctionApplication)
    def match_func_app(self, app):
        return app

    @add_match(Constant)
    def match_constant(self, constant):
        return constant
