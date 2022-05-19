from collections import Counter
from operator import eq
from ..logic.expression_processing import LogicSolver

from ..logic.transformations import (
    CollapseConjunctions,
    MoveQuantifiersUp,
    PushExistentialsDown,
)

from ..expression_walker import (
    ExpressionWalker,
    add_match,
    expression_iterator,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import (
    FALSE,
    TRUE,
    Conjunction,
    ExistentialPredicate,
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


class NestedExistentialChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            if isinstance(formula, ExistentialPredicate):
                forms += (self.walk(formula),)
            else:
                forms += (formula,)

        return Conjunction(forms)

    @add_match(ExistentialPredicate, _check_equality)
    def match_existential(self, existential):
        expression = MoveQuantifiersUp().walk(existential)
        expression = CollapseConjunctions().walk(expression)
        ext_vars = set()
        while hasattr(expression, "head"):
            ext_vars.add(expression.head)
            expression = expression.body

        pchoice_args = set()
        for _, e in expression_iterator(expression.formulas):
            if isinstance(
                e, FunctionApplication
            ) and not is_atom_a_probabilistic_choice_relation(
                e, self.symbol_table
            ):
                for arg in e.args:
                    pchoice_args.add(arg)

        no_pchoice_args = set()
        for _, e in expression_iterator(expression.formulas):
            if isinstance(
                e, FunctionApplication
            ) and not is_atom_a_probabilistic_choice_relation(
                e, self.symbol_table
            ):
                for arg in e.args:
                    no_pchoice_args.add(arg)

        only_pchoice_args = pchoice_args - no_pchoice_args
        only_ext_pchoice_args = only_pchoice_args.intersection(ext_vars)
        forms = tuple()
        remove_vars = set()
        for formula in expression.formulas:
            if isinstance(
                formula, FunctionApplication
            ) and formula.functor == Constant[any](eq):
                symbols, consts = self.get_symbols_constants_in_set(
                    formula.args, only_ext_pchoice_args
                )
                forms += TRUE
                if len(symbols) == 1 and len(consts) == 1:
                    remove_vars.add(symbols.pop())
                elif len(symbols) == 2:
                    # set(Constant(eq)(a, b) for a, b in replacements.items())
                    remove_vars.add(formula.args[0])
            else:
                forms += formula

        expression = Conjunction(forms)
        new_ext_vars = ext_vars - remove_vars
        for ext_var in new_ext_vars:
            expression = ExistentialPredicate(ext_var, expression)

        expression = PushExistentialsDown().walk(expression)
        expression = LogicSolver().walk(expression)
        return expression

    def get_symbols_constants_in_set(self, args, set):
        consts = set()
        symbols = set()
        for arg in args:
            if isinstance(arg, Constant) and arg in set:
                consts.add(arg)
            elif isinstance(arg, Symbol) and arg in set:
                symbols.add(arg)

        return symbols, consts

