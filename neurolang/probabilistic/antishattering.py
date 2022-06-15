from ast import walk
from collections import Counter
from operator import eq

from ..datalog.wrapped_collections import NAMED_DEE
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
from ..relational_algebra.relational_algebra import (
    ExtendedProjection,
    FunctionApplicationListMember,
    NumberColumns,
    int2columnint_constant,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import ONE, ProvenanceAlgebraSet
from .probabilistic_ra_utils import is_atom_a_probabilistic_choice_relation

PED = PushExistentialsDown()
MQU = MoveQuantifiersUp()
CC = CollapseConjunctions()


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
        expression = MQU.walk(conjunction)
        expression = CC.walk(expression)

        ext_vars = set()
        while hasattr(expression, "head"):
            ext_vars.add(expression.head)
            expression = expression.body

        walked_expression = self.walk(expression)
        if walked_expression == FALSE:
            return walked_expression

        while ext_vars:
            var = ext_vars.pop()
            walked_expression = ExistentialPredicate(var, walked_expression)

        walked_expression = PED.walk(walked_expression)

        return walked_expression


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

        if len(replacements) > 0:
            equalities = set(
                Constant(eq)(a, b) for a, b in replacements.items()
            )
            new_formulas = tuple(new_formulas) + tuple(equalities)

            conjunction = Conjunction(new_formulas)

        return conjunction


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


class NestedExistentialChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        modified = False
        for formula in conjunction.formulas:
            if isinstance(formula, ExistentialPredicate):
                walked_formula = self.walk(formula)
                forms += (walked_formula,)
                if formula is not walked_formula:
                    modified = True
            elif isinstance(
                formula, FunctionApplication
            ) and formula.functor == Constant(eq):
                consts = [
                    arg for arg in formula.args if isinstance(arg, Constant)
                ]
                symbols_str = [
                    str2columnstr_constant(arg.name)
                    for arg in formula.args
                    if isinstance(arg, Symbol)
                ]
                symbols = [
                    arg for arg in formula.args if isinstance(arg, Symbol)
                ]
                if len(consts) == 1:
                    new_symbol = self.create_named_dee(consts, symbols_str)
                    forms += (new_symbol(symbols[0]),)
                    modified = True
                else:
                    forms += (formula,)
            else:
                form = self.walk(formula)
                forms += (form,)

        if modified:
            conjunction = LogicQuantifiersSolver().walk(Conjunction(forms))

        return conjunction

    @add_match(ExistentialPredicate, _only_equality)
    def match_existential_equality(self, existential):
        return TRUE

    @add_match(ExistentialPredicate, _check_equality)
    def match_existential(self, existential):
        expression = MQU.walk(existential)
        expression = CC.walk(expression)
        ext_vars = set()
        while hasattr(expression, "head"):
            ext_vars.add(expression.head)
            expression = expression.body

        pchoice_args, no_pchoice_args = self.pchoice_args_separation(
            expression
        )

        only_pchoice_args = pchoice_args - no_pchoice_args
        if len(only_pchoice_args) == 0:
            only_ext_pchoice_args = ext_vars - no_pchoice_args
        else:
            only_ext_pchoice_args = only_pchoice_args.intersection(ext_vars)

        expression, new_ext_vars = self.replace_equalities_with_constants(
            expression, ext_vars, only_ext_pchoice_args
        )
        for ext_var in new_ext_vars:
            expression = ExistentialPredicate(ext_var, expression)

        expression = PED.walk(expression)
        expression = LogicQuantifiersSolver().walk(expression)
        return expression

    def create_named_dee(self, consts, symbols_str):
        new_symbol = Symbol.fresh()
        provenance_column = str2columnstr_constant(Symbol.fresh().name)

        constant = NumberColumns(
            ExtendedProjection(
                NAMED_DEE,
                tuple(
                    FunctionApplicationListMember(c, s)
                    for c, s in zip(consts, symbols_str)
                )
                + (FunctionApplicationListMember(ONE, provenance_column),),
            ),
            (provenance_column,) + tuple(symbols_str),
        )

        constant = ProvenanceAlgebraSet(constant, int2columnint_constant(0))
        constant_symbol = new_symbol.cast(constant.type)
        self.symbol_table[constant_symbol] = constant
        return new_symbol

    def replace_equalities_with_constants(
        self, expression, ext_vars, only_ext_pchoice_args
    ):
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
        return expression, new_ext_vars

    def pchoice_args_separation(self, expression):
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
        return pchoice_args, no_pchoice_args

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
        return quantifier.apply(quantifier.head, body)

    @add_match(FunctionApplication)
    def match_func_app(self, app):
        return app

    @add_match(Constant)
    def match_constant(self, constant):
        return constant
