import collections
import itertools
import operator

from ..expression_pattern_matching import add_match
from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionWalker,
    expression_iterator
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, ExistentialPredicate
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    LogicExpressionWalker,
    RemoveTrivialOperations,
    convert_to_pnf_with_dnf_matrix
)
from ..relational_algebra import (
    ColumnStr,
    Projection,
    Selection,
    int2columnint_constant,
    str2columnstr_constant
)
from .probabilistic_ra_utils import ProbabilisticFactSet

EQ = Constant(operator.eq)
NE = Constant(operator.ne)
RTO = RemoveTrivialOperations()


def atom_to_constant_to_RA_conditions(
    atom: FunctionApplication, head_vars: set
) -> set:
    conditions = set()
    for i, arg in enumerate(atom.args):
        if (
            isinstance(arg, Constant) or
            (isinstance(arg, Symbol) and arg in head_vars)
        ):
            if isinstance(arg, Symbol):
                arg = str2columnstr_constant(arg.name)
            conditions.add(EQ(int2columnint_constant(i), arg))
    return conditions

    return conditions

class NormalizeNotEquals(LogicExpressionWalker):
    def __init__(self, head_vars):
        self.head_vars = head_vars

    @add_match(FunctionApplication(NE, (Constant, Symbol)))
    def flip_ne_arguments(self, expression):
        return expression.apply(
            NE,
            expression.args[::-1]
        )

    @add_match(FunctionApplication(NE, (Symbol, Symbol)))
    def flip_ne_symbol_arguments(self, expression):
        if (
            expression.args[0] in self.head_vars and
            expression.args[1] not in self.head_vars
        ):
            expression = expression.apply(
                NE,
                expression.args[::-1]
            )

        return expression


def conditions_per_symbol(ucq_query):
    ucq_query_dnf = convert_to_pnf_with_dnf_matrix(ucq_query)
    head_vars = extract_logic_free_variables(ucq_query_dnf)
    ucq_query_dnf = NormalizeNotEquals(head_vars).walk(ucq_query_dnf)

    conditions_per_symbol = collections.defaultdict(set)
    number_of_args = dict()
    for atom in extract_logic_atoms(ucq_query_dnf):
        if not isinstance(atom.functor, Symbol):
            continue
        condition = atom_to_constant_to_RA_conditions(atom, head_vars)
        conditions_per_symbol[atom.functor] |= condition
        number_of_args[atom.functor] = len(atom.args)

    conjunctions = (
        expression for _, expression in expression_iterator(ucq_query_dnf)
        if isinstance(expression, Conjunction) and
        any(
            _formula_is_ne(formula)
            for formula in expression.formulas
        )
    )

    for conjunction in conjunctions:
        nes = (
            formula for formula in conjunction.formulas
            if _formula_is_ne(formula)
        )
        for ne in nes:
            ne_symbol = ne.args[0]
            ne_constant = ne.args[1]
            if isinstance(ne_constant, Symbol):
                ne_constant = str2columnstr_constant(ne_constant.name)
            for atom in extract_logic_atoms(conjunction):
                if not isinstance(atom.functor, Symbol):
                    continue
                ne_conditions = {
                    NE(int2columnint_constant(i), ne_constant)
                    for i, s in enumerate(atom.args)
                    if s == ne_symbol
                }
                if ne_conditions:
                    conditions_per_symbol[atom.functor] |= ne_conditions
                    number_of_args[atom.functor] = len(atom.args)

    return conditions_per_symbol, number_of_args


def _formula_is_ne(formula):
    return (
        isinstance(formula, FunctionApplication) and
        formula.functor == NE and
        isinstance(formula.args[0], Symbol)
    )


def sets_per_symbol(ucq_query):
    cps, number_of_args = conditions_per_symbol(ucq_query)

    complementary = {EQ: NE, NE: EQ}
    symbol_sets = dict()

    for symbol, conditions in cps.items():
        if len(conditions) == 0:
            continue
        column_conditions = collections.defaultdict(set)
        for condition in conditions:
            if condition.functor == NE:
                condition = condition.apply(EQ, condition.unapply()[1])
            column_conditions[condition.args[0]].add((condition,))

        for column, conditions in column_conditions.items():
            column_conditions[column].add(
                tuple(
                    complementary[c[0].functor](*c[0].args) for c in conditions

                )
            )

        val = lambda v: v.value if isinstance(v, Constant) else v.name

        set_formulas = list(
            Conjunction(tuple(
                sorted(
                    itertools.chain(*conditions),
                    key=lambda exp: (val(exp.args[0]), val(exp.args[1]))
                )
            ))
            for conditions in
            itertools.product(*column_conditions.values())
        )
        conditions_columns = [
            set(
                condition_item.args[0].value
                for condition_item in condition.formulas
                if condition_item.functor == EQ
            )
            for condition in set_formulas
        ]
        conditions_projected_args = [
            tuple(
                int2columnint_constant(i)
                for i in range(number_of_args[symbol])
                # if i not in condition_columns
            )
            for condition_columns in conditions_columns
        ]

        symbol_sets[symbol] = set(
            Projection(Selection(symbol, RTO.walk(condition)), projected_args)
            for condition, projected_args
            in zip(set_formulas, conditions_projected_args)
        )

    return symbol_sets


class HeadVar(ColumnStr):
    pass


def str2columnstr_headvar_constant(s):
    return Constant[HeadVar](s)


class ShatterEqualities(ExpressionWalker):
    def __init__(self, symbol_sets, symbol_table, head_vars):
        self.head_vars = head_vars
        self.symbol_sets = symbol_sets
        self.symbol_table = symbol_table
        self.shatter_symbols = dict()

    @add_match(FunctionApplication(Symbol, ...))
    def shatter_symbol(self, expression):
        expression = self._shatter_atom(expression)
        return expression

    def _shatter_atom(self, expression, nes=None):
        if expression.functor not in self.symbol_sets:
            return expression

        sets = self.symbol_sets[expression.functor]
        sets_to_keep = self._identify_sets_to_keep(expression, sets, nes=nes)

        expression, e_vars = self._compute_expression_arguments(
            expression, sets_to_keep
        )

        if len(expression) == 1:
            expression = expression[0]
        else:
            expression = Disjunction(expression)

        for head_var in (
            self.head_vars &
            extract_logic_free_variables(expression)
        ):
            new_var = Symbol.fresh()
            expression = ReplaceExpressionWalker(
                {head_var: new_var}
            ).walk(expression)
            e_vars.add(new_var)

        for e_var in e_vars:
            expression = ExistentialPredicate(e_var, expression)
        return expression

    @add_match(
        Conjunction,
        lambda expression: (
            any(
                _formula_is_ne(formula)
                for formula in expression.formulas
            )
        )
    )
    def shatter_inequalities(self, expression):
        nes = set()
        formulas_to_keep = []
        for formula in expression.formulas:
            if (
                _formula_is_ne(formula) and
                (
                    isinstance(formula.args[1], Constant) or
                    formula.args[1] in self.head_vars
                )
            ):
                if isinstance(formula.args[1], Symbol):
                    functor, args = formula.unapply()
                    args = (
                        args[0],
                        str2columnstr_headvar_constant(args[1].name)
                    )
                    formula = formula.apply(functor, args)
                nes.add(formula)
            else:
                formulas_to_keep.append(formula)

        resulting_formulas = tuple()
        for formula in formulas_to_keep:
            if formula.functor not in self.symbol_sets:
                resulting_formulas += (formula,)
            else:
                resulting_formulas += (self._shatter_atom(formula, nes=nes),)

        return Conjunction(resulting_formulas)

    def _compute_expression_arguments(self, expression, sets_to_keep):
        args = []
        for i, arg in enumerate(expression.args):
            if isinstance(arg, Symbol):
                args.append((i, arg, False))
            elif isinstance(arg, Constant):
                args.append((i, Symbol.fresh(), True))
        formulas = tuple()
        e_vars = set()
        for set_ in sets_to_keep:
            if set_ not in self.shatter_symbols:
                sym = Symbol.fresh()
                self.shatter_symbols[set_] = sym
                self.symbol_table[sym] = set_
            formula_args = []
            for i, arg, is_e_var in args:
                if int2columnint_constant(i) in set_.columns():
                    formula_args.append(arg)
                    if is_e_var:
                        e_vars.add(arg)
            formulas += (self.shatter_symbols[set_](*formula_args),)
        return formulas, e_vars

    def _identify_sets_to_keep(self, expression, sets, nes=None):
        conditions = atom_to_constant_to_RA_conditions(
            expression, self.head_vars
        )
        negative_conditions = set()
        if nes is not None:
            args = list(expression.args)
            for ne in nes:
                if ne.args[0] in expression.args:
                    negative_conditions.add(
                        EQ(
                            int2columnint_constant(args.index(ne.args[0])),
                            ne.args[1]
                        )
                    )
        sets_to_keep = []
        for set_ in sets:
            set_conditions = self._selection_formulas(set_)
            if (
                conditions <= set_conditions and
                set_conditions.isdisjoint(negative_conditions)
            ):
                sets_to_keep.append(set_)
        return sets_to_keep

    def _selection_formulas(self, set_):
        formula = set_.relation.formula
        if isinstance(formula, Conjunction):
            set_conditions = set(formula.formulas)
        else:
            set_conditions = {formula}
        return set_conditions


class RelationalAlgebraSelectionConjunction(ExpressionWalker):
    @add_match(Selection(..., Conjunction))
    def selection_conjunction(self, expression):
        and_ = Constant(operator.and_)
        formulas = expression.formula.formulas
        new_formula = formulas[0]
        for f in formulas[1:]:
            new_formula = and_(new_formula, f)
        return Selection(
            expression.relation,
            new_formula
        )


def shatter_constants(query, symbol_table, shatter_all=False):
    head_vars = extract_logic_free_variables(query)
    symbol_sets = sets_per_symbol(query)
    if not shatter_all:
        for k in list(symbol_sets.keys()):
            if not isinstance(symbol_table[k], ProbabilisticFactSet):
                del symbol_sets[k]
    old_keys = set(symbol_table.keys())
    se = ShatterEqualities(symbol_sets, symbol_table, head_vars)
    query = se.walk(RTO.walk(query))
    new_keys = set(symbol_table) - old_keys
    rasc = RelationalAlgebraSelectionConjunction()
    for k in new_keys:
        symbol_table[k] = rasc.walk(symbol_table[k])
    return query
