from operator import eq, invert
from typing import AbstractSet, Tuple, Callable

from ..exceptions import NeuroLangException
from ..expression_walker import (ExpressionBasicEvaluator,
                                 ReplaceExpressionsByValues, add_match)
from ..expressions import Constant, FunctionApplication, Symbol
from ..relational_algebra import (ColumnInt, ColumnStr, Difference,
                                  NameColumns, NaturalJoin, Projection,
                                  RenameColumn, Selection)
from ..utils import NamedRelationalAlgebraFrozenSet
from .expressions import Conjunction, Negation

EQ = Constant(eq)
EQ_pattern = Constant[Callable](eq)
REBV = ReplaceExpressionsByValues({})


class TranslateToNamedRA(ExpressionBasicEvaluator):
    """Partial implementation of algorithm 5.4.8 from [1]_.

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
       (Addison Wesley, 1995), Addison-Wesley.
    """
    @add_match(FunctionApplication(EQ_pattern, (Constant, Symbol)))
    def translate_eq_c_s(self, expression):
        return self.walk(EQ(*expression.args[::-1]))

    @add_match(FunctionApplication(EQ_pattern, (Symbol, Constant)))
    def translate_eq_s_c(self, expression):
        symbol, constant = expression.args
        return Constant[AbstractSet[Tuple[constant.type]]](
            NamedRelationalAlgebraFrozenSet(
                (symbol.name,),
                [(REBV.walk(constant),)]
            )
        )

    @add_match(FunctionApplication(EQ_pattern, ...))
    def translate_eq(self, expression):
        new_args = tuple()
        changed = False
        for arg in expression.args:
            new_arg = self.walk(arg)
            changed |= new_arg is not arg
            new_args += (new_arg,)

        if changed:
            return EQ(*new_args)
        else:
            return expression

    @add_match(FunctionApplication)
    def translate_fa(self, expression):
        functor = expression.functor
        named_args = tuple()
        projections = tuple()
        selections = dict()
        selection_columns = dict()
        for i, arg in enumerate(expression.args):
            if isinstance(arg, Constant):
                selections[i] = arg
            elif arg in named_args:
                selection_columns[i] = named_args.index(arg)
            else:
                projections += (Constant[ColumnInt](i, verify_type=False),)
                named_args += (arg,)

        in_set = self.generate_ra_expression(
            functor,
            selections, selection_columns,
            projections, named_args
        )

        return in_set

    def generate_ra_expression(
        self, functor, selections, selection_columns,
        projections, named_args
    ):
        in_set = functor
        for k, v in selections.items():
            criterium = EQ(
                Constant[ColumnInt](k, verify_type=False), v
            )
            in_set = Selection(in_set, criterium)
        for k, v in selection_columns.items():
            criterium = EQ(
                Constant[ColumnInt](k, verify_type=False),
                Constant[ColumnInt](v, verify_type=False)
            )
            in_set = Selection(in_set, criterium)

        in_set = Projection(in_set, projections)
        column_names = tuple(
            Constant[ColumnStr](ColumnStr(arg.name), verify_type=False)
            for arg in named_args
        )
        in_set = NameColumns(in_set, column_names)
        return in_set

    @add_match(Negation)
    def translate_negation(self, expression):
        if isinstance(expression.formula, Negation):
            return self.walk(expression.formula.formula)

        formula = self.walk(expression.formula)
        if (
            isinstance(formula, FunctionApplication) and
            isinstance(formula.functor, Constant)
        ):
            res = FunctionApplication(Constant(invert), (formula,))
        else:
            res = Negation(self.walk(expression.formula))
        return res

    @add_match(Conjunction)
    def translate_conj(self, expression):
        pos_formulas, neg_formulas, eq_formulas, named_columns = \
            self.classify_formulas_obtain_names(expression)

        if len(pos_formulas) > 0:
            output = pos_formulas[0]
            for pos_formula in pos_formulas[1:]:
                output = NaturalJoin(output, pos_formula)
        else:
            return NamedRelationalAlgebraFrozenSet([])

        output = TranslateToNamedRA.process_negative_formulas(
            neg_formulas, named_columns, output
        )

        output = TranslateToNamedRA.process_equality_formulas(
            eq_formulas, named_columns, output
        )

        return output

    def classify_formulas_obtain_names(self, expression):
        pos_formulas = []
        neg_formulas = []
        eq_formulas = []
        named_columns = set()
        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                neg_formulas.append(formula.formula)
            elif (
                isinstance(formula, FunctionApplication) and
                formula.functor == EQ
            ):
                if formula.args[0] != formula.args[1]:
                    eq_formulas.append(formula)
            else:
                pos_formulas.append(formula)
                if isinstance(formula, Constant):
                    named_columns.update(formula.value.columns)
                elif isinstance(formula, NameColumns):
                    named_columns.update(formula.column_names)
        return pos_formulas, neg_formulas, eq_formulas, named_columns

    @staticmethod
    def process_negative_formulas(neg_formulas, named_columns, output):
        for neg_formula in neg_formulas:
            neg_cols = TranslateToNamedRA.obtain_negative_columns(neg_formula)
            if named_columns > neg_cols:
                neg_formula = NaturalJoin(output, neg_formula)
            elif named_columns != neg_cols:
                raise NeuroLangException(
                    f'Negative predicate {neg_formula} is not safe range'
                )
            output = Difference(output, neg_formula)
        return output

    @staticmethod
    def obtain_negative_columns(neg_formula):
        if isinstance(neg_formula, NameColumns):
            neg_cols = set(neg_formula.column_names)
        elif isinstance(neg_formula, Constant):
            neg_cols = set(neg_formula.value.columns)
        else:
            raise NeuroLangException(
                f"Negative formula {neg_formula} is  not a named relation"
            )
        return neg_cols

    @staticmethod
    def process_equality_formulas(eq_formulas, named_columns, output):
        for formula in eq_formulas:
            left, right = formula.args
            left_col = Constant[ColumnStr](
                ColumnStr(left.name), verify_type=False
            )
            right_col = Constant[ColumnStr](
                ColumnStr(right.name), verify_type=False
            )
            criteria = EQ(left_col, right_col)
            if left in named_columns and right in named_columns:
                output = Selection(output, criteria)
            elif left in named_columns:
                output = Selection(NaturalJoin(
                        output, RenameColumn(output, left_col, right_col)
                    ),
                    criteria
                )
            elif right in named_columns:
                output = Selection(NaturalJoin(
                        output, RenameColumn(output, right_col, left_col)
                    ),
                    criteria
                )
            else:
                raise NeuroLangException(
                    f'At least one of the symbols {left} {right} must be '
                    'in the free variables of the antecedent'
                )
        return output
