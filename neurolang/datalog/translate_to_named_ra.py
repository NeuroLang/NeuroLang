from operator import eq, invert
from typing import AbstractSet, Callable, Tuple

from ..exceptions import NeuroLangException
from ..expression_walker import (ExpressionBasicEvaluator,
                                 ReplaceExpressionsByValues, add_match)
from ..expressions import Constant, FunctionApplication, Symbol, NonConstant
from ..relational_algebra import (ColumnInt, ColumnStr, Difference,
                                  ExtendedProjection,
                                  ExtendedProjectionListMember, NameColumns,
                                  NaturalJoin, Projection, RenameColumn,
                                  Selection)
from ..utils import NamedRelationalAlgebraFrozenSet
from .expressions import Conjunction, Negation

EQ = Constant(eq)
EQ_pattern = Constant[Callable](eq)
Builtin_pattern = Constant[Callable]
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

    @add_match(FunctionApplication(EQ_pattern, (FunctionApplication, Symbol)))
    def translate_eq_fa_s(self, expression):
        return self.walk(EQ(*expression.args[::-1]))

    @add_match(FunctionApplication(EQ_pattern, (Symbol, FunctionApplication)))
    def translate_eq_c_fa(self, expression):
        dst = Constant[ColumnStr](
            ColumnStr(expression.args[0].name),
            verify_type=False
        )
        processed_fa = self.walk(expression.args[1])
        return FunctionApplication(EQ, (dst, processed_fa))

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

    @add_match(
        FunctionApplication(Builtin_pattern, ...),
        lambda exp: any(isinstance(arg, NonConstant) for arg in exp.args)
    )
    def translate_builtin_fa(self, expression):
        args = expression.args
        new_args = tuple()
        changed = False
        for arg in args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Symbol):
                new_arg = Constant[ColumnStr](
                    ColumnStr(new_arg.name), verify_type=False
                )
                changed |= True
            else:
                changed |= new_arg is not arg
            new_args += (new_arg,)

        if changed:
            res = FunctionApplication(expression.functor, new_args)
        else:
            res = expression
        return res

    @add_match(FunctionApplication)
    def translate_fa(self, expression):
        functor = self.walk(expression.functor)
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
        classified_formulas = self.classify_formulas_obtain_names(expression)

        if len(classified_formulas['pos_formulas']) > 0:
            output = classified_formulas['pos_formulas'][0]
            for pos_formula in classified_formulas['pos_formulas'][1:]:
                output = NaturalJoin(output, pos_formula)
        else:
            return NamedRelationalAlgebraFrozenSet([])

        output = TranslateToNamedRA.process_negative_formulas(
            classified_formulas['neg_formulas'], classified_formulas['named_columns'],
            output
        )

        output = TranslateToNamedRA.process_equality_formulas(
            classified_formulas['eq_formulas'],
            classified_formulas['named_columns'],
            output
        )

        extended_projections = []
        for ext_proj in classified_formulas['ext_proj_formulas']:
            dst_column, fun_exp = ext_proj.args
            extended_projections.append(
                ExtendedProjectionListMember(fun_exp, dst_column)
            )
        if len(extended_projections) > 0:
            for column in classified_formulas['named_columns']:
                extended_projections.append(
                    ExtendedProjectionListMember(column, column)
                )
            output = ExtendedProjection(output, extended_projections)

        for selection in classified_formulas['selection_formulas']:
            output = Selection(output, selection)

        return output

    def classify_formulas_obtain_names(self, expression):
        pos_formulas = []
        neg_formulas = []
        selection_formulas = []
        ext_proj_formulas = []
        eq_formulas = []
        named_columns = set()
        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                neg_formulas.append(formula.formula)
            elif isinstance(formula, FunctionApplication):
                if formula.functor == EQ:
                    if formula.args[0] == formula.args[1]:
                        continue
                    elif isinstance(formula.args[1], (Constant, Symbol)):
                        eq_formulas.append(formula)
                    elif isinstance(formula.args[1], FunctionApplication):
                        ext_proj_formulas.append(formula)
                else:
                    selection_formulas.append(formula)
            else:
                pos_formulas.append(formula)
                if isinstance(formula, Constant):
                    named_columns.update(formula.value.columns)
                elif isinstance(formula, NameColumns):
                    named_columns.update(formula.column_names)
        return {
            'pos_formulas': pos_formulas,
            'neg_formulas': neg_formulas,
            'eq_formulas': eq_formulas,
            'ext_proj_formulas': ext_proj_formulas,
            'selection_formulas': selection_formulas,
            'named_columns': named_columns
        }

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
