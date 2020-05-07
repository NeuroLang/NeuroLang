from operator import contains, eq, invert
from typing import AbstractSet, Callable, Tuple

from ..exceptions import NeuroLangException
from ..expression_walker import (ExpressionBasicEvaluator,
                                 ReplaceExpressionsByValues, add_match)
from ..expressions import Constant, FunctionApplication, NonConstant, Symbol
from ..relational_algebra import (ColumnInt, ColumnStr, Destroy, Difference,
                                  ExtendedProjection,
                                  ExtendedProjectionListMember, NameColumns,
                                  NaturalJoin, Projection,
                                  RelationalAlgebraOperation, RenameColumn,
                                  Selection)
from ..utils import NamedRelationalAlgebraFrozenSet
from .expressions import Conjunction, Negation

EQ = Constant(eq)
CONTAINS = Constant(contains)
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
        processed_fa = self.walk(expression.args[1])
        if isinstance(processed_fa, RelationalAlgebraOperation):
            processed_fa = expression.args[1]
        if processed_fa is not expression.args[1]:
            res = self.walk(
                FunctionApplication(EQ, (expression.args[0], processed_fa))
            )
        else:
            dst = Constant[ColumnStr](
                ColumnStr(expression.args[0].name),
                verify_type=False
            )
            res = FunctionApplication(EQ, (dst, processed_fa))
        return res

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

    @add_match(
        FunctionApplication(Builtin_pattern, ...),
        lambda exp: all(
            isinstance(arg, Constant) and
            not issubclass(arg.type, (ColumnInt, ColumnStr))
            for arg in exp.args
        )
    )
    def translate_builtin_fa_constants(self, expression):
        return ExpressionBasicEvaluator.evaluate_function(self, expression)

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
            classified_formulas,
            output
        )

        output = TranslateToNamedRA.process_equality_formulas(
            classified_formulas,
            output
        )

        output = TranslateToNamedRA.process_extended_projection_formulas(
            classified_formulas,
            output
        )

        for selection in classified_formulas['selection_formulas']:
            output = Selection(output, selection)

        for destroy in classified_formulas['destroy_formulas']:
            output = Destroy(output, destroy.args[1], destroy.args[0])

        return output

    def classify_formulas_obtain_names(self, expression):
        classified_formulas = {
            'pos_formulas': [],
            'neg_formulas': [],
            'eq_formulas': [],
            'ext_proj_formulas': [],
            'selection_formulas': [],
            'destroy_formulas': [],
            'named_columns': set()
        }
        
        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                classified_formulas['neg_formulas'].append(formula.formula)
            elif isinstance(formula, FunctionApplication):
                self.classify_formulas_obtain_named_function_applications(
                    formula, classified_formulas
                )
            else:
                classified_formulas['pos_formulas'].append(formula)
                if isinstance(formula, Constant):
                    classified_formulas['named_columns'].update(
                        formula.value.columns
                    )
                elif isinstance(formula, NameColumns):
                    classified_formulas['named_columns'].update(
                        formula.column_names
                    )
        return classified_formulas

    def classify_formulas_obtain_named_function_applications(
        self, formula, classified_formulas
    ):
        if formula.functor == EQ:
            if formula.args[0] == formula.args[1]:
                pass
            elif isinstance(formula.args[1], (Constant, Symbol)):
                classified_formulas['eq_formulas'].append(formula)
            elif isinstance(formula.args[1], FunctionApplication):
                classified_formulas['ext_proj_formulas'].append(formula)
        elif formula.functor == CONTAINS:
            classified_formulas['destroy_formulas'].append(formula)
        else:
            classified_formulas['selection_formulas'].append(formula)

    @staticmethod
    def process_negative_formulas(classified_formulas, output):
        named_columns = classified_formulas['named_columns']
        for neg_formula in classified_formulas['neg_formulas']:
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
    def process_equality_formulas(classified_formulas, output):
        named_columns = classified_formulas['named_columns']
        for formula in classified_formulas['eq_formulas']:
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

    @staticmethod
    def process_extended_projection_formulas(classified_formulas, output):
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
        return output
