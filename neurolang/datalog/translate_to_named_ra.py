import collections
from operator import contains, eq, not_
from typing import AbstractSet, Callable, Tuple

from ..exceptions import ForbiddenExpressionError, NeuroLangException
from ..expression_walker import (
    ExpressionBasicEvaluator,
    ReplaceExpressionsByValues,
    add_match,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Disjunction
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    Destroy,
    Difference,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    NaturalJoin,
    Projection,
    RelationalAlgebraOperation,
    Selection,
    Union,
    get_expression_columns,
)
from ..type_system import is_leq_informative
from ..utils import NamedRelationalAlgebraFrozenSet
from .expressions import Conjunction, Negation

EQ = Constant(eq)
CONTAINS = Constant(contains)
EQ_pattern = Constant[Callable](eq)
Builtin_pattern = Constant[Callable]
REBV = ReplaceExpressionsByValues({})


class TranslateToNamedRAException(NeuroLangException):
    pass


class CouldNotTranslateConjunctionException(TranslateToNamedRAException):
    def __init__(self, output):
        super().__init__(f"Could not translate conjunction: {output}")
        self.output = output


class NegativeFormulaNotSafeRangeException(TranslateToNamedRAException):
    def __init__(self, formula):
        super().__init__(f"Negative predicate {formula} is not safe range")
        self.formula = formula


class NegativeFormulaNotNamedRelationException(TranslateToNamedRAException):
    def __init__(self, formula):
        super().__init__(f"Negative formula {formula} is not a named relation")
        self.formula = formula


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
        return self.walk(
            FunctionApplication(
                EQ,
                (
                    Constant[ColumnStr](symbol.name, verify_type=False),
                    constant,
                ),
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
                ColumnStr(expression.args[0].name), verify_type=False
            )
            res = FunctionApplication(EQ, (dst, processed_fa))
        return res

    @add_match(FunctionApplication(EQ_pattern, (Symbol, Symbol)))
    def translate_eq_c_c(self, expression):
        left = Constant[ColumnStr](
            ColumnStr(expression.args[0].name), verify_type=False
        )
        right = Constant[ColumnStr](
            ColumnStr(expression.args[1].name), verify_type=False
        )
        res = FunctionApplication(EQ, (left, right))
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
        lambda exp: len(exp._symbols) > 0,
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
            elif isinstance(new_arg, Constant[Tuple]) and all(
                isinstance(v, Symbol) for v in new_arg.value
            ):
                n = len(new_arg.value)
                new_arg = Constant[Tuple[(ColumnStr,) * n]](
                    tuple(ColumnStr(v.name) for v in new_arg.value)
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
            isinstance(arg, Constant)
            and not issubclass(arg.type, (ColumnInt, ColumnStr))
            for arg in exp.args
        ),
    )
    def translate_builtin_fa_constants(self, expression):
        return ExpressionBasicEvaluator.evaluate_function(self, expression)

    @add_match(FunctionApplication)
    def translate_fa(self, expression):
        functor = self.walk(expression.functor)
        named_args = list()
        projections = list()
        selections = dict()
        selection_columns = dict()
        stack = list(reversed(expression.args))
        counter = 0
        while stack:
            arg = stack.pop()
            if isinstance(arg, Constant):
                selections[counter] = arg
            elif arg in named_args:
                selection_columns[counter] = named_args.index(arg)
            elif isinstance(arg, FunctionApplication):
                stack += list(reversed(arg.args))
            else:
                projections.append(
                    Constant[ColumnInt](counter, verify_type=False)
                )
                named_args.append(arg)
            counter += 1
        in_set = self.generate_ra_expression(
            functor,
            selections,
            selection_columns,
            tuple(projections),
            tuple(named_args),
        )
        return in_set

    def generate_ra_expression(
        self, functor, selections, selection_columns, projections, named_args
    ):
        in_set = functor
        for k, v in selections.items():
            criterium = EQ(Constant[ColumnInt](k, verify_type=False), v)
            in_set = Selection(in_set, criterium)
        for k, v in selection_columns.items():
            criterium = EQ(
                Constant[ColumnInt](k, verify_type=False),
                Constant[ColumnInt](v, verify_type=False),
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

        formula = expression.formula
        if isinstance(formula, FunctionApplication) and isinstance(
            formula.functor, Constant
        ):
            res = FunctionApplication(Constant(not_), (formula,))
            res = self.walk(res)
        else:
            res = Negation(self.walk(expression.formula))
        return res

    @add_match(Disjunction)
    def translate_disjunction(self, expression):
        ra_formulas = self.walk(expression.formulas)
        ra_formulas = list(ra_formulas)
        formula = ra_formulas.pop()
        while len(ra_formulas) > 0:
            formula_ = ra_formulas.pop()
            formula = Union(formula_, formula)

        return formula

    @add_match(Conjunction)
    def translate_conjunction(self, expression):
        classified_formulas = self.classify_formulas_obtain_names(expression)

        output = self.process_positive_formulas(classified_formulas)

        output = self.process_negative_formulas(classified_formulas, output)

        while (
            len(classified_formulas["destroy_formulas"])
            + len(classified_formulas["selection_formulas"])
            + len(classified_formulas["eq_formulas"])
            + len(classified_formulas["ext_proj_formulas"])
        ) > 0:
            new_output = self.process_destroy_formulas(
                classified_formulas, output
            )

            new_output = self.process_equality_formulas(
                classified_formulas, new_output
            )

            new_output = self.process_extended_projection_formulas(
                classified_formulas, new_output
            )

            new_output = self.process_selection_formulas(
                classified_formulas, new_output
            )

            if new_output is output:
                new_output = self.process_equality_formulas_as_extended_projections(
                    classified_formulas, new_output
                )

            if new_output is output:
                raise CouldNotTranslateConjunctionException(expression)
            output = new_output

        return output

    def classify_formulas_obtain_names(self, expression):
        classified_formulas = {
            "pos_formulas": [],
            "neg_formulas": [],
            "eq_formulas": [],
            "ext_proj_formulas": [],
            "selection_formulas": [],
            "destroy_formulas": [],
            "named_columns": set(),
        }

        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                classified_formulas["neg_formulas"].append(formula.formula)
            elif isinstance(formula, FunctionApplication):
                self.classify_formulas_obtain_named_function_applications(
                    formula, classified_formulas
                )
            else:
                classified_formulas["pos_formulas"].append(formula)
                if isinstance(formula, Constant):
                    classified_formulas["named_columns"].update(
                        formula.value.columns
                    )
                elif isinstance(formula, NameColumns):
                    classified_formulas["named_columns"].update(
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
                classified_formulas["eq_formulas"].append(formula)
            elif isinstance(formula.args[1], FunctionApplication):
                classified_formulas["ext_proj_formulas"].append(formula)
        elif formula.functor == CONTAINS and (
            isinstance(formula.args[1], Constant[ColumnStr])
            or isinstance(formula.args[1], Constant[Tuple])
        ):
            classified_formulas["destroy_formulas"].append(formula)
        else:
            classified_formulas["selection_formulas"].append(formula)

    @staticmethod
    def process_positive_formulas(classified_formulas):
        if len(classified_formulas["pos_formulas"]) == 0:
            output = Constant[AbstractSet](NamedRelationalAlgebraFrozenSet([]))
        else:
            output = classified_formulas["pos_formulas"][0]
            for pos_formula in classified_formulas["pos_formulas"][1:]:
                output = NaturalJoin(output, pos_formula)

        return output

    @staticmethod
    def process_negative_formulas(classified_formulas, output):
        named_columns = classified_formulas["named_columns"]
        for neg_formula in classified_formulas["neg_formulas"]:
            neg_cols = TranslateToNamedRA.obtain_negative_columns(neg_formula)
            if named_columns > neg_cols:
                neg_formula = NaturalJoin(output, neg_formula)
            elif named_columns != neg_cols:
                raise NegativeFormulaNotSafeRangeException(neg_formula)
            output = Difference(output, neg_formula)
        return output

    @staticmethod
    def obtain_negative_columns(neg_formula):
        if isinstance(neg_formula, NameColumns):
            neg_cols = set(neg_formula.column_names)
        elif isinstance(neg_formula, Constant):
            neg_cols = set(neg_formula.value.columns)
        else:
            raise NegativeFormulaNotNamedRelationException(neg_formula)
        return neg_cols

    @staticmethod
    def process_destroy_formulas(classified_formulas, output):
        destroy_to_keep = []
        named_columns = classified_formulas["named_columns"]
        for destroy in classified_formulas["destroy_formulas"]:
            if destroy.args[0] in named_columns:
                output = Destroy(output, destroy.args[0], destroy.args[1])
                if is_leq_informative(destroy.args[1].type, Tuple):
                    for arg in destroy.args[1].value:
                        named_columns.add(Constant(arg))
                else:
                    named_columns.add(destroy.args[1])
            else:
                destroy_to_keep.append(destroy)
        classified_formulas["destroy_formulas"] = destroy_to_keep
        return output

    @staticmethod
    def process_equality_formulas(classified_formulas, output):
        named_columns = classified_formulas["named_columns"]
        to_keep = []
        for formula in classified_formulas["eq_formulas"]:
            new_output = TranslateToNamedRA.process_equality_formula(
                formula, named_columns, output
            )
            if new_output is output:
                to_keep.append(formula)
            output = new_output
        classified_formulas["eq_formulas"] = to_keep
        return output

    @staticmethod
    def process_equality_formula(formula, named_columns, output):
        left, right = formula.args

        if (
            isinstance(left, Constant[ColumnStr])
            and isinstance(right, Constant)
            and not isinstance(right, Constant[ColumnStr])
        ):
            return TranslateToNamedRA.process_equality_formulas_constant(
                output, left, right, named_columns
            )

        criteria = EQ(left, right)
        if left in named_columns and right in named_columns:
            output = Selection(output, criteria)
        return output

    @staticmethod
    def process_equality_formulas_constant(output, left, right, named_columns):
        if isinstance(output, Constant[AbstractSet]) and output.value.is_dum():
            return Constant[AbstractSet[Tuple[right.type]]](
                NamedRelationalAlgebraFrozenSet(
                    (left.value,), [(REBV.walk(right),)]
                )
            )
        elif left in named_columns:
            return Selection(output, EQ(left, right))
        else:
            return output

    @staticmethod
    def process_equality_formulas_as_extended_projections(
        classified_formulas, output
    ):
        named_columns = classified_formulas["named_columns"]
        extended_projections = tuple(
            ExtendedProjectionListMember(c, c) for c in named_columns
        )
        stack = list(classified_formulas["eq_formulas"])
        if len(stack) == 0:
            return output
        seen_counts = collections.defaultdict(int)
        while stack:
            formula = stack.pop()
            seen_counts[formula] += 1
            if seen_counts[formula] > 2:
                raise ForbiddenExpressionError(
                    f"Could not resolve equality {formula}"
                )
            # case y = x where y already in set (create new column x)
            if formula.args[0] in named_columns:
                src, dst = formula.args
            elif (
                # case x = y where y already in set (create new column x)
                formula.args[1] in named_columns
                # case x = C where C is a constant (create new constant col x)
                or TranslateToNamedRA.is_col_to_const_equality(formula)
            ):
                dst, src = formula.args
            else:
                stack.insert(0, formula)
                continue
            extended_projections += (ExtendedProjectionListMember(src, dst),)
            named_columns.add(dst)
            seen_counts = collections.defaultdict(int)
        new_output = ExtendedProjection(output, extended_projections)
        classified_formulas["eq_formulas"] = []
        return new_output

    @staticmethod
    def is_col_to_const_equality(formula):
        return (
            isinstance(formula.args[0], Constant)
            and formula.args[0].type is ColumnStr
            and isinstance(formula.args[1], Constant)
            and formula.args[1].type is not ColumnStr
        )

    @staticmethod
    def process_extended_projection_formulas(classified_formulas, output):
        extended_projections = []
        to_keep = []
        named_columns = classified_formulas["named_columns"]
        dst_columns = set()
        for ext_proj in classified_formulas["ext_proj_formulas"]:
            dst_column, fun_exp = ext_proj.args
            cols_for_fun_exp = get_expression_columns(fun_exp)
            if cols_for_fun_exp.issubset(named_columns):
                extended_projections.append(
                    ExtendedProjectionListMember(fun_exp, dst_column)
                )
                dst_columns.add(dst_column)
            else:
                to_keep.append(ext_proj)
        if len(extended_projections) > 0:
            for column in classified_formulas["named_columns"]:
                extended_projections.append(
                    ExtendedProjectionListMember(column, column)
                )
            output = ExtendedProjection(output, extended_projections)

        named_columns |= dst_columns
        classified_formulas["ext_proj_formulas"] = to_keep
        return output

    @staticmethod
    def process_selection_formulas(classified_formulas, output):
        to_keep = []
        for selection in classified_formulas["selection_formulas"]:
            selection_columns = get_expression_columns(selection)
            if selection_columns.issubset(
                classified_formulas["named_columns"]
            ):
                output = Selection(output, selection)
            else:
                to_keep.append(selection)
        classified_formulas["selection_formulas"] = to_keep
        return output
