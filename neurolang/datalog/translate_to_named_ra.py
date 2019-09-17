from operator import eq
from typing import AbstractSet, Tuple

from ..exceptions import NeuroLangException
from ..expression_walker import ExpressionBasicEvaluator, add_match
from ..expressions import Constant, FunctionApplication, Symbol
from ..relational_algebra import (Column, Difference, NameColumns, NaturalJoin,
                                  Projection, Selection)
from ..utils import NamedRelationalAlgebraFrozenSet
from .expressions import Conjunction, Negation

EQ = Constant(eq)


class TranslateToNamedRA(ExpressionBasicEvaluator):
    """Partial implementation of algorithm 5.4.8 from [1]_.

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
       (Addison Wesley, 1995), Addison-Wesley.
    """
    @add_match(FunctionApplication(EQ, (Constant, Symbol)))
    def translate_eq_c_s(self, expression):
        return self.walk(EQ(*expression.args[::-1]))

    @add_match(FunctionApplication(EQ, (Symbol, Constant)))
    def translate_eq_s_c(self, expression):
        symbol, constant = expression.args
        return Constant[AbstractSet[Tuple[constant.type]]](
            NamedRelationalAlgebraFrozenSet((symbol.name,), (constant.value,))
        )

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
                projections += (Constant[Column](i, verify_type=False),)
                named_args += (arg,)

        in_set = functor
        for k, v in selections.items():
            criterium = EQ(
                Constant[Column](k, verify_type=False), v
            )
            in_set = Selection(in_set, criterium)
        for k, v in selection_columns.items():
            criterium = EQ(
                Constant[Column](k, verify_type=False),
                Constant[Column](v, verify_type=False)
            )
            in_set = Selection(in_set, criterium)

        in_set = Projection(in_set, projections)
        in_set = NameColumns(in_set, named_args)

        return in_set

    @add_match(Negation)
    def translate_negation(self, expression):
        if isinstance(expression.formula, Negation):
            return self.walk(expression.formula.formula)
        else:
            return Negation(self.walk(expression.formula))

    @add_match(Conjunction)
    def translate_conj(self, expression):
        pos_formulas, neg_formulas, named_columns = \
            self.classify_formulas_obtain_names(expression)

        if len(pos_formulas) > 0:
            output = pos_formulas[0]
            for pos_formula in pos_formulas[1:]:
                output = NaturalJoin(output, pos_formula)
        else:
            return NamedRelationalAlgebraFrozenSet([])

        for neg_formula in neg_formulas:
            if isinstance(neg_formula, NameColumns):
                neg_cols = set(neg_formula.column_names)
            elif isinstance(neg_formula, Constant):
                neg_cols = set(neg_formula.value.columns)
            else:
                raise NeuroLangException(
                    f"Negative formula {neg_formula} is  not a named relation"
                )
            if named_columns > neg_cols:
                neg_formula = NaturalJoin(output, neg_formula)
            elif named_columns != neg_cols:
                raise NeuroLangException(
                    f'Negative predicate {neg_formula} is not safe range'
                )
            output = Difference(output, neg_formula)

        return output

    def classify_formulas_obtain_names(self, expression):
        pos_formulas = []
        neg_formulas = []
        named_columns = set()
        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                neg_formulas.append(formula.formula)
            else:
                pos_formulas.append(formula)
                if isinstance(formula, Constant):
                    named_columns.update(formula.value.columns)
                elif isinstance(formula, NameColumns):
                    named_columns.update(formula.column_names)
        return pos_formulas, neg_formulas, named_columns
