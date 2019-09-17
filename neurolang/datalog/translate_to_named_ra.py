from typing import AbstractSet, Tuple

from ..expression_walker import add_match, ExpressionBasicEvaluator
from .expressions import Conjunction, Negation
from ..expressions import FunctionApplication, Constant, Symbol
from ..relational_algebra import NaturalJoin, Difference
from ..utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
from ..exceptions import NeuroLangException


class TranslateToNamedRA(ExpressionBasicEvaluator):
    """Partial implementation of algorithm 5.4.8 from [1]_.

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
       (Addison Wesley, 1995), Addison-Wesley.

    Parameters
    ----------
    symbol_table : Mapping
        Mapping from symbols to `RelationalAlgebraFrozenSet` instances.
    """

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(FunctionApplication)
    def translate_fa(self, expression):
        functor = expression.functor
        if isinstance(functor, Symbol):
            if functor in self.symbol_table:
                functor = self.symbol_table[functor]
                if not (
                    isinstance(functor, Constant[AbstractSet]) and
                    isinstance(functor.value, RelationalAlgebraFrozenSet)
                ):
                    raise NeuroLangException(
                        "Sets must be instances of in the instance and be "
                        "an instance of RelationalAlgebraFrozenSet"
                    )
            else:
                inner_type = tuple(arg.type for arg in expression.args)
                functor = Constant[AbstractSet[Tuple[inner_type]]](set())

        named_args = []
        projections = []
        new_type = tuple()
        selections = dict()
        selection_columns = dict()
        set_type = functor.type.__args__[0].__args__
        for i, arg in enumerate(expression.args):
            if isinstance(arg, Constant):
                selections[i] = arg.value
            elif arg in named_args:
                selection_columns[i] = named_args.index(arg)
            else:
                projections.append(i)
                named_args.append(arg.name)
                new_type += (set_type[i],)

        in_set = functor.value
        if len(in_set) > 0:
            if len(selections) > 0:
                in_set = in_set.selection(selections)
            if len(selection_columns) > 0:
                in_set = in_set.selection_columns(selection_columns)
            in_set = in_set.projection(*projections)

        out_set = Constant[AbstractSet[Tuple[new_type]]](
            NamedRelationalAlgebraFrozenSet(
                named_args, in_set
            )
        )

        return out_set

    @add_match(Negation)
    def translate_negation(self, expression):
        if isinstance(expression.formula, Negation):
            return self.walk(expression.formula.formula)
        else:
            return Negation(self.walk(expression.formula))

    @add_match(Conjunction)
    def translate_conj(self, expression):
        pos_formulas = []
        neg_formulas = []
        named_columns = set()
        for formula in expression.formulas:
            formula = self.walk(formula)
            if isinstance(formula, Negation):
                neg_formulas.append(formula.formula)
            else:
                pos_formulas.append(formula)
                named_columns.update(formula.value.columns)

        if len(pos_formulas) > 0:
            pos_formulas = sorted(pos_formulas, key=lambda x: len(x.value))
            output = pos_formulas[0]
            for pos_formula in pos_formulas[1:]:
                output = NaturalJoin(output, pos_formula)
        else:
            return NamedRelationalAlgebraFrozenSet([])

        for neg_formula in neg_formulas:
            neg_cols = set(neg_formula.value.columns)
            if named_columns > neg_cols:
                neg_formula = NaturalJoin(output, neg_formula)
            elif named_columns != neg_cols:
                raise NeuroLangException(
                    f'Negative predicate {neg_formula} is not safe range'
                )
            output = Difference(output, neg_formula)

        return output
