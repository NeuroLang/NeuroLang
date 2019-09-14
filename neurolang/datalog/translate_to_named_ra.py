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
            functor = self.symbol_table[functor]

        if not (
            isinstance(functor, Constant[AbstractSet]) and
            isinstance(functor.value, RelationalAlgebraFrozenSet)
        ):
            raise NeuroLangException(
                "Sets must be instances of in the instance and be "
                "an instance of RelationalAlgebraFrozenSet"
            )

        named_args = []
        projections = []
        new_type = tuple()
        selections = dict()
        set_type = functor.type.__args__[0].__args__
        for i, arg in enumerate(expression.args):
            if isinstance(arg, Constant):
                selections[i] = arg.value
            else:
                projections.append(i)
                named_args.append(arg.name)
                new_type += (set_type[i],)

        in_set = functor.value
        if len(selections) > 0:
            in_set = in_set.selection(selections)
        if len(projections) > 0:
            in_set = in_set.projection(*projections)

        out_set = Constant[AbstractSet[Tuple[new_type]]](
            NamedRelationalAlgebraFrozenSet(
                named_args, in_set
            )
        )

        return out_set

    @add_match(Negation)
    def translate_negation(self, expression):
        if isinstance(expression.literal, Negation):
            return self.walk(expression.literal.literal)
        else:
            return Negation(self.walk(expression.literal))

    @add_match(Conjunction)
    def translate_conj(self, expression):
        pos_literals = []
        neg_literals = []
        named_columns = set()
        for literal in expression.literals:
            literal = self.walk(literal)
            if isinstance(literal, Negation):
                neg_literals.append(literal.literal)
            else:
                pos_literals.append(literal)
                named_columns.update(literal.value.columns)

        if len(pos_literals) > 0:
            pos_literals = sorted(pos_literals, key=lambda x: len(x.value))
            output = pos_literals[0]
            for pos_literal in pos_literals[1:]:
                output = NaturalJoin(output, pos_literal)
        else:
            return NamedRelationalAlgebraFrozenSet([])

        for neg_literal in neg_literals:
            neg_cols = set(neg_literal.value.columns)
            if named_columns > neg_cols:
                neg_literal = NaturalJoin(output, neg_literal)
            elif named_columns != neg_cols:
                raise NeuroLangException(
                    f'Negative predicate {neg_literal} is not safe range'
                )
            output = Difference(output, neg_literal)

        return output
