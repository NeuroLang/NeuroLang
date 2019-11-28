from typing import Mapping

from ..exceptions import NeuroLangException
from ..expressions import Definition, Constant, Symbol, FunctionApplication
from ..relational_algebra import RelationalAlgebraOperation


class ProbabilisticPredicate(Definition):
    def __init__(self, probability, body):
        if not isinstance(probability, (Constant, Symbol)):
            raise NeuroLangException(
                "Probability must be a symbol or constant"
            )
        if not isinstance(body, FunctionApplication):
            raise NeuroLangException("Body must be a function application")
        self.probability = probability
        self.body = body
        self._symbols = body._symbols | self.probability._symbols

    def __repr__(self):
        return "ProbabilisticPredicate{{{} :: {} : {}}}".format(
            self.probability, self.body, self.type
        )


class Grounding(Definition):
    def __init__(self, expression, relation):
        self.expression = expression
        self.relation = relation


class PfactGrounding(Grounding):
    def __init__(self, expression, relation, params_relation):
        super().__init__(expression, relation)
        self.params_relation = params_relation


class GraphicalModel(Definition):
    def __init__(self, edges, cpds, groundings):
        self.edges = edges
        self.cpds = cpds
        self.groundings = groundings

    @property
    def random_variables(self):
        return set(self.cpds.value)


class Distribution(Definition):
    def __init__(self, parameters):
        self.parameters = parameters


class DiscreteDistribution(Distribution):
    pass


class TableDistribution(DiscreteDistribution):
    def __init__(self, table, parameters=Constant[Mapping]({})):
        self.table = table
        super().__init__(parameters)

    def __repr__(self):
        return "TableDistribution[\n{}\n]".format(
            "\n".join(
                [
                    f"\t{value}:\t{prob}"
                    for value, prob in self.table.value.items()
                ]
            )
        )


class VectorisedTableDistribution(TableDistribution):
    def __init__(self, table, grounding, parameters=Constant[Mapping]({})):
        self.grounding = grounding
        super().__init__(table, parameters)


class ConcatenateColumn(RelationalAlgebraOperation):
    def __init__(self, relation, column, column_values):
        self.relation = relation
        self.column = column
        self.column_values = column_values


class AddIndexColumn(RelationalAlgebraOperation):
    def __init__(self, relation, index_column):
        self.relation = relation
        self.index_column = index_column


class AddRepeatedValueColumn(RelationalAlgebraOperation):
    def __init__(self, relation, repeated_value):
        self.relation = relation
        self.repeated_value = repeated_value


class RenameColumns(RelationalAlgebraOperation):
    def __init__(self, relation, old_names, new_names):
        self.relation = relation
        self.old_names = old_names
        self.new_names = new_names


class ArithmeticOperationOnColumns(RelationalAlgebraOperation):
    def __init__(self, relation):
        self.relation = relation


class SumColumns(ArithmeticOperationOnColumns):
    pass


class MultiplyColumns(ArithmeticOperationOnColumns):
    pass


class DivideColumns(ArithmeticOperationOnColumns):
    def __init__(self, relation, numerator_column, denominator_column):
        super().__init__(relation)
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column


class RandomVariableValuePointer(Symbol):
    pass


class NegateProbability(RelationalAlgebraOperation):
    def __init__(self, relation):
        self.relation = relation


class MultipleNaturalJoin(RelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = relations


def make_numerical_col_symb():
    return Symbol("__numerical__" + Symbol.fresh().name)


class Aggregate(RelationalAlgebraOperation):
    def __init__(self, relation, group_columns, agg_column, dst_column):
        self.relation = relation
        self.group_columns = tuple(group_columns)
        self.agg_column = agg_column
        self.dst_column = dst_column


class SumAggregate(Aggregate):
    pass


class MeanAggregate(Aggregate):
    pass


class CountAggregate(Aggregate):
    pass


class ExtendedProjection(RelationalAlgebraOperation):
    """
    Projection operator extended to allow computation on components of tuples.

    The concept of extended projection is formally defined in section 5.2.5
    of [1]_.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, relation, projection_list):
        self.relation = relation
        self.projection_list = tuple(projection_list)

    def __repr__(self):
        join_str = "," if len(self.functions) < 2 else ",\n"
        return "π_[{}]({})".format(
            join_str.join([repr(fun) for fun in self.functions]),
            repr(self.relation),
        )


class ExtendedProjectionListMember(Definition):
    """
    Member of a projection list.

    As described in [1]_, a projection list member can either be
        - a single attribute (column) name in the relation, resulting in a
          normal non-extended projection,
        - an expression `x -> y` where `x` and `y` are both attribute (column)
          names, `x` effectively being rename as `y`,
        - or an expression `E -> z` where `E` is an expression involving
          attributes of the relation, arithmetic operators, and string
          operators, and `z` is a new name for the attribute that results from
          the calculation implied by `E`. For example, `a + b -> x` represents
          the sum of the attributes `a` and `b`, renamed `x`.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, fun_exp, dst_column):
        self.fun_exp = fun_exp
        self.dst_column = dst_column

    def __repr__(self):
        return "{} -> {}".format(self.fun_exp, self.dst_column)
