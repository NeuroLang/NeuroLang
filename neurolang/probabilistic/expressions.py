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


class RandomVariableValuePointer(Symbol):
    pass


class NegateProbability(RelationalAlgebraOperation):
    def __init__(self, relation):
        self.relation = relation


class MultipleNaturalJoin(RelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = relations
