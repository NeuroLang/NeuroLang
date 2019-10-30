from typing import Mapping

from ..exceptions import NeuroLangException
from ..expressions import Definition, Constant, Symbol, FunctionApplication


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
    def __init__(self, expression, algebra_set):
        self.expression = expression
        self.algebra_set = algebra_set


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


class VectorisedTableDistributionOperation(Definition):
    pass


class VectorPointer(Symbol):
    pass


class RandomVariableVectorPointer(VectorPointer):
    pass


class ParameterVectorPointer(VectorPointer):
    pass


class VectorBinaryOperation(Definition):
    def __init__(self, first, second):
        self.first = first
        self.second = second


class ReindexVector(VectorBinaryOperation):
    @property
    def vector(self):
        return self.first

    @property
    def index(self):
        return self.second


class SumVectors(VectorBinaryOperation):
    pass


class SubtractVectors(VectorBinaryOperation):
    pass


class MultiplyVectors(VectorBinaryOperation):
    pass


class IndexedGrounding(Grounding):
    def __init__(self, expression, algebra_set, index_columns):
        self.index_columns = index_columns
        super().__init__(expression, algebra_set)
