from typing import Mapping

from ..exceptions import NeuroLangException
from ..expressions import Definition, Constant, Quantifier, Expression


class ProbQuantifier(Quantifier):
    def __init__(self, probability, body):
        if not isinstance(probability, Expression):
            raise NeuroLangException("Probability must be an expression")
        self.probability = probability
        self.body = body
        self._symbols = body._symbols | self.probability._symbols

    def __repr__(self):
        return "ProbQuantifier{{{} :: {} : {}}}".format(
            self.probability, self.body, self.type
        )


class Distribution(Definition):
    def __init__(self, parameters):
        self.parameters = parameters


class DiscreteDistribution(Distribution):
    pass


class TableDistribution(DiscreteDistribution):
    def __init__(self, table):
        self.table = table
        super().__init__(Constant[Mapping]({}))

    def __repr__(self):
        return "TableDistribution[\n{}\n]".format(
            "\n".join(
                [
                    f"\t{value}:\t{prob}"
                    for value, prob in self.table.value.items()
                ]
            )
        )
