from typing import Mapping

from ..exceptions import NeuroLangException
from ..expressions import Constant, Definition, FunctionApplication, Symbol


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


class ProbabilisticChoice(Definition):
    def __init__(self, predicate):
        self.predicate = predicate


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


class ChoiceDistribution(DiscreteDistribution):
    def __init__(self, grounding):
        self.grounding = grounding


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


class SuccQuery(Definition):
    def __init__(self, predicate):
        self.predicate = predicate

    def __repr__(self):
        return "SUCC( {} )".format(repr(self.predicate))


class MargQuery(Definition):
    def __init__(self, predicate, evidence):
        self.predicate = predicate
        self.evidence = evidence


class VectorisedTableDistribution(TableDistribution):
    def __init__(self, table, grounding, parameters=Constant[Mapping]({})):
        self.grounding = grounding
        super().__init__(table, parameters)


class RandomVariableValuePointer(Symbol):
    pass
