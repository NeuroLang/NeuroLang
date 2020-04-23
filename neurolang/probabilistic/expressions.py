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


class Grounding(Definition):
    def __init__(self, expression, relation):
        self.expression = expression
        self.relation = relation

    def __repr__(self):
        return "Grounding{{{}}}\n{}".format(
            repr(self.expression), repr(self.relation)
        )


class ProbabilisticChoiceGrounding(Grounding):
    """
    Class used to differentiate the grounding of a probabilistic
    choice from the grounding of other choices.

    """

    def __init__(self, expression, relation, probability_column):
        super().__init__(expression, relation)
        self.probability_column = probability_column


class GraphicalModel(Definition):
    def __init__(self, edges, cpd_factories, expressions):
        self.edges = edges
        self.cpd_factories = cpd_factories
        self.expressions = expressions

    @property
    def random_variables(self):
        return set(self.cpd_factories.value)
