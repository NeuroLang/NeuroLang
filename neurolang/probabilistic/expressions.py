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


class ProbabilisticChoice(Definition):
    def __init__(self, predicate):
        self.predicate = predicate


class Grounding(Definition):
    def __init__(self, expression, relation):
        self.expression = expression
        self.relation = relation
