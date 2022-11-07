from ..expressions import Definition, FunctionApplication, Symbol
from ..logic import BinaryLogicOperator

PROB = Symbol("PROB")


class ProbabilisticBinaryLogicOperator(BinaryLogicOperator):
    pass


class Condition(ProbabilisticBinaryLogicOperator):
    def __init__(self, conditioned, conditioning):
        self.conditioned = conditioned
        self.conditioning = conditioning
        self._symbols = conditioned._symbols | conditioning._symbols

    def __repr__(self):
        return f"[{self.conditioned} | {self.conditioning}]"


class ProbabilisticPredicate(Definition):
    def __init__(self, probability, body):
        self.probability = probability
        self.body = body
        self._symbols = body._symbols | self.probability._symbols

    @property
    def functor(self):
        return self.body.functor

    @property
    def args(self):
        return self.body.args

    def __repr__(self):
        return "ProbabilisticPredicate{{{} :: {} : {}}}".format(
            self.body, self.probability, self.type
        )


class ProbabilisticChoice(ProbabilisticPredicate):
    def __repr__(self):
        return "ProbabilisticChoice{{{} :: {} : {}}}".format(
            self.body, self.probability, self.type
        )


class ProbabilisticFact(ProbabilisticPredicate):
    def __repr__(self):
        return "ProbabilisticFact{{{} :: {} : {}}}".format(
            self.body, self.probability, self.type
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


class ProbabilisticQuery(FunctionApplication):
    pass
