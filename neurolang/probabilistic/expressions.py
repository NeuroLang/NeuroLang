from ..expressions import Definition, FunctionApplication, Symbol

PROB = Symbol("PROB")


class ProbabilisticPredicate(Definition):
    def __init__(self, probability, body):
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


class ProbabilisticQuery(FunctionApplication):
    pass
