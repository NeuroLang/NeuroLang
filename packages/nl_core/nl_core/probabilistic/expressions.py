from nl_expr.expressions import (
    Constant, Definition, Expression, FunctionApplication, Symbol,
)
from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.logic import (
    TRUE,
    BinaryLogicOperator,
    LogicOperator,
    NaryLogicOperator,
)

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


class Expectation(LogicOperator):
    """
    ``EXPECT[function | event // known] = result`` in the body of a rule:
    the expected value of *function* summed over the answers of *event*,
    knowing *known* (``TRUE`` for none); *normalised*, a boolean constant,
    for ``EXPECT_NORM``.
    """

    def __init__(self, function, event, known, normalised, result):
        self.function = function
        self.event = event
        self.known = known
        self.normalised = normalised
        self.result = result
        self._symbols = (
            function._symbols | event._symbols | known._symbols
            | result._symbols
        )

    def __repr__(self):
        return "EXPECT{}[{} | {} // {}] = {}".format(
            "_NORM" if self.normalised.value else "", self.function,
            self.event, self.known, self.result,
        )


def expectation(function, event, result, known=None, normalised=False):
    """
    The ``Expectation`` of *function* over *event*, knowing *known* if
    given, bound to *result*: what the parser and the query builder
    (:meth:`NeurolangPDL.expect`) both build.
    """
    return Expectation(
        function, event, TRUE if known is None else known,
        Constant(normalised), result,
    )


class _HasExpectation(PatternWalker):
    """Walk the logic operators and the arguments of applications, not the
    contents of constants, which may be large."""

    @add_match(Expectation)
    def expectation(self, expression):
        return True

    @add_match(NaryLogicOperator)
    def nary(self, expression):
        return any(self.walk(f) for f in expression.formulas)

    @add_match(LogicOperator)
    def operator(self, expression):
        return any(self.walk(f) for f in expression.unapply())

    @add_match(FunctionApplication)
    def application(self, expression):
        return any(self.walk(arg) for arg in expression.args)

    @add_match(Expression)
    def atom(self, expression):
        return False


def contains_expectation(expression):
    """Whether *expression* is, or holds, an ``Expectation``."""
    return _HasExpectation().walk(expression)


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
