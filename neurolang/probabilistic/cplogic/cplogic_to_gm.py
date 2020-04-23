from typing import AbstractSet, Mapping

from ...expression_pattern_matching import add_match
from ...expression_walker import PatternWalker
from ...expressions import (
    Constant,
    Definition,
    ExpressionBlock,
    FunctionApplication,
    Symbol,
)
from ...logic import Implication
from ...logic.expression_processing import extract_logic_predicates
from ...relational_algebra import (
    ConcatenateConstantColumn,
    RelationalAlgebraSolver,
    str2columnstr_constant,
)
from ..expressions import (
    GraphicalModel,
    Grounding,
    ProbabilisticPredicate,
    ProbabilisticChoiceGrounding,
)
from .grounding import topological_sort_groundings


class CPDFactory(Definition):
    """
    Object used to produce conditional probability distributions of
    multiple random variables at the same time.

    Each random variable is represented by a row in the ``relation``
    attribute of the factory.

    Attributes
    ----------
    relation : Constant[AbstractSet]
        A relation that encodes all the variables for which a
        conditional probability distribution will be produced.

    """

    def __init__(self, relation):
        self.relation = relation


class BernoulliCPDFactory(CPDFactory):
    """
    Object used to produce Bernoulli-distributed random variables.

    Although this object is called a "CPD" factory, there is no parent
    to this distribution. Instead, it represents the marginal
    distributions of a batch of independent Bernoulli random variables.

    An extra column in the relation used to represent the random
    variables is used to encode the probability of that specific
    random variable's Bernoulli distribution:

    For example, to represent the program

        P(a) : 0.7 <- T
        P(b) : 0.2 <- T

    One would use the following relation

         _p_ | x
        =====|===
         0.7 | a
         0.2 | b

    where each row represents one random variable P(x) and where the
    column _p_ encodes the Bernoulli parameter of the distribution
    of P(x).

    Attributes
    ----------
    relation: Constant[AbstractSet]
        A relation that encodes all the variables for which a
        conditional probability distribution will be produced.
    probability_column: Constant[ColumnStr]
        The column that contains the Bernoulli parameter associated
        to each random variable.

    """

    def __init__(self, relation, probability_column):
        super().__init__(relation)
        self.probability_column = probability_column


class AndCPDFactory(CPDFactory):
    """
    Object used to produce deterministic conditional probabilities
    that conjunct the values of boolean parent random variables.

    It only makes sense to use this CPD if all the parents are
    boolean random variables and if the random variable itself
    is a boolean random variable.

    Let X be the random variable for which the CPD is calculated
    and let Y_1, ..., Y_n be n parent boolean random variables.

    The AND CPD encodes that
                                              |- 1  if all y_i = T
        P(X = T | Y_1 = y_1, ... Y_n = y_1) = |
                                              |- 0  otherwise

    """


def is_extensional_grounding(grounding):
    """TODO: represent extensional grounding with a fact instead?"""
    return (
        isinstance(grounding.expression, Implication)
        and isinstance(grounding.expression.consequent, FunctionApplication)
        and grounding.expression.antecedent == Constant[bool](True)
    )


class CPLogicGroundingToGraphicalModelTranslator(PatternWalker):
    def __init__(self):
        self.edges = dict()
        self.cpd_factories = dict()
        self.expressions = dict()

    @add_match(ExpressionBlock)
    def block_of_groundings(self, block):
        if any(not isinstance(exp, Grounding) for exp in block.expressions):
            raise ValueError("Expected block of groundings")
        for grounding in topological_sort_groundings(block.expressions):
            self.walk(grounding)
        return GraphicalModel(
            Constant[Mapping](self.edges),
            Constant[Mapping](self.cpd_factories),
            Constant[Mapping](self.expressions),
        )

    @add_match(Grounding, is_extensional_grounding)
    def extensional_set_grounding(self, grounding):
        """
        Represent a set of ground facts as a Bernoulli node with
        all probabilities set to 1.0.
        """
        rv_symb = grounding.expression.consequent.functor
        probability_column = str2columnstr_constant(Symbol.fresh().name)
        relation = ConcatenateConstantColumn(
            grounding.relation, probability_column, Constant[float](1.0)
        )
        relation = RelationalAlgebraSolver().walk(relation)
        cpd_factory = BernoulliCPDFactory(relation, probability_column)
        expression = grounding.expression
        self.add_random_variable(rv_symb, cpd_factory, expression)

    @add_match(Grounding(Implication(ProbabilisticPredicate, ...), ...))
    def probfact_set_grounding(self, grounding):
        """
        Represent a set of probabilistic facts with a Bernoulli node.
        """
        rv_symb = grounding.expression.consequent.body.functor
        probability_column = str2columnstr_constant(
            grounding.expression.consequent.probability.name
        )
        relation = grounding.relation
        cpd_factory = BernoulliCPDFactory(relation, probability_column)
        expression = grounding.expression
        self.add_random_variable(rv_symb, cpd_factory, expression)

    @add_match(ProbabilisticChoiceGrounding)
    def probchoice_grounding(self, grounding):
        """
        Represent a probabilistic choice as a n-ary choice random
        variable.
        """
        rv_symb = grounding.expression.predicate.functor
        probability_column = str2columnstr_constant()

    @add_match(Grounding(Implication, Constant[AbstractSet]))
    def intensional_rule_grounding(self, grounding):
        """
        Represent a deterministic intensional rule with an AND node.
        """
        rv_symb = grounding.expression.consequent.functor
        parent_rv_symbs = set(
            predicate.functor
            for predicate in extract_logic_predicates(
                grounding.expression.antecedent
            )
        )
        cpd_factory = AndCPDFactory(grounding.relation)
        expression = grounding.expression
        self.add_random_variable(
            rv_symb, cpd_factory, expression, parent_rv_symbs
        )

    def add_random_variable(
        self, rv_symb, cpd_factory, expression, parent_rv_symbs=None
    ):
        self.cpd_factories[rv_symb] = cpd_factory
        self.expressions[rv_symb] = expression
        if parent_rv_symbs is not None:
            self.edges[rv_symb] = parent_rv_symbs
