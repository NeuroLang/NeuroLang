from typing import AbstractSet

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
    Projection,
    RelationalAlgebraSolver,
    str2columnstr_constant,
)
from ..expressions import (
    Grounding,
    ProbabilisticChoiceGrounding,
    ProbabilisticPredicate,
)
from .grounding import get_grounding_pred_symb, topological_sort_groundings


class GraphicalModel(Definition):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_node(self, node_symbol):
        nodes_as_dict = dict(self.nodes)
        if node_symbol not in nodes_as_dict:
            raise KeyError(f"Node {node_symbol} not found")
        return nodes_as_dict[node_symbol]

    def get_parent_node_symbols(self, node_symbol):
        return set(
            edge_parent_node_symb
            for edge_node_symb, edge_parent_node_symb in self.edges
            if edge_node_symb == node_symbol
        )


class PlateNode(Definition):
    def __init__(self, node_symbol, expression, relation):
        self.node_symbol = node_symbol
        self.expression = expression
        self.relation = relation

    def __repr__(self):
        return f"{self.node_symbol} ~ {self.__class__.__name__}"


class ProbabilisticPlateNode(PlateNode):
    """
    Object used to represent non-deterministic nodes.represent
    non-deterministic nodes

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

    where each row represents one random variable P(x) and where
    the column _p_ encodes the Bernoulli parameter of the
    distribution of P(x).

    Attributes
    ----------
    relation: Constant[AbstractSet]
        A relation that encodes all the variables for which a
        conditional probability distribution will be produced.
    probability_column: Constant[ColumnStr]
        The column that contains the Bernoulli parameter associated
        to each random variable.

    """

    def __init__(self, node_symbol, expression, relation, probability_column):
        super().__init__(node_symbol, expression, relation)
        self.probability_column = probability_column


class BernoulliPlateNode(ProbabilisticPlateNode):
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

    where each row represents one random variable P(x) and where
    the column _p_ encodes the Bernoulli parameter of the
    distribution of P(x).

    Attributes
    ----------
    relation: Constant[AbstractSet]
        A relation that encodes all the variables for which a
        conditional probability distribution will be produced.
    probability_column: Constant[ColumnStr]
        The column that contains the Bernoulli parameter associated
        to each random variable.

    """


class AndPlateNode(PlateNode):
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


class NaryChoicePlateNode(ProbabilisticPlateNode):
    """
    Object used to represent the distribution of a single
    n-ary choice random variable.

    The relation does not represent multiple random variables
    but the different predicates that can be chosen by the rule,
    alongside their associated probability.

    Given a CP-Event of the form

        P(a_1) : p_1  v  ...  v  P(a_n) : p_n  <-  T

    where p_1, ..., p_n are the probability labels attached to each
    head predicate P(a_1), ..., P(a_n) -- respectively,

    the conditional probability distribution of the choice random
    variable (which is its marginal because there is no parent) is
    defined by the relation

        _p_ | x
        ====|====
        p_1 | a_1
        p_2 | a_2
        ... | ...
        p_n | a_n

    where each row represents a possible value of the random variable
    with its probability.

    """


class NaryChoiceResultPlateNode(PlateNode):
    pass


def is_extensional_grounding(grounding):
    """TODO: represent extensional grounding with a fact instead?"""
    return (
        isinstance(grounding.expression, Implication)
        and isinstance(grounding.expression.consequent, FunctionApplication)
        and grounding.expression.antecedent == Constant[bool](True)
    )


class CPLogicGroundingToGraphicalModelTranslator(PatternWalker):
    def __init__(self):
        self.edges = []
        self.nodes = []

    @add_match(ExpressionBlock)
    def block_of_groundings(self, block):
        if any(not isinstance(exp, Grounding) for exp in block.expressions):
            raise ValueError("Expected block of groundings")
        for grounding in topological_sort_groundings(block.expressions):
            self.walk(grounding)
        return GraphicalModel(tuple(self.nodes), tuple(self.edges))

    @add_match(Grounding, is_extensional_grounding)
    def extensional_set_grounding(self, grounding):
        """
        Represent a set of ground facts as a Bernoulli node with
        all probabilities set to 1.0.

        """
        expression = grounding.expression
        node_symbol = get_grounding_pred_symb(expression)
        probability_column = str2columnstr_constant(Symbol.fresh().name)
        relation = ConcatenateConstantColumn(
            grounding.relation, probability_column, Constant[float](1.0)
        )
        relation = RelationalAlgebraSolver().walk(relation)
        node = BernoulliPlateNode(
            node_symbol, expression, relation, probability_column
        )
        self.add_plate_node(node_symbol, node)

    @add_match(ProbabilisticChoiceGrounding)
    def probabilistic_choice_grounding(self, grounding):
        """
        Represent a probabilistic choice as a n-ary choice node.
        """
        expression = grounding.expression
        choice_node_symb = Symbol.fresh()
        probability_column = str2columnstr_constant(
            grounding.expression.consequent.probability.name
        )
        relation = grounding.relation
        # add a n-ary choice random variable
        choice_node = NaryChoicePlateNode(
            choice_node_symb, expression, relation, probability_column
        )
        self.add_plate_node(choice_node_symb, choice_node)
        # remove the probability column as it is not neeeded to represent the
        # CPD factories of boolean random variables whose value is
        # deterministically determined by the value of their parent choice
        # variable
        relation = RelationalAlgebraSolver().walk(
            Projection(
                relation,
                tuple(
                    str2columnstr_constant(col)
                    for col in relation.value.columns
                    if col != probability_column.value
                ),
            )
        )
        node_symbol = get_grounding_pred_symb(expression)
        node = NaryChoiceResultPlateNode(node_symbol, expression, relation)
        self.add_plate_node(
            node_symbol, node, parent_node_symbols={choice_node_symb}
        )

    @add_match(Grounding(Implication(ProbabilisticPredicate, ...), ...))
    def probfact_set_grounding(self, grounding):
        """
        Represent a set of probabilistic facts with a Bernoulli node.
        """
        expression = grounding.expression
        node_symbol = get_grounding_pred_symb(expression)
        probability_column = str2columnstr_constant(
            grounding.expression.consequent.probability.name
        )
        relation = grounding.relation
        node = BernoulliPlateNode(
            node_symbol, expression, relation, probability_column
        )
        self.add_plate_node(node_symbol, node)

    @add_match(Grounding(Implication, Constant[AbstractSet]))
    def intensional_rule_grounding(self, grounding):
        """
        Represent a deterministic intensional rule with an AND node.
        """
        expression = grounding.expression
        node_symbol = get_grounding_pred_symb(expression)
        parent_node_symbols = set(
            predicate.functor
            for predicate in extract_logic_predicates(
                grounding.expression.antecedent
            )
        )
        node = AndPlateNode(node_symbol, expression, grounding.relation)
        self.add_plate_node(node_symbol, node, parent_node_symbols)

    def add_plate_node(self, node_symbol, node, parent_node_symbols=None):
        self.nodes.append((node_symbol, node))
        if parent_node_symbols is not None:
            self.edges += [
                (node_symbol, parent_node_symb)
                for parent_node_symb in parent_node_symbols
            ]
