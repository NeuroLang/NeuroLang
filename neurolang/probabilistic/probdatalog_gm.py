from typing import Mapping

from ..expression_walker import PatternWalker
from ..expression_pattern_matching import add_match
from ..expressions import Definition, Symbol, Constant
from .probdatalog import Grounding, is_probabilistic_fact
from .probdatalog_bn import BayesianNetwork


class Plate(Definition):
    def __init__(self, bayesian_network, var_domains):
        self.bayesian_network = bayesian_network
        self.var_domains = var_domains


def vectorised_pfact_cpd_factory(grounding):
    pass


def vectorised_deterministic_or_cpd_factory(parent_values):
    pass


def construct_probfact_bn_from_grounding(grounding):
    pfact_pred = grounding.expression.consequent.body
    choice_pred = Symbol[int].fresh()(pfact_pred.args)
    rv_to_cpd_factory = Constant[Mapping](
        {
            choice_pred: vectorised_pfact_cpd_factory(grounding),
            pfact_pred: vectorised_deterministic_or_cpd_factory,
        }
    )
    edges = Constant[Mapping]({pfact_pred: choice_pred})
    return BayesianNetwork(edges, rv_to_cpd_factory)


class TranslateGroundedProbDatalogToGraphicalModel(PatternWalker):
    @add_match(Grounding, lambda exp: is_probabilistic_fact(exp.expression))
    def grounding_probabilistic_fact(self, grounding):
        return Plate(
            construct_probfact_bn_from_grounding(grounding),
            grounding.name_domains,
        )
