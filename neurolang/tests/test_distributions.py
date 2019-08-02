from ..expressions import Constant
from ..probabilistic.distributions import TableDistribution

C_ = Constant


def test_table_distribution():
    table = {
        C_('cat'): C_(0.2),
        C_('dog'): C_(0.8),
    }
    distrib = TableDistribution(table)
    assert distrib.probability(C_('cat')) == C_(0.2)
    assert distrib.probability(C_('dog')) == C_(0.8)
    assert distrib.support == frozenset({C_('cat'), C_('dog')})
