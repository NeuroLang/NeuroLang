import numpy as np

from ..exceptions import NeuroLangException
from ..expressions import Definition, Constant


class Distribution(Definition):
    def __init__(self, **parameters):
        pass

    def probability(self, value):
        raise NeuroLangException('Not implemented for abstract class')

    @property
    def support(self):
        raise NeuroLangException('Not implemented for abstract class')


class DiscreteDistribution(Distribution):
    pass


class TableDistribution(DiscreteDistribution):
    def __init__(self, table, **parameters):
        super().__init__(**parameters)
        if not np.isclose(sum(v.value for v in table.values()), 1.0):
            raise NeuroLangException('Table probabilities do not sum to 1')
        self.table = table

    def probability(self, value):
        return self.table.get(value, Constant[float](0.))

    @property
    def support(self):
        return frozenset(self.table.keys())
