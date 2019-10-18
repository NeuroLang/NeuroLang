import numpy as np

from ..exceptions import NeuroLangException


class InvalidProbabilityDistribution(NeuroLangException):
    pass


class Distribution:
    def probability(self, value):
        raise NeuroLangException("Not implemented for abstract class")

    @property
    def support(self):
        raise NeuroLangException("Not implemented for abstract class")

    def expectation(self, fun):
        raise NeuroLangException("Not implemented for abstract class")


class DiscreteDistribution(Distribution):
    pass


class TableDistribution(DiscreteDistribution):
    def __init__(self, table):
        self.table = table

    def probability(self, value):
        return self.table.get(value, 0.0)

    @property
    def support(self):
        return frozenset(self.table)

    def expectation(self, fun):
        return sum(fun(val) * prob for val, prob in self.table.items())

    def conditioned_on(self, condition):
        """
        Compute a new distribution for random variable Y(X) such that
        P(Y(X)) = P(X | C(X) = True), where C is a condition function.
        """
        new_table = {
            value: prob
            for value, prob in self.table.items()
            if condition(value)
        }
        sum_prob = sum(prob for prob in new_table.values())
        for value, prob in new_table.items():
            new_table[value] = prob / sum_prob
        return TableDistribution(new_table)

    def __repr__(self):
        return "TableDistribution[\n{}\n]".format(
            "\n".join(
                [f"{value}: {prob}" for value, prob in self.table.items()]
            )
        )

    def __eq__(self, other):
        if not isinstance(other, TableDistribution):
            raise NeuroLangException(
                "Can only compare with other TableDistribution"
            )
        return set(other.table.keys()) == set(self.table.keys()) and not any(
            not np.isclose(other.table[k], self.table[k]) for k in self.table
        )
