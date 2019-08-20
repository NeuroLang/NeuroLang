from collections.abc import Set

from ..solver_datalog_naive import Fact


class SetInstance(Set):
    def __init__(self, elements):
        self.elements = elements

    def __contains__(self, fact):
        predicate = fact.consequent.functor
        tuple_values = fact.consequent.args
        return (
            predicate in self.elements and
            tuple_values in self.elements[predicate]
        )

    def __len__(self):
        return sum(len(v) for v in self.elements.values())

    def __iter__(self):
        for predicate, tuples in self.elements.items():
            for t in tuples:
                yield Fact(predicate(*t))

    def __hash__(self):
        return hash(frozenset(self))

    def as_map_instance(self):
        return MapInstance(self.elements)
