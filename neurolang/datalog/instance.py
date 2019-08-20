from functools import lru_cache
from collections.abc import Set, Mapping, MutableSet

from ..solver_datalog_naive import Fact
from ..exceptions import NeuroLangException


class MapInstance(Mapping):
    def __init__(self, elements):
        if any(
            not isinstance(tuple_set, Set) or
            isinstance(tuple_set, MutableSet)
            for tuple_set in elements.values()
        ):
            raise NeuroLangException('Expected immutable tuple sets')
        self.elements = elements
        self.cached_hash = None

    def __getitem__(self, predicate):
        return self.elements[predicate]

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __hash__(self):
        if self.cached_hash is None:
            self.cached_hash = hash(tuple(zip(self.elements.items())))
        return self.cached_hash


class SetInstance(Set):
    def __init__(self, elements):
        self.elements = elements
        self.cached_hash = None

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
        if self.cached_hash is None:
            self.cached_hash = hash(tuple(zip(self.elements.items())))
        return self.cached_hash
