from collections.abc import Set, Mapping, MutableSet, MutableMapping, Iterable
from typing import AbstractSet, Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant
from ..exceptions import NeuroLangException
from .wrapped_collections import WrappedRelationalAlgebraSet
from ..utils import RelationalAlgebraFrozenSet, RelationalAlgebraSet
from ..type_system import infer_type


def predicate_iterable_as_dict(predicate_set, set_type=frozenset):
    result = dict()
    for predicate in predicate_set:
        symbol = predicate.functor
        if symbol not in result:
            result[symbol] = []
        result[symbol].append(tuple(predicate.args))
    for symbol in result:
        result[symbol] = set_type(result[symbol])
    return result


class FrozenInstance:
    _set_type = frozenset
    _rebv = ReplaceExpressionsByValues({})

    def __init__(self, elements=None):
        if elements is None:
            elements = dict()
        if isinstance(elements, Mapping):
            in_elements = elements
            elements = dict()
            for k, v in in_elements.items():
                if isinstance(v, Constant[AbstractSet[Tuple]]):
                    v = self._rebv.walk(v)
                if len(v) > 0:
                    elements[k] = self._set_type(v)
        elif isinstance(elements, Iterable):
            elements = predicate_iterable_as_dict(
                elements, set_type=self._set_type
            )
        self.elements = elements
        self.cached_hash = None

    def __hash__(self):
        if self.cached_hash is None:
            self.cached_hash = hash(tuple(zip(self.elements.items())))
        return self.cached_hash

    def __or__(self, other):
        new_elements = dict()
        for predicate in (self.elements.keys() | other.elements.keys()):
            new_elements[predicate] = self._set_type(
                self.elements.get(predicate, self._set_type()) |
                other.elements.get(predicate, self._set_type())
            )

        return type(self)(new_elements)

    def __sub__(self, other):
        new_elements = dict()
        for predicate, tuple_set in self.elements.items():
            new_set = tuple_set - other.elements.get(predicate, set())
            if len(new_set) > 0:
                new_elements[predicate] = new_set
        return type(self)(new_elements)

    def __and__(self, other):
        new_elements = dict()
        for predicate in (self.elements.keys() & other.elements.keys()):
            new_set = self.elements[predicate] & other.elements[predicate]
            if len(new_set) > 0:
                new_elements[predicate] = new_set
        return type(self)(new_elements)

    def copy(self):
        new_copy = type(self)()
        new_copy.elements = self.elements.copy()
        new_copy.hash = self.cached_hash
        return new_copy


class FrozenMapInstance(FrozenInstance, Mapping):
    def _set_to_constant(self, set_):
        if len(set_) > 0:
            first = next(iter(set_))
            first_type = infer_type(first)
            return Constant[AbstractSet[first_type]](
                WrappedRelationalAlgebraSet(set_),
                verify_type=False
            )
        else:
            return Constant[AbstractSet](
                WrappedRelationalAlgebraSet(),
                verify_type=False
            )

    def __getitem__(self, predicate):
        return self._set_to_constant(self.elements[predicate])

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def items(self):
        for k, v in self.elements.items():
            yield k, self._set_to_constant(v)

    def values(self):
        for v in self.values():
            yield self._set_to_constant(v)

    def as_set(self):
        out = FrozenSetInstance()
        out.elements = self.elements
        return out

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.elements == other.elements
        else:
            return super().__eq__(other)


class FrozenSetInstance(FrozenInstance, Set):
    def __contains__(self, predicate):
        functor = predicate.functor
        tuple_values = predicate.args
        return (
            functor in self.elements and
            tuple_values in self.elements[functor]
        )

    def __len__(self):
        return sum(len(v) for v in self.elements.values())

    def __iter__(self):
        for predicate, tuples in self.elements.items():
            for t in tuples:
                yield predicate(*t)

    def as_map(self):
        out = FrozenMapInstance()
        out.elements = self.elements
        return out

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.elements == other.elements
        else:
            return super().__eq__(other)


class Instance(FrozenInstance):
    _set_type = RelationalAlgebraSet

    def __init__(self, elements=None):
        super().__init__(elements=elements)

    def __hash__(self):
        raise TypeError('Instance objects are mutable and cannot be hashed')

    def __ior__(self, other):
        for predicate in (self.elements.keys() & other.elements.keys()):
            self.elements[predicate] |= other.elements[predicate]
        for predicate in (other.elements.keys() - self.elements.keys()):
            self.elements[predicate] = other.elements[predicate]
        return self

    def __isub__(self, other):
        for predicate in (self.elements.keys() & other.elements.keys()):
            self.elements[predicate] -= other.elements[predicate]
            if len(self.elements[predicate]) == 0:
                del self.elements[predicate]
        return self

    def __iand__(self, other):
        subs = self.elements.keys() - other.elements.keys()
        for predicate in subs:
            del self.elements[predicate]
        for predicate in self.elements:
            self.elements[predicate] &= other.elements[predicate]
            if len(self.elements[predicate]) == 0:
                del self.elements[predicate]
        return self

    def copy(self):
        new_copy = type(self)()
        new_copy.elements = self.elements.copy()
        return new_copy


class MapInstance(Instance, FrozenMapInstance, MutableMapping):
    def __setitem__(self, predicate, value):
        self.elements[predicate] = self._set_type(value.value)

    def __delitem__(self, predicate):
        del self.elements[predicate]

    def as_set(self):
        out = SetInstance()
        out.elements = self.elements
        return out


class SetInstance(Instance, FrozenSetInstance, MutableSet):
    _rebv = ReplaceExpressionsByValues(dict())

    def add(self, fact):
        if fact.functor not in self.elements:
            self.elements[fact.functor] = self._set_type
        self.elements[fact.functor].add(self._rebv(fact.args))

    def discard(self, fact):
        value = self._rebv(fact.args)
        self.elements[fact.functor].discard(value)
        if len(self.elements[fact.functor]) == 0:
            del self.elements[fact.functor]

    def as_map(self):
        out = MapInstance()
        out.elements = self.elements
        return out
