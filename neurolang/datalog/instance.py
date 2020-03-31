from collections.abc import Iterable, Mapping, MutableMapping, MutableSet, Set
from typing import AbstractSet, Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant, FunctionApplication
from ..type_system import Unknown
from .wrapped_collections import (WrappedRelationalAlgebraFrozenSet,
                                  WrappedRelationalAlgebraSet)


class FrozenInstance:
    _set_type = WrappedRelationalAlgebraFrozenSet
    _rebv = ReplaceExpressionsByValues({})

    def __init__(self, elements=None):
        self.cached_hash = None
        if elements is None:
            elements = dict()
        elif isinstance(elements, FrozenInstance):
            self.cached_hash = elements.cached_hash
            elements = elements.elements.copy()
        elif isinstance(elements, Mapping):
            elements = self._elements_from_mapping(elements)
        elif isinstance(elements, Iterable):
            elements = self._elements_from_iterable(elements)
        self.elements = elements

    def _elements_from_mapping(self, elements):
        in_elements = elements
        elements = dict()
        for k, v in in_elements.items():
            if not isinstance(v, self._set_type):
                v = self._get_set_and_type(v)
                if len(v) > 0:
                    v = self._set_type(v)
            if len(v) > 0:
                elements[k] = v
        return elements

    def _get_set_and_type(self, v):
        if isinstance(v, Constant[AbstractSet[Tuple]]):
            v = v.value
        else:
            is_expression = self._is_expression_iterable(v)
            if is_expression and not isinstance(v, self._set_type):
                v = set(self._rebv.walk(e) for e in v)
        return v

    def _is_expression_iterable(self, v):
        is_expression = False
        for element in v:
            if isinstance(element, Constant):
                is_expression = True
            elif (
                isinstance(element, tuple) and
                isinstance(element[0], Constant)
            ):
                is_expression = True
            break
        return is_expression

    def _elements_from_iterable(self, iterable):
        result = dict()
        for predicate in iterable:
            symbol = predicate.functor
            if symbol not in result:
                result[symbol] = []
            result[symbol].append(self._rebv.walk(predicate.args))

        for symbol in result:
            result[symbol] = self._set_type(result[symbol])

        return result

    def __hash__(self):
        if self.cached_hash is None:
            self.cached_hash = hash(tuple(zip(self.elements.items())))
        return self.cached_hash

    def __or__(self, other):
        if not isinstance(other, Instance):
            return super().__or__(other)
        new_elements = dict()
        for predicate in (self.elements.keys() | other.elements.keys()):
            new_elements[predicate] = self._set_type(
                self.elements.get(predicate, self._set_type()) |
                other.elements.get(predicate, self._set_type())
            )

        return type(self)(new_elements)

    def __sub__(self, other):
        if not isinstance(other, Instance):
            return super().__sub__(other)

        new_elements = dict()
        for predicate, tuple_set in self.elements.items():
            new_set = tuple_set - other.elements.get(predicate, set())
            if len(new_set) > 0:
                new_elements[predicate] = new_set
        return type(self)(new_elements)

    def __and__(self, other):
        if not isinstance(other, Instance):
            return super().__and__(other)

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

    def _create_view(self, class_):
        out = class_()
        out.elements = self.elements
        return out

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.elements == other.elements
        else:
            return super().__eq__(other)


class FrozenMapInstance(FrozenInstance, Mapping):
    def _set_to_constant(self, set_, type_=Unknown):
        if type_ is Unknown:
            type_ = set_.row_type
        return Constant[AbstractSet[type_]](
            set_,
            verify_type=False
        )

    def __getitem__(self, predicate_symbol):
        set_ = self.elements[predicate_symbol]
        return self._set_to_constant(
            set_,
            type_=set_.row_type
        )

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def items(self):
        for k, v in self.elements.items():
            yield k, self._set_to_constant(
                v, type_=v.row_type
            )

    def values(self):
        for k, v in self.elements.items():
            yield self._set_to_constant(v, type_=v.row_type)

    def as_set(self):
        return self._create_view(FrozenSetInstance)

    def as_map(self):
        return self

    def __hash__(self):
        return super().__hash__()


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
                arg = t.value
                yield FunctionApplication(
                    predicate, arg
                )

    def as_set(self):
        return self

    def as_map(self):
        return self._create_view(FrozenMapInstance)

    def __hash__(self):
        return super().__hash__()


class Instance(FrozenInstance):
    _set_type = WrappedRelationalAlgebraSet

    def __init__(self, elements=None):
        super().__init__(elements=elements)

    def __hash__(self):
        raise TypeError('Instance objects are mutable and cannot be hashed')

    def __ior__(self, other):
        if not isinstance(other, Instance):
            return super().__ior__(other)

        for predicate in (self.elements.keys() & other.elements.keys()):
            self.elements[predicate] |= other.elements[predicate]
        for predicate in (other.elements.keys() - self.elements.keys()):
            self.elements[predicate] = other.elements[predicate]
        return self

    def _remove_predicate_symbol(self, predicate_symbol):
        del self.elements[predicate_symbol]

    def __isub__(self, other):
        if not isinstance(other, Instance):
            return super().__isub__(other)
        for predicate in (self.elements.keys() & other.elements.keys()):
            self.elements[predicate] -= other.elements[predicate]
            if len(self.elements[predicate]) == 0:
                self._remove_predicate_symbol(predicate)
        return self

    def __iand__(self, other):
        if isinstance(other, Instance):
            return super().__iand__(other)
        subs = self.elements.keys() - other.elements.keys()
        for predicate in subs:
            self._remove_predicate_symbol(predicate)
        for predicate in self.elements:
            self.elements[predicate] &= other.elements[predicate]
            if len(self.elements[predicate]) == 0:
                self._remove_predicate_symbol(predicate)
        return self

    def copy(self):
        new_copy = type(self)()
        new_copy.elements = self.elements.copy()
        return new_copy


class MapInstance(Instance, FrozenMapInstance, MutableMapping):
    def __setitem__(self, predicate_symbol, value):
        set_ = self._set_type(value.value)
        self.elements[predicate_symbol] = set_

    def __delitem__(self, predicate_symbol):
        self._remove_predicate_symbol(predicate_symbol)

    def as_set(self):
        return self._create_view(SetInstance)

    def as_map(self):
        return self


class SetInstance(Instance, FrozenSetInstance, MutableSet):
    def add(self, value):
        functor = value.functor
        args = value.args
        if functor not in self.elements:
            self.elements[functor] = self._set_type()
        self.elements[value.functor].add(args)

    def discard(self, value):
        functor = value.functor
        value = value.args
        self.elements[functor].discard(value)
        if len(self.elements[functor]) == 0:
            self._remove_predicate_symbol(functor)

    def as_set(self):
        return self

    def as_map(self):
        return self._create_view(MapInstance)
