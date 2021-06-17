from collections.abc import Iterable, Mapping, MutableMapping, MutableSet, Set
from typing import AbstractSet, Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant, FunctionApplication
from ..type_system import Unknown, get_args
from ..utils.relational_algebra_set import RelationalAlgebraFrozenSet
from .wrapped_collections import (WrappedRelationalAlgebraFrozenSet,
                                  WrappedRelationalAlgebraSet)


class FrozenInstance:
    _set_type = WrappedRelationalAlgebraFrozenSet
    _rebv = ReplaceExpressionsByValues({})

    def __init__(self, elements=None):
        self.cached_hash = None
        self._elements_clean = False
        self._no_sets = None
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

    def _clean_elements(self):
        if self._elements_clean:
            return
        to_clean = []
        for k, v in self.elements.items():
            if v.is_empty():
                to_clean.append(k)
        for k in to_clean:
            del self.elements[k]
        self._elements_clean = True
        self._no_sets = len(self.elements) == 0

    def _reset_cached(self):
        self._elements_clean = False
        self._no_sets = None

    def _elements_from_mapping(self, elements):
        in_elements = elements
        elements = dict()
        for k, v in in_elements.items():
            v = self._get_set(v)
            elements[k] = v
        return elements

    def _get_set(self, v):
        v_type = Unknown
        if not isinstance(v, self._set_type):
            v, v_type = self._get_set_and_type(v)
            v = self._set_type(
                v, row_type=v_type, verify_row_type=False
            )
        return v

    def _set_not_empty(self, v):
        if isinstance(v, RelationalAlgebraFrozenSet):
            return not v.is_empty()
        else:
            return len(v) > 0

    def _get_set_and_type(self, v):
        row_type = Unknown
        if isinstance(v, Constant[AbstractSet[Tuple]]):
            row_type = get_args(v.type)[0]
            if isinstance(v.value, self._set_type):
                v = v.value
            else:
                v = self._rebv.walk(v)
        return v, row_type

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
        self._clean_elements()
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
        res = type(self)(new_elements)
        if self._no_sets is not None and other._no_sets is not None:
            res._no_sets = self._no_sets | other._no_sets
        elif self._no_sets is False or other._no_sets is False:
            res._no_sets = False
        return res

    def __sub__(self, other):
        if not isinstance(other, Instance):
            return super().__sub__(other)
        new_elements = dict()
        for predicate, tuple_set in self.elements.items():
            other_tuple_set = other.elements.get(predicate)
            if other_tuple_set is not None:
                new_set = tuple_set - other_tuple_set
            else:
                new_set = tuple_set
            # if not new_set.is_empty():
            new_elements[predicate] = new_set
        res = type(self)(new_elements)
        return res

    def __and__(self, other):
        if not isinstance(other, Instance):
            return super().__and__(other)

        new_elements = dict()
        for predicate in (self.elements.keys() & other.elements.keys()):
            new_set = self.elements[predicate] & other.elements[predicate]
            new_elements[predicate] = new_set
        res = type(self)(new_elements)
        return res

    def copy(self):
        new_copy = type(self)()
        new_copy.elements = self.elements.copy()
        new_copy._elements_clean = self._elements_clean
        new_copy._no_sets = self._no_sets
        new_copy.hash = self.cached_hash
        return new_copy

    def _create_view(self, class_):
        out = class_()
        out.elements = self.elements
        out._elements_clean = self._elements_clean
        out._no_sets = self._no_sets
        return out

    def __eq__(self, other):
        self._clean_elements()
        if isinstance(other, Instance):
            other._clean_elements()
            return self.elements == other.elements
        else:
            return super().__eq__(other)

    def is_empty(self):
        if self._no_sets is None:
            to_clean = []
            self._no_sets = True
            for k, v in self.elements.items():
                if not v.is_empty():
                    self._no_sets = False
                    break
                to_clean.append(k)
            for k in to_clean:
                del self.elements[k]
        return self._no_sets

    def __repr__(self):
        return repr(self.elements)


class FrozenMapInstance(FrozenInstance, Mapping):
    def _set_to_constant(self, set_, type_=Unknown):
        if type_ is Unknown:
            type_ = set_.row_type
        return Constant[AbstractSet[type_]](
            set_,
            verify_type=False
        )

    def __getitem__(self, predicate_symbol):
        self._clean_elements()
        set_ = self.elements[predicate_symbol]
        if set_.is_empty():
            del self.elements[predicate_symbol]
            raise KeyError()
        type_ = set_.row_type
        return self._set_to_constant(set_, type_=type_)

    def __iter__(self):
        self._clean_elements()
        return iter(self.elements)

    def __len__(self):
        self._clean_elements()
        return len(self.elements)

    def items(self):
        self._clean_elements()
        for k, v in self.elements.items():
            row_type = v.row_type
            yield k, self._set_to_constant(
                v, type_=row_type
            )

    def values(self):
        self._clean_elements()
        for k, v in self.elements.items():
            row_type = v.row_type
            yield self._set_to_constant(v, type_=row_type)

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
        self._reset_cached()
        del self.elements[predicate_symbol]

    def __isub__(self, other):
        self._elements_clean = False
        if not isinstance(other, Instance):
            return super().__isub__(other)
        for predicate in (self.elements.keys() & other.elements.keys()):
            self.elements[predicate] -= other.elements[predicate]
            if self.elements[predicate].is_empty():
                self._remove_predicate_symbol(predicate)
        return self

    def __iand__(self, other):
        self._elements_clean = False
        if isinstance(other, Instance):
            return super().__iand__(other)
        subs = self.elements.keys() - other.elements.keys()
        for predicate in subs:
            self._remove_predicate_symbol(predicate)
        for predicate in self.elements:
            self.elements[predicate] &= other.elements[predicate]
            if self.elements[predicate].is_empty():
                self._remove_predicate_symbol(predicate)
        return self

    def copy(self):
        new_copy = type(self)()
        new_copy.elements = self.elements.copy()
        new_copy._elements_clean = self._elements_clean
        new_copy._no_sets = self._no_sets
        return new_copy


class MapInstance(Instance, FrozenMapInstance, MutableMapping):
    def __setitem__(self, predicate_symbol, value):
        set_ = self._set_type(value.value)
        self._reset_cached()
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
        if self.elements[functor].is_empty():
            self._remove_predicate_symbol(functor)

    def as_set(self):
        return self

    def as_map(self):
        return self._create_view(MapInstance)
