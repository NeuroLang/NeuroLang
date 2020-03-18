from collections.abc import Set, Mapping, MutableSet, MutableMapping, Iterable
from typing import AbstractSet, Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant, FunctionApplication
from .wrapped_collections import WrappedRelationalAlgebraSet
from ..utils import RelationalAlgebraFrozenSet, RelationalAlgebraSet
from ..type_system import infer_type, Unknown


class FrozenInstance:
    _set_type = RelationalAlgebraFrozenSet
    _rebv = ReplaceExpressionsByValues({})

    def __init__(self, elements=None):
        self.set_types = dict()
        self.cached_hash = None
        if elements is None:
            elements = dict()
        elif isinstance(elements, FrozenInstance):
            self.set_types = elements.set_types.copy()
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
            v, set_type = self._get_set_and_type(v)
            if len(v) > 0:
                elements[k] = self._set_type(v)
                self.set_types[k] = set_type
        return elements

    def _get_set_and_type(self, v):
        set_type = Unknown
        if isinstance(v, Constant[AbstractSet[Tuple]]):
            set_type = v.type.__args__[0]
            if isinstance(v.value, self._set_type):
                v = v.value
            else:
                v = self._rebv.walk(v)
        else:
            is_expression, set_type = self._infer_type(v, set_type)
            if is_expression and not isinstance(v, self._set_type):
                v = set(self._rebv.walk(e) for e in v)
        return v, set_type

    def _infer_type(self, v, set_type):
        is_expression = False
        for element in v:
            if isinstance(element, Constant):
                set_type = element.type
                is_expression = True
            elif (
                isinstance(element, tuple) and
                isinstance(element[0], Constant)
            ):
                set_type = Tuple[tuple(arg.type for arg in element)]
                is_expression = True
            else:
                set_type = infer_type(element)
            break
        return is_expression, set_type

    def _elements_from_iterable(self, iterable):
        result = dict()
        for predicate in iterable:
            symbol = predicate.functor
            if symbol not in result:
                result[symbol] = []
                self.set_types[symbol] = Tuple[
                    tuple(arg.type for arg in predicate.args)
                ]
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
        new_copy.set_types = self.set_types.copy()
        return new_copy

    def _create_view(self, class_):
        out = class_()
        out.elements = self.elements
        out.set_types = self.set_types
        return out

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.elements == other.elements
        else:
            return super().__eq__(other)


class FrozenMapInstance(FrozenInstance, Mapping):
    def _set_to_constant(self, set_, type_=Unknown):
        if len(set_) > 0:
            if type_ is Unknown:
                first = next(iter(set_))
                type_ = infer_type(first)
            return Constant[AbstractSet[type_]](
                WrappedRelationalAlgebraSet(set_),
                verify_type=False
            )
        else:
            return Constant[AbstractSet](
                WrappedRelationalAlgebraSet(),
                verify_type=False
            )

    def __getitem__(self, predicate_symbol):
        return self._set_to_constant(
            self.elements[predicate_symbol],
            type_=self.set_types[predicate_symbol]
        )

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def items(self):
        for k, v in self.elements.items():
            yield k, self._set_to_constant(
                v, type_=self.set_types[k]
            )

    def values(self):
        for k, v in self.elements.items():
            yield self._set_to_constant(v, type_=self.set_types[k])

    def as_set(self):
        return self._create_view(FrozenSetInstance)

    def as_map(self):
        return self

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in self.items()) + '}'


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
            types_ = self.set_types[predicate].__args__
            for t in tuples:
                arg = tuple(
                    Constant[type_](v, verify_type=False)
                    for type_, v in zip(types_, t)
                )

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
    _set_type = RelationalAlgebraSet

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
            self.set_types[predicate] = other.set_types[predicate]
        return self

    def _remove_predicate_symbol(self, predicate_symbol):
        del self.elements[predicate_symbol]
        del self.set_types[predicate_symbol]

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
        new_copy.set_types = self.set_types.copy()
        return new_copy


class MapInstance(Instance, FrozenMapInstance, MutableMapping):
    def __setitem__(self, predicate_symbol, value):
        self.elements[predicate_symbol] = self._set_type(value.value)
        self.set_types[predicate_symbol] = value.type.__args__[0]

    def __delitem__(self, predicate_symbol):
        self._remove_predicate_symbol(predicate_symbol)

    def as_set(self):
        return self._create_view(SetInstance)

    def as_map(self):
        return self


class SetInstance(Instance, FrozenSetInstance, MutableSet):
    def add(self, value):
        if value.functor not in self.elements:
            self.elements[value.functor] = self._set_type()
            self.set_types[value.functor] = Tuple[
                tuple(arg.type for arg in value.args)
            ]
        self.elements[value.functor].add(self._rebv.walk(value.args))

    def discard(self, value):
        functor = value.functor
        value = self._rebv.walk(value.args)
        self.elements[functor].discard(value)
        if len(self.elements[functor]) == 0:
            self._remove_predicate_symbol(functor)

    def as_set(self):
        return self

    def as_map(self):
        return self._create_view(MapInstance)
