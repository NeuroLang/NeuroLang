from collections import namedtuple
from inspect import isclass
from itertools import tee
from functools import lru_cache
from typing import Tuple

import numpy as np

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant
from ..type_system import (Unknown, get_args, infer_type,
                           unify_types)
from ..utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet,
    RelationalAlgebraSet)

REBV = ReplaceExpressionsByValues(dict())


class WrappedTypeMap:
    row_maps = {
        np.integer: int,
        np.float: float
    }

    @lru_cache(maxsize=256)
    def backend_2_python(self, value):
        for k, v in self.row_maps.items():
            if (
                (isclass(value) and issubclass(value, k)) or
                value == k
            ):
                return v
        return value


TYPEMAP = WrappedTypeMap()


class WrappedRelationalAlgebraSetBaseMixin:
    def __init__(
        self, iterable=None, row_type=Unknown, verify_row_type=True, **kwargs
    ):
        iterable = WrappedRelationalAlgebraSetBaseMixin._get_init_iterable(
            iterable
        )
        super().__init__(iterable=iterable, **kwargs)
        self._set_row_type(iterable, row_type, verify_row_type)

    def _set_row_type(self, iterable, row_type, verify_row_type):
        if row_type is not Unknown:
            if verify_row_type:
                raise NotImplementedError()
            self._row_type = row_type
        elif isinstance(iterable, WrappedRelationalAlgebraSetBaseMixin):
            self._row_type = iterable._row_type
        else:
            self._row_type = None

    @staticmethod
    def _get_init_iterable(iterable):
        if iterable is not None:
            if isinstance(
                iterable,
                (
                    WrappedRelationalAlgebraSetBaseMixin,
                    RelationalAlgebraFrozenSet
                )
            ):
                iterable = iterable
            elif hasattr(iterable, '__getitem__'):
                iterable = (
                    WrappedRelationalAlgebraSetBaseMixin
                    ._obtain_value_collection(
                        iterable
                    )
                )
            else:
                iterable = (
                    WrappedRelationalAlgebraSetBaseMixin
                    ._obtain_value_iterable(
                        iterable
                    )
                )
        return iterable

    def __contains__(self, element):
        element = REBV.walk(element)
        return super().__contains__(element)

    def _operator_wrapped(self, op, other):
        other_is_wras = isinstance(other, WrappedRelationalAlgebraSetBaseMixin)
        if other_is_wras:
            other = other.unwrap()
        else:
            other = {el for el in self._obtain_value_iterable(other)}
        operator = getattr(self.unwrap(), op)
        res = operator(other)
        if isinstance(res, WrappedRelationalAlgebraSetBaseMixin):
            res._row_type = self._get_new_row_type(other, other_is_wras)
        elif isinstance(res, RelationalAlgebraFrozenSet):
            res = type(self)(iterable=res)
        return res

    def _get_new_row_type(self, other, other_is_wras):
        row_type = self._row_type
        if other_is_wras:
            if row_type is not None and other._row_type is not None:
                row_type = unify_types(row_type, other._row_type)
            elif row_type is None:
                row_type = other._row_type
        return row_type

    @staticmethod
    def _obtain_value_iterable(iterable):
        it1, it2 = tee(iterable)
        iterator_of_constants = False
        for val in it1:
            iterator_of_constants = (
                WrappedRelationalAlgebraSetBaseMixin.
                is_constant_tuple_or_tuple_of_constants(val)
            )
            break
        if not iterator_of_constants:
            iterator = it2
        else:
            iterator = (REBV.walk(e) for e in it2)
        for e in iterator:
            yield e

    @staticmethod
    def _obtain_value_collection(iterable):
        if len(iterable) == 0:
            return iterable

        val = iterable[0]
        collection_of_constants = (
            WrappedRelationalAlgebraSetBaseMixin.
            is_constant_tuple_or_tuple_of_constants(val)
        )
        if not collection_of_constants:
            return iterable
        else:
            return (REBV.walk(e) for e in iterable)

    @staticmethod
    def is_constant_tuple_or_tuple_of_constants(val):
        return (
            isinstance(val, Constant[Tuple]) or
            (
                isinstance(val, tuple) and (len(val) > 0)
                and isinstance(val[0], Constant)
            )
        )

    def __eq__(self, other):
        return self._operator_wrapped('__eq__', other)

    def __ne__(self, other):
        return self._operator_wrapped('__ne__', other)

    def __lt__(self, other):
        return self._operator_wrapped('__lt__', other)

    def __gt__(self, other):
        return self._operator_wrapped('__gt__', other)

    def __le__(self, other):
        return self._operator_wrapped('__le__', other)

    def __ge__(self, other):
        return self._operator_wrapped('__ge__', other)

    def __and__(self, other):
        return self._operator_wrapped('__and__', other)

    def __or__(self, other):
        return self._operator_wrapped('__or__', other)

    def __sub__(self, other):
        return self._operator_wrapped('__sub__', other)

    def __hash__(self):
        return super().__hash__()

    def unwrapped_iter(self):
        return super().__iter__()

    @property
    def row_type(self):
        if self._row_type is None:
            if self.arity > 0 and not self.is_empty():
                self._row_type = Tuple[tuple(
                    TYPEMAP.backend_2_python(t)
                    for t in get_args(infer_type(super().fetch_one()))
                )]
            else:
                self._row_type = Tuple

        return self._row_type


class WrappedRelationalAlgebraFrozenSetMixin(
    WrappedRelationalAlgebraSetBaseMixin
):
    def __iter__(self):
        type_ = self.row_type
        element_types = get_args(type_)

        for t in super().__iter__():
            yield Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(element_types, t)
                ),
                verify_type=False
            )


class WrappedRelationalAlgebraSetMixin(
    WrappedRelationalAlgebraSetBaseMixin
):
    def __iter__(self):
        type_ = self.row_type
        element_types = get_args(type_)

        for t in super().__iter__():
            yield Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(element_types, t)
                ),
                verify_type=False
            )

    def add(self, value):
        return super().add(REBV.walk(value))

    def discard(self, value):
        return super().discard(REBV.walk(value))


def named_tuple_as_dict(*args, **kwargs):
    nt = namedtuple(*args, **kwargs)
    nt.get = lambda self, key, default=None: self._asdict().get(key, default)
    nt.keys = lambda self: self._asdict().keys()
    nt.items = lambda self: self._asdict().items()
    nt.values = lambda self: self._asdict().values()
    nt.__getitem__ = lambda self, key: self._asdict()[key]
    return nt


class WrappedNamedRelationalAlgebraFrozenSetMixin(
    WrappedRelationalAlgebraSetBaseMixin
):
    def __init__(
        self, columns=None, iterable=None,
        row_type=Unknown, verify_row_type=True, **kwargs
    ):
        iterable = WrappedRelationalAlgebraSetBaseMixin._get_init_iterable(
            iterable
        )
        if columns is None and iterable is not None:
            columns = iterable.columns
        super().__init__(columns=columns, iterable=iterable, **kwargs)
        self._set_row_type(iterable, row_type, verify_row_type)
        self.named_tuple_type = None

    @property
    def row_type(self):
        if self._row_type is None:
            if (self.arity > 0 and not self.is_empty()):
                element = super().fetch_one()
                self._row_type = Tuple[tuple(
                    Constant(getattr(element, c)).type
                    for c in self.columns
                )]
            else:
                self._row_type = Tuple

        return self._row_type

    def __iter__(self):
        if self.arity > 0:
            if self.named_tuple_type is None:
                self.named_tuple_type = named_tuple_as_dict(
                    'tuple', self.columns
                )

            row_types = {
                c: t
                for c, t in zip(self.columns, get_args(self.row_type))
            }
            for row in super().__iter__():
                nt = self.named_tuple_type(**{
                    f: Constant[row_types[f]](
                        v, verify_type=False
                    )
                    for f, v in zip(row._fields, row)
                })
                yield nt
        else:
            for _ in range(len(self)):
                yield dict()


class WrappedRelationalAlgebraFrozenSet(
    WrappedRelationalAlgebraFrozenSetMixin,
    RelationalAlgebraFrozenSet
):
    def unwrap(self):
        return RelationalAlgebraFrozenSet.create_view_from(self)


class WrappedRelationalAlgebraSet(
    WrappedRelationalAlgebraSetMixin,
    RelationalAlgebraSet
):
    def unwrap(self):
        return RelationalAlgebraSet.create_view_from(self)


class WrappedNamedRelationalAlgebraFrozenSet(
    WrappedNamedRelationalAlgebraFrozenSetMixin,
    NamedRelationalAlgebraFrozenSet
):
    def unwrap(self):
        return NamedRelationalAlgebraFrozenSet.create_view_from(self)
