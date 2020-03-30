import operator as op
from collections.abc import Set
from functools import wraps
from inspect import getmro
from itertools import tee
from typing import Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant
from ..type_system import infer_type
from ..utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet, RelationalAlgebraSet)

REBV = ReplaceExpressionsByValues(dict())


def _obtain_value_iterable(iterable):
    it1, it2 = tee(iterable)
    iterator_of_constants = False
    for val in it1:
        iterator_of_constants = isinstance(val, Constant[Tuple])
        break
    if not iterator_of_constants:
        return iterable
    else:
        for e in it2:
            yield REBV.walk(e)


def unwrapped_operator_factory(operator, operator_name):
    @wraps(operator)
    def unwrapped_operator(self, other):
        if not isinstance(other, WrappedRelationalAlgebraSetMixin):
            other = (REBV.walk(el) for el in other)
        return operator(self, other)
    return unwrapped_operator


class WrappedRelationalAlgebraSetType(type):
    def __new__(cls, name, bases, classdict, **kwargscls):
        for operator_name in (
            '__sub__', '__or__', '__and__',
            '__eq__', '__ne__',
            '__lt__', '__gt__',
            '__le__', '__ge__',
            '__ior__', '__isub__'
        ):
            old_method = classdict[operator_name]
            wrapped_operator_name = f'_wrapped_{operator_name}'
            classdict[wrapped_operator_name] = old_method
            classdict[operator_name] = unwrapped_operator_factory(
                old_method,
                wrapped_operator_name
            )
        return super().__new__(cls, name, bases, classdict, **kwargscls)


class WrappedRelationalAlgebraSetMixin:
    def __init__(self, iterable=None, **kwargs):
        kwargs = kwargs.copy()
        if iterable is not None:
            if isinstance(iterable, (WrappedNamedRelationalAlgebraFrozenSet, WrappedRelationalAlgebraSetMixin)):
                iterable = iterable.unwrap()
            else:
                iterable = _obtain_value_iterable(iterable)
        super().__init__(iterable=iterable, **kwargs)
        self._row_type = None

    def __contains__(self, element):
        element = REBV.walk(element)
        return super().__contains__(element)

    def __iter__(self):
        type_ = self.row_type
        element_types = type_.__args__

        for t in super().__iter__():
            yield Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(element_types, t)
                ),
                verify_type=False
            )

    def unwrapped_iter(self):
        return super().__iter__()

    def unwrap(self):
        raise super().copy()

    @property
    def row_type(self):
        if self._row_type is None:
            if self.arity > 0 and len(self) > 0:
                self._row_type = infer_type(next(super().__iter__()))
            else:
                self._row_type = Tuple

        return self._row_type


def unwrap(op, operator_name):
    @wraps(op)
    def fun(self, other):
        if hasattr(super(), operator_name):
            if not isinstance(other, WrappedRelationalAlgebraSetMixin):
                other = (REBV.walk(el) for el in other)
            return getattr(super(), operator_name)(other)
        else:
            return None

    return fun


class WrappedRelationalAlgebraSet(
    WrappedRelationalAlgebraSetMixin, RelationalAlgebraSet
):
    def add(self, element):
        super().add(REBV.walk(element))

    def discard(self, element):
        super().discard(REBV.walk(element))

    def __eq__(self, other):
        if not isinstance(other, WrappedRelationalAlgebraSetMixin):
            other = (REBV.walk(el) for el in other)
        return super().__eq__(other)



class WrappedNamedRelationalAlgebraFrozenSet(
    WrappedRelationalAlgebraSetMixin, NamedRelationalAlgebraFrozenSet
):
    @property
    def row_type(self):
        if self._row_type is None:
            if (self.arity > 0 and len(self) > 0):
                element = next(super().__iter__())
                self._row_type = {
                    c: Constant(getattr(element, c)).type
                    for c in self.columns
                }
            else:
                self._row_type = dict()

        return self._row_type

    def __iter__(self):
        if self.arity > 0:
            row_types = self.row_type
            for row in super().__iter__():
                yield {
                    f: Constant[row_types[f]](
                        v, verify_type=False
                    )
                    for f, v in zip(row._fields, row)
                }
        else:
            for _ in range(len(self)):
                yield dict()

    def __eq__(self, other):
        if not isinstance(other, WrappedRelationalAlgebraSetMixin):
            other = (REBV.walk(el) for el in other)
        return super().__eq__(other)

