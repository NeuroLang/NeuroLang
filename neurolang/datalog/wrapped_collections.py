from itertools import tee
from typing import Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant
from ..type_system import Unknown, get_args, infer_type, unify_types
from ..utils.relational_algebra_set import (NamedRelationalAlgebraFrozenSet,
                                            RelationalAlgebraFrozenSet,
                                            RelationalAlgebraSet)

REBV = ReplaceExpressionsByValues(dict())


class WrappedRelationalAlgebraSetMixin:
    def __init__(
        self, iterable=None, row_type=Unknown, verify_row_type=True, **kwargs
    ):
        iterable = WrappedRelationalAlgebraSetMixin._get_init_iterable(
            iterable
        )
        super().__init__(iterable=iterable, **kwargs)
        self._set_row_type(iterable, row_type, verify_row_type)

    def _set_row_type(self, iterable, row_type, verify_row_type):
        if row_type is not Unknown:
            if verify_row_type:
                raise NotImplemented()
            self._row_type = row_type
        elif isinstance(iterable, WrappedRelationalAlgebraSetMixin):
            self._row_type = iterable._row_type
        else:
            self._row_type = None

    @staticmethod
    def _get_init_iterable(iterable):
        if iterable is not None:
            if isinstance(
                iterable,
                (WrappedRelationalAlgebraSetMixin, RelationalAlgebraFrozenSet)
            ):
                iterable = iterable  # ._container.values  # iterable.unwrap()
            else:
                iterable = (
                    WrappedRelationalAlgebraSetMixin._obtain_value_iterable(
                        iterable
                    )
                )
        return iterable

    def __contains__(self, element):
        element = REBV.walk(element)
        return super().__contains__(element)

    def _operator_wrapped(self, op, other):
        other_is_wras = isinstance(other, WrappedRelationalAlgebraSetMixin)
        if not other_is_wras:
            other = {el for el in self._obtain_value_iterable(other)}
        operator = getattr(super(), op)
        res = operator(other)
        if isinstance(res, WrappedRelationalAlgebraSetMixin):
            row_type = self._row_type
            if other_is_wras:
                if row_type is not None and other._row_type is not None:
                    row_type = unify_types(row_type, other._row_type)
                elif row_type is None:
                    row_type = other._row_type
            res._row_type = row_type
        return res

    @staticmethod
    def _obtain_value_iterable(iterable):
        it1, it2 = tee(iterable)
        iterator_of_constants = False
        for val in it1:
            iterator_of_constants = (
                isinstance(val, Constant[Tuple]) or
                (
                    isinstance(val, tuple) and (len(val) > 0)
                    and isinstance(val[0], Constant)
                )
            )
            break
        if not iterator_of_constants:
            iterator = it2
        else:
            iterator = (REBV.walk(e) for e in it2)
        for e in iterator:
            yield e

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

    def unwrap(self):
        return super().copy()

    @property
    def row_type(self):
        if self._row_type is None:
            if self.arity > 0 and len(self) > 0:
                self._row_type = infer_type(next(super().__iter__()))
            else:
                self._row_type = Tuple

        return self._row_type


class WrappedRelationalAlgebraFrozenSet(
    WrappedRelationalAlgebraSetMixin, RelationalAlgebraFrozenSet
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


class WrappedRelationalAlgebraSet(
    WrappedRelationalAlgebraSetMixin, RelationalAlgebraSet
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


class WrappedNamedRelationalAlgebraFrozenSet(
    WrappedRelationalAlgebraSetMixin, NamedRelationalAlgebraFrozenSet
):
    def __init__(
        self, columns=None, iterable=None,
        row_type=Unknown, verify_row_type=True, **kwargs
    ):
        iterable = WrappedRelationalAlgebraSetMixin._get_init_iterable(
            iterable
        )
        super().__init__(columns=columns, iterable=iterable, **kwargs)
        self._set_row_type(iterable, row_type, verify_row_type)

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
