from inspect import getmro
from itertools import tee
from typing import Tuple

from ..expression_walker import ReplaceExpressionsByValues
from ..expressions import Constant
from ..type_system import infer_type
from ..utils.relational_algebra_set.sql import (
    NamedRelationalAlgebraFrozenSet, RelationalAlgebraSet)

REBV = ReplaceExpressionsByValues(dict())


class WrappedExpressionIterable:
    def __init__(self, iterable=None):
        self.__row_type = None
        if iterable is not None:
            if isinstance(iterable, type(self)):
                iterable = iterable.unwrapped_iter()
            else:
                iterable = type(self)._obtain_value_iterable(iterable)
            iterable = list(iterable)
        super().__init__(iterable)

    @staticmethod
    def _obtain_value_iterable(iterable):
        it1, it2 = tee(iterable)
        iterator_of_constants = False
        for val in it1:
            iterator_of_constants = isinstance(val, Constant[Tuple])
            break
        if iterator_of_constants:
            iterable = []
            for e in it2:
                iterable.append(REBV.walk(e))
        return iterable

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
        raise NotImplementedError()

    def add(self, element):
        if isinstance(element, Constant[Tuple]):
            element = element.value
        element_ = tuple()
        for e in element:
            if isinstance(e, Constant):
                e = e.value
            element_ += (e,)
        super().add(element_)

    @property
    def row_type(self):
        if self.__row_type is None:
            if self.arity > 0 and len(self) > 0:
                self.__row_type = infer_type(next(super().__iter__()))
            else:
                self.__row_type = Tuple

        return self.__row_type


class WrappedRelationalAlgebraSet(
    WrappedExpressionIterable, RelationalAlgebraSet
):
    def __contains__(self, element):
        if (
            isinstance(element, Constant) or (
                isinstance(element, tuple) and
                len(element) > 0 and
                isinstance(element[0], Constant)
            )
        ):
            element = REBV.walk(element)
        else:
            element = self._normalise_element(element)
        return super(RelationalAlgebraSet, self).__contains__(element)

    def __eq__(self, other):
        if isinstance(other, WrappedRelationalAlgebraSet):
            return super().__eq__(other)
        else:
            return all(
                e in self for e in other
            )

    def unwrap(self):
        res = RelationalAlgebraSet(iterable=self)
        return res


class WrappedNamedRelationalAlgebraFrozenSet(NamedRelationalAlgebraFrozenSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_types = dict()
        self.row_types

    @property
    def row_types(self):
        if (
            len(self._row_types) == 0 and
            self.arity > 0 and len(self) > 0
        ):
            element = next(super().__iter__())
            self._row_types = {
                c: Constant(getattr(element, c)).type
                for c in self.columns
            }

        return self._row_types

    def __iter__(self):
        if self.arity > 0:
            row_types = self.row_types
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

    def unwrapped_iter(self):
        return super().__iter__()
