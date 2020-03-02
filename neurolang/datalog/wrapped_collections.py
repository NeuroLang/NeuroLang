from itertools import tee
from typing import Tuple

from ..expressions import Constant
from ..expression_walker import ReplaceExpressionsByValues
from ..type_system import infer_type
from ..utils import RelationalAlgebraSet


REBV = ReplaceExpressionsByValues(dict())


class WrappedExpressionIterable:
    def __init__(self, iterable=None):
        if iterable is not None:
            orig_it = list(iterable)
            iterable = orig_it
        self.__row_type = None
        if iterable is not None:
            if isinstance(iterable, type(self)):
                iterable = iterable.unwrapped_iter()
            else:
                iterable = self._obtain_value_iterable(iterable)
            iterable = list(iterable)
        super().__init__(iterable)

    def _obtain_value_iterable(self, iterable):
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
        for t in super().__iter__():
            yield Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(type_.__args__, t)
                ),
                verify_type=False
            )

    def unwrapped_iter(self):
        return super().__iter__()

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
        if len(self) == 0:
            return None

        if self.__row_type is None:
            self.__row_type = infer_type(next(super().__iter__()))

        return self.__row_type


class WrappedRelationalAlgebraSet(
    WrappedExpressionIterable, RelationalAlgebraSet
):
    def __contains__(self, element):
        if not isinstance(element, Constant):
            element = self._normalise_element(element)
        return (
            self._container is not None and
            hash(element) in self._container.index
        )

    def __eq__(self, other):
        if isinstance(other, WrappedRelationalAlgebraSet):
            return super().__eq__(other)
        else:
            return all(
                e in self for e in other
            )
