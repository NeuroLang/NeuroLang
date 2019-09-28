from itertools import tee
from typing import Tuple

from ..expressions import Constant
from ..type_system import infer_type
from ..utils import RelationalAlgebraSet


class WrappedExpressionIterable:
    def __init__(self, iterable=None):
        self.__row_type = None
        if iterable is not None:
            if isinstance(iterable, type(self)):
                iterable = iterable.unwrapped_iter()
            else:
                it1, it2 = tee(iterable)
                try:
                    if isinstance(next(it1), Constant[Tuple]):
                        iterable = list(
                            tuple(a.value for a in e.value)
                            for e in it2
                        )
                except StopIteration:
                    pass

        super().__init__(iterable)

    def __iter__(self):
        type_ = self.row_type
        return (
            Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(type_.__args__, t)
                ),
                verify_type=False
            )
            for t in super().__iter__()
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
