from collections.abc import Set, Sequence
from itertools import tee


class OrderedSet(Set, Sequence):
    def __init__(self, iterable):

        it1, it2 = tee(iterable)
        self._set = frozenset(it1)
        self._list = []
        for i in it2:
            if i not in self._list:
                self._list.append(i)

    def __contains__(self, element):
        return element in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, slice_):
        return self._list[slice_]

    def __repr__(self):
        return 'OrderedSet(' + ','.join(repr(i) for i in self) + ')'
