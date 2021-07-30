from collections.abc import MutableSet, Sequence
from itertools import tee


class OrderedSet(MutableSet, Sequence):
    def __init__(self, iterable=None):
        if iterable is None:
            iterable = []

        it1, it2 = tee(iterable)
        self._set = set(it1)
        self._list = []
        for i in it2:
            if i not in self._list:
                self._list.append(i)

    def add(self, value):
        if value not in self._set:
            self._set.add(value)
            self._list.append(value)

    def discard(self, value):
        if value in self._set:
            self._set.discard(value)
            self._list.remove(value)

    def replace(self, src, dst):
        src_ix = self._list.index(src)
        self._list = (
            self._list[:src_ix] +
            [dst] +
            self._list[src_ix + 1:]
        )
        self._set.discard(src)
        self._set.add(dst)

    def issubset(self, other):
        return self <= other

    def issuperset(self, other):
        return self >= other

    def __contains__(self, value):
        return value in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, slice_):
        return self._list[slice_]

    def index(self, value, start=0, stop=None):
        return self._list[start: stop].index(value)

    def __repr__(self):
        return 'OrderedSet(' + ','.join(repr(i) for i in self) + ')'

    def copy(self):
        res = OrderedSet()
        res._set = self._set.copy()
        res._list = self._list.copy()
        return res
