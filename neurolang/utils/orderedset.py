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

    def add(self, element):
        if element not in self._set:
            self._set.add(element)
            self._list.append(element)

    def discard(self, element):
        if element in self._set:
            self._set.discard(element)
            self._list.remove(element)

    def __contains__(self, element):
        return element in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, slice_):
        return self._list[slice_]

    def index(self, element):
        return self._list.index(element)

    def __repr__(self):
        return 'OrderedSet(' + ','.join(repr(i) for i in self) + ')'
