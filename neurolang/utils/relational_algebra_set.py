from typing import MutableSet

import pandas as pd


class RelationalAlgebraSet(MutableSet):
    def __init__(self, iterable=None):
        self._container = None
        if iterable is not None:
            it = iter(iterable)
            self._container = pd.DataFrame(
                iterable,
                index=[hash(e) for e in it]
            ).drop_duplicates()

    def __contains__(self, element):
        return (
            self._container is not None and
            hash(element) in self._container.index
        )

    def __iter__(self):
        if self._container is not None:
            return (tuple(v) for v in self._container.values)
        else:
            return iter({})

    def __len__(self):
        if self._container is None:
            return 0
        return len(self._container)

    def add(self, element):
        e_hash = hash(element)
        if self._container is None:
            self._container = pd.DataFrame([element], index=[e_hash])
        else:
            self._container.loc[hash(element)] = element

    def discard(self, element):
        self._container.drop(index=hash(element), inplace=True)

    @staticmethod
    def _renew_index(container, drop_duplicates=True):
        container.set_index(
            container.apply(lambda x: hash(tuple(x)), axis=1),
            inplace=True
        )

        if drop_duplicates:
            container.drop_duplicates(inplace=True)

        return container

    def projection(self, *columns):
        new_container = self._container[list(columns)]
        output = type(self)()
        output._container = self._renew_index(new_container)
        return output

    def selection(self, select_criteria):
        def crit(x):
            it = iter(select_criteria.items())
            i, j = next(it)
            selection = x[i] == j
            for i, j in it:
                selection &= x[i] == j
            return selection

        new_container = self._container[crit]

        output = type(self)()
        output._container = new_container
        return output

    def natural_join(self, other, join_indices):
        other = other._container
        orig_columns = other.columns
        self_n_columns = len(self._container.columns)
        other.columns += self_n_columns
        left_on, right_on = zip(*join_indices)
        new_container = self._container.merge(
            other,
            left_on=list(left_on),
            right_on=list(i + self_n_columns for i in right_on)
        )
        other.columns = orig_columns
        output = type(self)()
        output._container = self._renew_index(new_container)
        output._container.columns = range(0, output._container.shape[1])
        return output
