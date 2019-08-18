from itertools import product
from typing import MutableSet

import pandas as pd


class RelationalAlgebraSet(MutableSet):
    def __init__(self, iterable=None):
        self._container = None
        if iterable is not None:
            it = iter(iterable)
            self._container = pd.DataFrame(
                list(iterable),
                index=[hash(e) for e in it]
            )
            if len(self._container > 0):
                duplicated = self._container.index.duplicated()
                if duplicated.any():
                    self._container = self._container.loc[~duplicated].dropna()

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
        try:
            self._container.drop(index=hash(element), inplace=True)
        except KeyError:
            pass

    @staticmethod
    def _renew_index(container, drop_duplicates=True):
        container.set_index(
            container.apply(lambda x: hash(tuple(x)), axis=1),
            inplace=True
        )

        if drop_duplicates:
            duplicated = container.index.duplicated()
            if duplicated.any():
                container = container.loc[~duplicated].dropna()

        return container

    @property
    def arity(self):
        if self._container is None:
            return 0
        else:
            return len(self._container.columns)

    def projection(self, *columns):
        new_container = self._container[list(columns)]
        output = type(self)()
        output._container = self._renew_index(new_container)
        return output

    def selection(self, select_criteria):
        it = iter(select_criteria.items())
        col, value = next(it)
        ix = self._container[col] == value
        for col, value in it:
            ix &= self._container[col] == value

        new_container = self._container[ix]

        output = type(self)()
        output._container = new_container
        return output

    def selection_columns(self, select_criteria):
        it = iter(select_criteria.items())
        col1, col2 = next(it)
        ix = self._container[col1] == self._container[col2]
        for col, value in it:
            ix &= self._container[col1] == self._container[col2]

        new_container = self._container[ix]

        output = type(self)()
        output._container = new_container
        return output

    def equijoin(self, other, join_indices, return_mappings=False):
        other = pd.DataFrame(
            other._container.values,
            index=other._container.index,
            columns=range(self.arity, other._container.shape[1] + self.arity),
            copy=False
        )
        left_on, right_on = zip(*join_indices)
        left_on = list(left_on)
        right_on = list(l + self.arity for l in right_on)
        new_container = self._container.merge(
            other,
            left_on=left_on,
            right_on=right_on,
            sort=False,
        )
        output = type(self)()
        output._container = self._renew_index(new_container)
        return output

    def cross_product(self, other):
        new_container = pd.DataFrame([
            tuple(t1) + tuple(t2)
            for t1, t2 in product(
                self._container.values,
                other._container.values
            )
        ])
        result = type(self)()
        result._container = self._renew_index(new_container)
        return result

    def copy(self):
        output = type(self)()
        if self._container is not None:
            output._container = self._container.copy()
        return output

    def __repr__(self):
        if self._container is None:
            return '{}'
        return repr(
            self._container.reset_index()
            .drop('index', axis=1)
        )

    def __or__(self, other):
        if self is other:
            return self.copy()
        elif isinstance(other, RelationalAlgebraSet):
            other = other._container
            new_container = self._container.append(
                other.loc[~other.index.isin(self._container.index)]
            )
            output = output = type(self)()
            output._container = new_container
            return output
        else:
            return super().__or__(self, other)

    def __and__(self, other):
        if isinstance(other, RelationalAlgebraSet):
            index_intersection = self._container.index & other._container.index
            new_container = self._container.loc[index_intersection]
            output = output = type(self)()
            output._container = new_container
            return output

        else:
            return super().__and__(self, other)

    def __isub__(self, other):
        if isinstance(other, RelationalAlgebraSet):
            diff_ix = ~self._container.index.isin(other._container.index)
            self._container = self._container.loc[diff_ix]
            return self
        else:
            return super().__isub__(self, other)
