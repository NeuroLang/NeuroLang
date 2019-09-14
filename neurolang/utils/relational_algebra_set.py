from itertools import product
from collections.abc import MutableSet, Set
from typing import Iterable

import pandas as pd


class RelationalAlgebraFrozenSet(Set):
    def __init__(self, iterable=None):
        self._container = None
        if iterable is not None:
            if isinstance(iterable, RelationalAlgebraFrozenSet):
                self._container = iterable._container
            else:
                self._container = pd.DataFrame(
                    list(iterable),
                )
                self._container = self._renew_index(self._container)

    def __contains__(self, element):
        element = self._normalise_element(element)
        return (
            self._container is not None and
            hash(element) in self._container.index
        )

    @staticmethod
    def _normalise_element(element):
        if isinstance(element, tuple):
            pass
        elif hasattr(element, '__iter__'):
            element = tuple(element)
        else:
            element = (element,)
        return element

    def __iter__(self):
        if self._container is not None:
            return (tuple(v) for v in self._container.values)
        else:
            return iter({})

    def __len__(self):
        if self._container is None:
            return 0
        return len(self._container)

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
        if self._container is None:
            return type(self)()
        new_container = self._container[list(columns)]
        output = type(self)()
        output._container = self._renew_index(
            new_container,
            drop_duplicates=True
        )
        return output

    def selection(self, select_criteria):
        if self._container is None:
            return type(self)()
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
        if self._container is None:
            return type(self)()
        it = iter(select_criteria.items())
        col1, col2 = next(it)
        ix = self._container[col1] == self._container[col2]
        for col1, col2 in it:
            ix &= self._container[col1] == self._container[col2]

        new_container = self._container[ix]

        output = type(self)()
        output._container = new_container
        return output

    def equijoin(self, other, join_indices, return_mappings=False):
        if self._container is None or len(other) == 0:
            return type(self)()
        other_columns = range(
            self.arity,
            other._container.shape[1] + self.arity
        )
        other = other._container.copy(deep=False)
        other.columns = other_columns
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
        if self._container is None:
            return type(self)()
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
            output = type(self)()
            output._container = new_container
            return output
        else:
            return super().__or__(other)

    def __and__(self, other):
        if self._container is None:
            return self.copy()
        if isinstance(other, RelationalAlgebraSet):
            index_intersection = self._container.index & other._container.index
            new_container = self._container.loc[index_intersection]
            output = type(self)()
            output._container = new_container
            return output

        else:
            return super().__and__(other)

    def groupby(self, columns):
        if self._container is None:
            raise StopIteration
        if not isinstance(columns, Iterable):
            columns = [columns]
        for g_id, group in self._container.groupby(by=list(columns)):
            group_set = type(self)()
            group_set._container = group
            yield g_id, group_set

    def itervalues(self):
        if self._container is None:
            raise StopIteration
        return iter(self._container.values.squeeze())


class NamedRelationalAlgebraFrozenSet(RelationalAlgebraFrozenSet):
    def __init__(self, columns, iterable=None):
        self._columns = tuple(columns)
        self._columns_sort = tuple(pd.Index(columns).argsort())
        if iterable is None:
            iterable = []

        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._container = iterable._container.copy(deep=False)
            if len(self._columns) != iterable.arity:
                raise ValueError(
                    'columns should have the same '
                    'length as columns of {iterable}'
                )
            self._container.columns = self._columns
            self._container.sort_index(axis=1, inplace=True)
        else:
            self._container = pd.DataFrame(
                list(iterable),
                columns=self._columns
            )
            self._container = self._renew_index(self._container)

    @property
    def arity(self):
        return len(self._columns)

    @property
    def columns(self):
        return self._columns

    def __contains__(self, element):
        if isinstance(element, dict) and len(element) == self.arity:
            element = tuple(element[c] for c in self._container.columns)
        else:
            element = tuple(element[i] for i in self._columns_sort)
        return super().__contains__(element)

    def projection(self, *columns):
        if self._container is None:
            return type(self)(columns)
        new_container = self._container[list(columns)]
        output = type(self)(columns)
        output._container = self._renew_index(
            new_container,
            drop_duplicates=True
        )
        return output

    def selection(self, select_criteria):
        if self._container is None:
            return type(self)(self.columns)
        it = iter(select_criteria.items())
        col, value = next(it)
        ix = (self._container[col] == value).values
        for col, value in it:
            ix *= (self._container[col] == value).values

        new_container = self._container[ix]

        output = type(self)(self.columns)
        output._container = new_container
        return output

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def naturaljoin(self, other):
        on = [
            c for c in self.columns
            if c in other.columns
        ]

        if len(on) == 0:
            return self.cross_product(other)

        new_columns = self.columns + tuple(
            c for c in other.columns
            if c not in self.columns
        )

        new_container = self._container.merge(other._container)

        output = type(self)(new_columns)
        output._container = output._renew_index(new_container)
        return output

    def cross_product(self, other):
        if len(
            self._container.columns.intersection(other.columns)
        ) > 0:
            raise ValueError(
                'Cross product with common columns '
                'is not valid'
            )
        new_columns = self.columns + other.columns
        if self._container is None:
            return type(self)(new_columns)
        new_container = pd.DataFrame(
            [
                tuple(t1) + tuple(t2)
                for t1, t2 in product(
                    self._container.values,
                    other._container.values
                )
            ],
            columns=(
                tuple(self._container.columns) +
                tuple(other._container.columns)
            )
        )
        result = type(self)(new_columns)
        result._container = self._renew_index(new_container)
        return result

    def __eq__(self, other):
        scontainer = self._container
        ocontainer = other._container
        return (
            scontainer.columns.equals(ocontainer.columns) and
            (
                len(scontainer.index.difference(ocontainer.index))
                == 0
            )
        )

    def _renew_index(self, container, drop_duplicates=True):
        container.sort_index(axis=1, inplace=True)
        return super()._renew_index(container, drop_duplicates=True)

    def groupby(self, columns):
        if self._container is None:
            raise StopIteration
        if not isinstance(columns, Iterable):
            columns = [columns]
        for g_id, group in self._container.groupby(by=list(columns)):
            group_set = type(self)(self.columns)
            group_set._container = group
            yield g_id, group_set

    def __iter__(self):
        container = self._container[list(self.columns)]
        return container.itertuples(index=False, name='tuple')

    def to_unnamed(self):
        container = self._container[list(self.columns)]
        output = RelationalAlgebraFrozenSet()
        output._container = container
        return output

    def __sub__(self, other):
        if self.columns != other.columns:
            raise ValueError(
                'Difference defined only for '
                'sets with the same columns'
            )
        new_container_ix = self._container.index.difference(
            other._container.index
        )
        new_container = self._container.loc[new_container_ix]
        output = type(self)(self.columns)
        output._container = new_container
        return output

    def __or__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):
    def add(self, value):
        value = self._normalise_element(value)
        e_hash = hash(value)
        if self._container is None:
            self._container = pd.DataFrame([value], index=[e_hash])
        else:
            self._container.loc[e_hash] = value

    def discard(self, value):
        if self._container is not None:
            try:
                value = self._normalise_element(value)
                self._container.drop(index=hash(value), inplace=True)
            except KeyError:
                pass

    def __isub__(self, other):
        if self._container is None:
            return self
        if isinstance(other, RelationalAlgebraSet):
            diff_ix = ~self._container.index.isin(other._container.index)
            self._container = self._container.loc[diff_ix]
            return self
        else:
            return super().__isub__(other)
