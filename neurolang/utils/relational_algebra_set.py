from collections.abc import MutableSet, Iterable

import pandas as pd


__all__ = [
    'RelationalAlgebraSet'
]


class RelationalAlgebraSet(MutableSet):
    def __init__(self, iterable=None):
        if iterable is None:
            self._set = pd.DataFrame()
        else:
            try:
                self._set = pd.DataFrame.from_records(iter(iterable))
                self._set = self._set.drop_duplicates()
            except TypeError:
                raise TypeError('Elements must be iterables')

    def __contains__(self, element):
        if len(self._set) == 0:
            return False
        if len(self._set.columns) == 1 and not isinstance(element, Iterable):
            return any(self._set.isin(element))
        else:
            iterator = enumerate(element)
            i, e = next(iterator)
            selection = (self._set[i] != e)
            if all(selection):
                return False
            for i, e in iterator:
                selection |= (self._set[i] != e)
                if all(selection):
                    break
            else:
                return True
            return False

    def __iter__(self):
        return (tuple(e) for e in self._set.values)

    def __len__(self):
        return len(self._set)

    def __or__(self, other):
        if len(self) == 0:
            return other.copy()

        if len(other) == 0:
            return self.copy()

        if not isinstance(other, RelationalAlgebraSet):
            other = RelationalAlgebraSet(other)
        newset = pd.merge(self._set, other._set, how='outer')
        res = RelationalAlgebraSet()
        res._set = newset
        return res

    def __and__(self, other):
        if len(self) == 0 or len(other) == 0:
            return RelationalAlgebraSet()

        if not isinstance(other, RelationalAlgebraSet):
            other = RelationalAlgebraSet(other)
        newset = pd.merge(self._set, other._set)
        res = RelationalAlgebraSet()
        res._set = newset
        return res

    def __eq__(self, other):
        if isinstance(other, RelationalAlgebraSet):
            return len(self & other) == len(self)
        else:
            return super().__eq__(other)

    def __le__(self, other):
        if isinstance(super, RelationalAlgebraSet):
            return len(self & other) <= len(self)
        else:
            return super().__le__(other)

    def __ior__(self, other):
        res = self | other
        self._set = res._set
        return self

    def __iand__(self, other):
        res = self & other
        self._set = res._set
        return self

    def add(self, element):
        if len(self) == 0:
            self._set = pd.DataFrame.from_records([element])
        else:
            if element not in self:
                element = {i: e for i, e in enumerate(element)}
                self._set = self._set.append(element, ignore_index=True)

    def discard(self, element):
        iterator = enumerate(element)
        i, e = next(iterator)
        selection = (self._set[i] != e)
        if all(selection):
            return
        for i, e in iterator:
            selection |= (self._set[i] != e)
            if all(selection):
                break
        else:
            self._set = self._set[selection]

    def project(self, columns):
        """Relational Algebra projection without mapping"""
        if len(columns) == 0 or len(self) == 0:
            return RelationalAlgebraSet()

        if not isinstance(columns, Iterable):
            columns = (columns,)
        ret = RelationalAlgebraSet()
        ret._set = self._set[list(columns)]
        return ret

    def select_equality(self, criteria):
        """
        Select on equality constraints.

        Criteria is a dict of column number, constant to compare
        """
        if len(criteria) == 0 or len(self) == 0:
            return self

        iterator = iter(criteria.items())
        k, v = next(iterator)
        selection = (self._set[k] == v)
        for k, v in iterator:
            selection &= (self._set[k] == v)

        res = RelationalAlgebraSet()
        res._set = self._set[selection]
        res._set = self._set[selection]
        res._set.columns = range(len(res._set.columns))
        return res

    def select_columns(self, criteria):
        """
        Select on equality of columns.

        Criteria is a dict of column number pairs
        """
        if len(criteria) == 0 or len(self) == 0:
            return self

        iterator = iter(criteria.items())
        k, v = next(iterator)
        selection = (self._set[k] == self._set[v])
        for k, v in iterator:
            selection &= (self._set[k] == self._set[v])

        res = RelationalAlgebraSet()
        res._set = self._set[selection]
        res._set.columns = range(len(res._set.columns))
        return res

    def join_by_columns(self, other, columns_self, columns_other):
        if len(self) == 0 or len(other) == 0:
            return RelationalAlgebraSet()

        other_frame = other._set.copy(deep=False)
        other_frame.columns += len(self._set.columns)
        columns_other = tuple(
            i + len(self._set.columns) for i in columns_other
        )

        joined = pd.merge(
            self._set, other_frame,
            left_on=columns_self,
            right_on=columns_other
        )

        res = RelationalAlgebraSet()
        res._set = joined
        return res

    def __repr__(self):
        return repr(self._set)

    def copy(self):
        res = RelationalAlgebraSet()
        res._set = self._set.copy()
        return res
