from collections.abc import MutableSet, Set
from typing import Iterable
from uuid import uuid1

import pandas as pd
import re


class RelationalAlgebraFrozenSet(Set):
    def __init__(self, iterable=None):
        self._container = None
        if iterable is not None:
            if isinstance(iterable, RelationalAlgebraFrozenSet):
                self._container = iterable._container
            else:
                self._container = pd.DataFrame(iterable)
                self._container = self._renew_index(self._container)

    def __contains__(self, element):
        element = self._normalise_element(element)
        return len(self) > 0 and hash(element) in self._container.index

    @staticmethod
    def _normalise_element(element):
        if isinstance(element, tuple):
            pass
        elif hasattr(element, "__iter__"):
            element = tuple(element)
        else:
            element = (element, )
        return element

    def __iter__(self):
        if len(self) == 0:
            values = {}
        else:
            values = self._container.values
        for v in values:
            yield tuple(v)

    def __len__(self):
        if self._container is None:
            return 0
        return len(self._container)

    @staticmethod
    def _renew_index(container, drop_duplicates=True):
        if len(container) == 0 or len(container.columns) == 0:
            return container

        RelationalAlgebraFrozenSet.refresh_index(container)
        if drop_duplicates:
            duplicated = container.index.duplicated()
            if duplicated.any():
                container = container.loc[~duplicated].dropna()

        return container

    @staticmethod
    def refresh_index(container):
        new_indices = pd.Index(
            hash(t) for t in container.itertuples(index=False, name=None)
        )
        container.set_index(new_indices, inplace=True)

    @property
    def arity(self):
        if len(self) == 0:
            return 0
        else:
            return len(self._container.columns)

    def _empty_set_same_structure(self):
        return type(self)()

    def projection(self, *columns):
        if len(self) == 0:
            return self._empty_set_same_structure()
        new_container = self._container[list(columns)]
        output = self._empty_set_same_structure()
        output._container = self._renew_index(
            new_container, drop_duplicates=True
        )
        output._container.rename(
            columns={c: i
                     for i, c in enumerate(output._container.columns)},
            inplace=True,
        )
        return output

    def selection(self, select_criteria):
        if len(self) == 0:
            return self._empty_set_same_structure()
        it = iter(select_criteria.items())
        col, value = next(it)
        ix = self._container[col] == value
        for col, value in it:
            ix &= self._container[col] == value

        new_container = self._container[ix]

        output = self._empty_set_same_structure()
        output._container = new_container
        return output

    def selection_columns(self, select_criteria):
        if len(self) == 0:
            return self._empty_set_same_structure()
        it = iter(select_criteria.items())
        col1, col2 = next(it)
        ix = self._container[col1] == self._container[col2]
        for col1, col2 in it:
            ix &= self._container[col1] == self._container[col2]

        new_container = self._container[ix]

        output = self._empty_set_same_structure()
        output._container = new_container
        return output

    def equijoin(self, other, join_indices, return_mappings=False):
        if len(self) == 0 or len(other) == 0:
            return self._empty_set_same_structure()
        other_columns = range(
            self.arity, other._container.shape[1] + self.arity
        )
        other = other._container.copy(deep=False)
        other.columns = other_columns
        left_on, right_on = zip(*join_indices)
        left_on = list(left_on)
        right_on = list(l + self.arity for l in right_on)
        new_container = self._container.merge(
            other, left_on=left_on, right_on=right_on, sort=False
        )
        output = self._empty_set_same_structure()
        output._container = self._renew_index(new_container)
        return output

    def cross_product(self, other):
        if len(self) == 0:
            return self._empty_set_same_structure()
        left = self._container.copy(deep=False)
        right = other._container.copy(deep=False)
        tmpcol = str(uuid1())
        left[tmpcol] = 1
        right[tmpcol] = 1
        new_container = pd.merge(left, right, on=tmpcol)
        del new_container[tmpcol]
        result = self._empty_set_same_structure()
        result._container = self._renew_index(new_container)
        return result

    def copy(self):
        output = self._empty_set_same_structure()
        if len(self) > 0:
            output._container = self._container.copy()
        return output

    def __repr__(self):
        if len(self) == 0:
            return "{}"
        return repr(self._container.reset_index().drop("index", axis=1))

    def __or__(self, other):
        if self is other:
            return self.copy()
        elif isinstance(other, RelationalAlgebraSet):
            other = other._container
            if self._container is None and other is None:
                new_container = None
            elif self._container is None:
                new_container = other.copy()
            elif other is None:
                new_container = self._container.copy()
            else:
                new_container = self._container.append(
                    other.loc[~other.index.isin(self._container.index)]
                )
            output = self._empty_set_same_structure()
            output._container = new_container
            return output
        else:
            return super().__or__(other)

    def __and__(self, other):
        if len(self) == 0:
            return self.copy()
        if isinstance(other, RelationalAlgebraSet):
            index_intersection = self._container.index & other._container.index
            new_container = self._container.loc[index_intersection]
            output = self._empty_set_same_structure()
            output._container = new_container
            return output

        else:
            return super().__and__(other)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            scont = self._container
            ocont = other._container
            return ((len(scont) == 0 and len(ocont) == 0) or
                    (len(scont.columns) == 0 and len(ocont.columns) == 0) or (
                        scont is not None and ocont is not None and
                        len(scont.index.difference(ocont.index)) == 0
                    ))
        else:
            return super().__eq__(other)

    def groupby(self, columns):
        if len(self) > 0:
            if not isinstance(columns, Iterable):
                columns = [columns]
            for g_id, group in self._container.groupby(by=list(columns)):
                group_set = self._empty_set_same_structure()
                group_set._container = group
                yield g_id, group_set

    def itervalues(self):
        if len(self) == 0:
            return iter([])
        else:
            return iter(self._container.values)

    def __hash__(self):
        v = self._container.values
        v.flags.writeable = False
        return hash(v.data.tobytes())


class NamedRelationalAlgebraFrozenSet(RelationalAlgebraFrozenSet):
    def __init__(self, columns, iterable=None):
        self._columns = tuple(columns)
        self._columns_sort = tuple(pd.Index(columns).argsort())
        if iterable is None:
            iterable = []

        if isinstance(iterable, NamedRelationalAlgebraFrozenSet):
            self._initialize_from_named_ra_set(iterable)
        elif isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_unnamed_ra_set(iterable)
        else:
            self._container = pd.DataFrame(
                list(iterable), columns=self._columns
            )
        self._container = self._renew_index(self._container)

    def _initialize_from_named_ra_set(self, other):
        if len(self._columns) != other.arity:
            raise ValueError("Relations must have the same arity")
        self._container = other._container[list(other.columns
                                                )].copy(deep=False)
        self._container.sort_index(axis=1, inplace=True)

    def _initialize_from_unnamed_ra_set(self, other):
        if other._container is None:
            self._container = pd.DataFrame(list(other), columns=self._columns)
        else:
            if len(self._columns) != other.arity:
                raise ValueError("Relations must have the same arity")
            self._container = other._container.copy(deep=False)
            self._container.columns = self._columns
        self._container.sort_index(axis=1, inplace=True)

    @property
    def arity(self):
        return len(self._columns)

    def _empty_set_same_structure(self):
        return type(self)(self.columns)

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
        if len(self) == 0:
            return type(self)(columns)
        new_container = self._container[list(columns)]
        output = type(self)(columns)
        output._container = self._renew_index(
            new_container, drop_duplicates=True
        )
        return output

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def naturaljoin(self, other):
        on = [c for c in self.columns if c in other.columns]

        if len(on) == 0:
            return self.cross_product(other)

        new_columns = self.columns + tuple(
            c for c in other.columns if c not in self.columns
        )

        new_container = self._container.merge(other._container)

        output = type(self)(new_columns)
        output._container = output._renew_index(new_container)
        return output

    def cross_product(self, other):
        if len(self._container.columns.intersection(other.columns)) > 0:
            raise ValueError(
                "Cross product with common columns "
                "is not valid"
            )
        new_columns = self.columns + other.columns
        if len(self) == 0:
            return type(self)(new_columns)
        left = self._container.copy(deep=False)
        right = other._container.copy(deep=False)
        tmpcol = str(uuid1())
        left[tmpcol] = 1
        right[tmpcol] = 1
        new_container = pd.merge(left, right, on=tmpcol)
        del new_container[tmpcol]
        new_container.columns = tuple(self._container.columns
                                      ) + tuple(other._container.columns)
        result = type(self)(new_columns)
        result._container = self._renew_index(new_container)
        return result

    def rename_column(self, src, dst):
        if src not in self._columns:
            raise ValueError(f"{src} not in columns")
        if src == dst:
            return self
        if dst in self._columns:
            raise ValueError(f"{dst} cannot be in the columns")
        src_idx = self._columns.index(src)
        new_columns = self._columns[:src_idx] + (dst,
                                                 ) + self._columns[src_idx +
                                                                   1:]
        new_container = self._container.rename(columns={src: dst})
        new_container.sort_index(axis=1, inplace=True)

        new_set = type(self)(new_columns)
        new_set._container = new_container
        new_set._columns_sort = tuple(pd.Index(new_columns).argsort())

        return new_set

    def __eq__(self, other):
        scontainer = self._container
        ocontainer = other._container
        return scontainer.columns.equals(
            ocontainer.columns
        ) and (len(scontainer.index.difference(ocontainer.index)) == 0)

    def _renew_index(self, container, drop_duplicates=True):
        container.sort_index(axis=1, inplace=True)
        return super()._renew_index(container, drop_duplicates=True)

    def groupby(self, columns):
        if len(self) == 0:
            raise StopIteration
        if not isinstance(columns, Iterable):
            columns = [columns]
        for g_id, group in self._container.groupby(by=list(columns)):
            group_set = type(self)(self.columns)
            group_set._container = group
            yield g_id, group_set

    def aggregate(self, group_columns, aggregate_function):
        new_container = self._container.groupby(group_columns
                                                ).agg(aggregate_function)
        new_container.reset_index(inplace=True)
        output = self._empty_set_same_structure()
        output._container = self._renew_index(new_container)
        return output

    def extended_projection(self, eval_expressions):
        new_columns = []
        new_container = self._container.copy()
        for op_column, operation in eval_expressions.items():
            new_columns.append(op_column)
            if isinstance(operation, str):
                op = f"{op_column}={operation}"
                new_container = self._container.eval(op)
            elif callable(operation):
                new_container[op_column] = self._container.apply(
                    operation, axis=1
                )
            else:
                new_container[op_column] = operation

        new_columns = self.columns + tuple(new_columns)
        output = type(self)(new_columns)
        output._container = self._renew_index(new_container)
        return output

    def __iter__(self):
        container = self._container[list(self.columns)]
        return container.itertuples(index=False, name="tuple")

    def to_unnamed(self):
        container = self._container[list(self.columns)].copy()
        container.columns = range(len(container.columns))
        output = RelationalAlgebraFrozenSet()
        output._container = container
        RelationalAlgebraFrozenSet._renew_index(
            container, drop_duplicates=False
        )
        return output

    def __sub__(self, other):
        if not self._container.columns.equals(other._container.columns):
            raise ValueError(
                "Difference defined only for sets with the same columns"
            )
        new_container_ix = self._container.index.difference(
            other._container.index
        )
        new_container = self._container.loc[new_container_ix]
        output = type(self)(self.columns)
        output._container = new_container
        return output

    def __or__(self, other):
        if self.columns != other.columns:
            raise ValueError(
                "Union defined only for sets with the same columns"
            )
        new_container = pd.merge(
            left=self._container.reset_index(),
            right=other._container.reset_index(),
            how="outer",
        ).set_index("index")
        output = type(self)(self.columns)
        output._container = new_container
        return output

    def __and__(self, other):
        if self.columns != other.columns:
            raise ValueError(
                "Union defined only for sets with the same columns"
            )
        new_container = pd.merge(
            left=self._container.reset_index(),
            right=other._container.reset_index(),
            how="inner",
        ).set_index("index")
        output = type(self)(self.columns)
        output._container = new_container
        return output

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
        if len(self) == 0:
            self._container = pd.DataFrame([value], index=[e_hash])
        else:
            self._container.loc[e_hash] = value

    def discard(self, value):
        if len(self) > 0:
            try:
                value = self._normalise_element(value)
                self._container.drop(index=hash(value), inplace=True)
            except KeyError:
                pass

    def __ior__(self, other):
        if self._container is not None:
            self._container = self._renew_index(self._container)
        if other._container is not None:
            other._container = other._renew_index(other._container)
        if len(self) == 0:
            self._container = other._container.copy()
            return self
        if isinstance(other, RelationalAlgebraSet):
            diff_ix = ~other._container.index.isin(self._container.index)
            self._container = self._container.append(
                other._container.loc[diff_ix]
            )
            return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        if self._container is not None:
            self._container = self._renew_index(self._container)
        if other._container is not None:
            other._container = other._renew_index(other._container)
        if len(self) == 0:
            return self
        if isinstance(other, RelationalAlgebraSet):
            diff_ix = ~self._container.index.isin(other._container.index)
            self._container = self._container.loc[diff_ix]
            return self
        else:
            return super().__isub__(other)
