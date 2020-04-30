from collections import OrderedDict
from collections.abc import MutableSet, Set
from typing import Iterable
from uuid import uuid1

import pandas as pd


class RelationalAlgebraStringExpression(str):
    def __repr__(self):
        return "{}{{ {} }}".format(self.__class__.__name__, super().__repr__())


class RelationalAlgebraFrozenSet(Set):
    def __init__(self, iterable=None):
        self._container = None
        self._might_have_duplicates = True
        if iterable is not None:
            if isinstance(iterable, RelationalAlgebraFrozenSet):
                self._container = iterable._container
            else:
                self._container = pd.DataFrame(iterable)

    def _drop_duplicates_if_needed(self):
        if self._might_have_duplicates:
                self._container = self._drop_duplicates(self._container)
                self._might_have_duplicates = False

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._container = other._container
        return output

    def is_null(self):
        return (
            self._container is None or
            len(self._container) == 0
        )

    def __contains__(self, element):
        element = self._normalise_element(element)
        if (
            len(self) > 0 and
            len(element) == self.arity
        ):
            col = True
            for e, c in zip(element, self._container.iteritems()):
                col = col & (c[1] == e)
            return col.any()
        else:
            return False

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
        if self.is_null():
            values = {}
        else:
            self._drop_duplicates_if_needed()
            values = self._container.values
        for v in values:
            yield tuple(v)

    def fetch_one(self):
        return tuple(next(iter(self._container.values)))

    def __len__(self):
        if self._container is None:
            return 0
        self._drop_duplicates_if_needed()
        return len(self._container)

    @staticmethod
    def _drop_duplicates(container):
        if len(container) == 0 or len(container.columns) == 0:
            return container
        container = container.drop_duplicates()
        return container

    @property
    def arity(self):
        if self.is_null():
            return 0
        else:
            return len(self._container.columns)

    @property
    def columns(self):
        return self._container.columns

    def _empty_set_same_structure(self):
        return type(self)()

    def projection(self, *columns):
        if self.is_null():
            return self._empty_set_same_structure()
        new_container = self._container[list(columns)]
        output = self._empty_set_same_structure()
        output._container = new_container
        output._container.rename(
            columns={c: i
                     for i, c in enumerate(output._container.columns)},
            inplace=True,
        )
        return output

    def selection(self, select_criteria):
        if self.is_null():
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
        if self.is_null():
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
        if self.is_null() or other.is_null():
            return self._empty_set_same_structure()
        other_columns = range(
            self.arity, other._container.shape[1] + self.arity
        )
        ocont = other._container.copy(deep=False)
        ocont.columns = other_columns
        left_on, right_on = zip(*join_indices)
        left_on = list(left_on)
        right_on = list(l + self.arity for l in right_on)
        new_container = self._container.merge(
            ocont, left_on=left_on, right_on=right_on, sort=False
        )
        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = (
            self._might_have_duplicates |
            other._might_have_duplicates
        )
        return output

    def cross_product(self, other):
        if self.is_null():
            return self._empty_set_same_structure()
        left = self._container.copy(deep=False)
        right = other._container.copy(deep=False)
        tmpcol = str(uuid1())
        left[tmpcol] = 1
        right[tmpcol] = 1
        new_container = pd.merge(left, right, on=tmpcol)
        del new_container[tmpcol]
        new_container.columns = range(new_container.shape[1])
        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = (
            self._might_have_duplicates |
            other._might_have_duplicates
        )
        return output

    def copy(self):
        output = self._empty_set_same_structure()
        if len(self) > 0:
            output._container = self._container.copy()
        return output

    def __repr__(self):
        if self.is_null():
            return "{}"
        return repr(self._container.reset_index().drop("index", axis=1))

    def __or__(self, other):
        if self is other:
            return self.copy()
        elif isinstance(other, RelationalAlgebraSet):
            ocont = other._container
            if self._container is None and ocont is None:
                new_container = None
            elif self._container is None:
                new_container = ocont.copy()
            elif ocont is None:
                new_container = self._container.copy()
            else:
                new_container = pd.merge(
                    left=self._container,
                    right=ocont,
                    how="outer",
                )
            output = self._empty_set_same_structure()
            output._container = new_container
            return output
        else:
            return super().__or__(other)

    def __and__(self, other):
        if self.is_null():
            return self.copy()
        if isinstance(other, RelationalAlgebraSet):
            new_container = pd.merge(
                left=self._container,
                right=other._container,
                how="inner",
            )
            output = self._empty_set_same_structure()
            output._container = new_container
            output._might_have_duplicates = (
                self._might_have_duplicates |
                other._might_have_duplicates
            )
            return output

        else:
            return super().__and__(other)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            scont = self._container
            ocont = other._container
            if len(scont) == 0 and len(ocont) == 0:
                res = True
            elif len(scont.columns) == 0 and len(ocont.columns) == 0:
                res = len(scont) > 0 and len(ocont) > 0
            elif scont is not None and ocont is not None:
                intersection_dups = scont.merge(
                    ocont, how='outer', indicator=True
                ).iloc[:, -1]
                res = (intersection_dups == 'both').all()
            else:
                res = False
            return res
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
        if self.is_null():
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
        self._might_have_duplicates = True
        if iterable is None:
            iterable = []

        if isinstance(iterable, NamedRelationalAlgebraFrozenSet):
            self._initialize_from_named_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        elif isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_unnamed_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        else:
            self._container = pd.DataFrame(
                iterable, columns=self._columns
            )
        self._container = self._sort_columns(
            self._container,
            drop_duplicates=False
        )

    def _initialize_from_named_ra_set(self, other):
        if (
            not (other.is_null() and other.arity == 0)
            and len(self._columns) != other.arity
        ):
            raise ValueError("Relations must have the same arity")

        if not other.is_null():
            self._container = other._container[list(other.columns
                                                    )].copy(deep=False)
        else:
            self._container = pd.DataFrame(columns=self._columns)
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

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls(columns=tuple())
        output._container = other._container
        output._columns = other._columns
        output._columns_sort = other._columns_sort
        output._might_have_duplicates = other._might_have_duplicates
        return output

    def _empty_set_same_structure(self):
        return type(self)(self.columns)

    def _light_init_same_structure(
        self, container,
        sort_columns=False,
        might_have_duplicates=True,
        columns=None
    ):
        if columns is None:
            columns = self.columns
        output = type(self)(columns)
        output._container = container
        if sort_columns:
            output._container = self._sort_columns(
                container,
                drop_duplicates=False
            )
        output._might_have_duplicates = might_have_duplicates
        return output

    @property
    def columns(self):
        return self._columns

    @property
    def arity(self):
        return len(self._columns)

    def __contains__(self, element):
        if isinstance(element, dict) and len(element) == self.arity:
            element = tuple(element[c] for c in self._container.columns)
        else:
            element = tuple(element[i] for i in self._columns_sort)
        return super().__contains__(element)

    def projection(self, *columns):
        if self.is_null():
            return type(self)(columns)
        new_container = self._container[list(columns)]
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=True,
            columns=columns
        )

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
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates |
                other._might_have_duplicates
            ),
            sort_columns=True,
            columns=new_columns
        )

    def cross_product(self, other):
        if len(self._container.columns.intersection(other.columns)) > 0:
            raise ValueError(
                "Cross product with common columns "
                "is not valid"
            )
        new_columns = self.columns + other.columns
        if self.is_null():
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
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates |
                other._might_have_duplicates
            ),
            sort_columns=True
        )

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
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=True,
            columns=new_columns
        )

    def rename_columns(self, renames):
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self._columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )
        new_columns = tuple(
            renames.get(col, col) for col in self._columns
        )
        new_container = self._container.rename(columns=renames)
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=True,
            columns=new_columns
        )

    def __eq__(self, other):
        scont = self._container
        ocont = other._container
        if not scont.columns.equals(ocont.columns):
            res = False
        elif len(scont) == 0 and len(ocont) == 0:
            res = True
        elif len(scont.columns) == 0 and len(ocont.columns) == 0:
            res = len(scont) > 0 and len(ocont) > 0
        elif scont is not None and ocont is not None:
            intersection_dups = scont.merge(
                ocont, how='outer', indicator=True
            ).iloc[:, -1]
            res = (intersection_dups == 'both').all()
        else:
            res = False
        return res

    def _sort_columns(self, container, drop_duplicates=True):
        container = container.sort_index(axis=1)
        if drop_duplicates:
            container = self._drop_duplicates(container)
        return container

    def groupby(self, columns):
        if self.is_null():
            raise StopIteration
        if not isinstance(columns, Iterable):
            columns = [columns]
        for g_id, group in self._container.groupby(by=list(columns)):
            group_set = self._light_init_same_structure(
                group,
                might_have_duplicates=self._might_have_duplicates,
                sort_columns=False,
                columns=self.columns
            )
            yield g_id, group_set

    def aggregate(self, group_columns, aggregate_function):
        group_columns = list(group_columns)
        if len(group_columns) > 0:
            groups = self._container.groupby(group_columns)
        else:
            groups = self._container.groupby(lambda x: 0)

        if (
            isinstance(aggregate_function, (tuple, list))
        ):
            args = OrderedDict({
                t[0]: pd.NamedAgg(t[1], t[2])
                for t in aggregate_function
            })
            new_container = groups.agg(**args)
            new_container.index = pd.RangeIndex(len(new_container))
        else:
            new_container = groups.agg(aggregate_function)
            new_container.reset_index(inplace=True)

        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=True,
            columns=list(new_container.columns)
        )
        return output

    def extended_projection(self, eval_expressions):
        proj_columns = list(eval_expressions.keys())
        new_container = self._container.copy()
        for dst_column, operation in eval_expressions.items():
            if isinstance(operation, RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    new_container = new_container.eval(
                        "{}={}".format(str(dst_column), str(operation))
                    )
            elif callable(operation):
                new_container[dst_column] = new_container.apply(
                    operation, axis=1
                )
            else:
                new_container[dst_column] = operation
        new_container = new_container[proj_columns]
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=True,
            columns=proj_columns
        )
        return output

    def __iter__(self):
        self._drop_duplicates_if_needed()
        container = self._container[list(self.columns)]
        return container.itertuples(index=False, name="tuple")

    def fetch_one(self):
        container = self._container[list(self.columns)]
        return next(container.itertuples(index=False, name="tuple"))

    def to_unnamed(self):
        container = self._container[list(self.columns)].copy()
        container.columns = range(len(container.columns))
        output = RelationalAlgebraFrozenSet()
        output._container = container
        return output

    def __sub__(self, other):
        if (
            (self.arity > 0 and other.arity > 0) and
            not self._container.columns.equals(other._container.columns)
        ):
            raise ValueError(
                "Difference defined only for sets with the same columns"
            )
        if self.is_null() or other.is_null():
            return self.copy()
        if (
            self.arity == 0 and
            len(self._container) > 0 and
            len(other._container) > 0
        ):
            return type(self)(self.columns)
        new_container = self._container.merge(
            other._container,
            indicator=True,
            how='outer'
        )
        new_container = new_container[
            new_container.iloc[:, -1] == 'left_only'
        ].iloc[:, :-1]

        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=False,
        )
        return output

    def __or__(self, other):
        if self.columns != other.columns:
            raise ValueError(
                "Union defined only for sets with the same columns"
            )
        if len(self.columns) == 0:
            if len(self._container) > 0:
                return self.copy()
            else:
                return other.copy()
        new_container = pd.merge(
            left=self._container,
            right=other._container,
            how="outer",
        )
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=True,
            sort_columns=False,
        )
        return output

    def __and__(self, other):
        if self.columns != other.columns:
            raise ValueError(
                "Union defined only for sets with the same columns"
            )
        new_container = pd.merge(
            left=self._container,
            right=other._container,
            how="inner",
        )
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            sort_columns=False,
        )
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
        if self.is_null():
            self._container = pd.DataFrame([value], index=[e_hash])
        else:
            self._container.loc[e_hash] = value

    def discard(self, value):
        if len(self) > 0:
            try:
                value = self._normalise_element(value)
                col = True
                for e, c in zip(value, self._container.iteritems()):
                    col = col & (c[1] == e)
                ix = self._container.index[col]
                self._container.drop(index=ix, inplace=True)
            except KeyError:
                pass

    def __ior__(self, other):
        if isinstance(other, RelationalAlgebraSet):
            if other.is_null() or other.arity == 0:
                return self
            if self.is_null():
                self._container = other._container.copy()
                self._might_have_duplicates = other._might_have_duplicates
                return self
            if other.arity != self.arity:
                raise ValueError(
                    "Operation only valid for sets with the same arity"
                )
            new_container = pd.merge(
                left=self._container,
                right=other._container,
                how="outer",
            )
            self._might_have_duplicates = True
            self._container = new_container
            return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        if self.is_null():
            return self
        if isinstance(other, RelationalAlgebraSet):
            if other.is_null() or other.arity == 0:
                if self.is_null() or other.arity == 0:
                    self._container = self._container.iloc[:0]
            elif other.arity != self.arity:
                raise ValueError(
                    "Operation only valid for sets with the same arity"
                )
            else:
                new_container = pd.merge(
                    left=self._container,
                    right=other._container,
                    how="left",  indicator=True
                )
                new_container = new_container[
                    new_container.iloc[:, -1] == 'left_only'
                ].iloc[:, :-1]
                self._container = new_container
            return self
        else:
            return super().__isub__(other)
