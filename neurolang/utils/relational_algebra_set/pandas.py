from collections import OrderedDict
import logging
from typing import Iterable, Callable, Dict, Union
from uuid import uuid1

import numpy as np
import pandas as pd
from . import abstract as abc
from ..various import log_performance


LOG = logging.getLogger(__name__)


NA = pd.NA


class RelationalAlgebraStringExpression(str):
    def __repr__(self):
        return "{}{{ {} }}".format(self.__class__.__name__, super().__repr__())


class RelationalAlgebraFrozenSet(abc.RelationalAlgebraFrozenSet):
    """
    RelationalAlgebraFrozenSet implementation using in-memory pandas.DataFrame
    as container for the elements.
    """

    def __init__(self, iterable=None, has_duplicates=True):
        """
        Create a new RelationalAlgebraFrozenSet.
        A RelationalAlgebraSet is a set of n elements, each element being
        a tuple of arity k.

        Parameters
        ----------
        iterable : Iterable
            Initial values for the set
        
        has_duplicates : bool
            Indicate wether the iterable has duplicate rows

        Examples
        --------
        Constructing a Set from a list.

        >>> a = [(i % 2, i, i * 2) for i in range(5)]
        >>> ras = RelationalAlgebraFrozenSet(a)
        >>> ras
           0  1  2
        0  0  0  0
        1  1  1  2
        2  0  2  4
        3  1  3  6
        4  0  4  8

        Returns
        -------
        RelationalAlgebraFrozenSet
        """
        self._container = None
        self._might_have_duplicates = has_duplicates
        if iterable is not None:
            if isinstance(iterable, RelationalAlgebraFrozenSet):
                self._container = iterable._container
                self._might_have_duplicates = iterable._might_have_duplicates
            elif isinstance(iterable, pd.DataFrame):
                self._container = iterable.copy()
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

    @classmethod
    def dee(cls):
        output = cls()
        output._container = pd.DataFrame([()])
        return output

    @classmethod
    def dum(cls):
        return cls()

    def is_empty(self):
        return self._container is None or len(self._container) == 0

    def __contains__(self, element):
        element = self._normalise_element(element)
        if self.is_empty() or self.is_dee() or len(element) != self.arity:
            res = False
        else:
            col = True
            for e, c in zip(element, self._container.iteritems()):
                col = col & (c[1] == e)
            res = col.any()
        return res

    @staticmethod
    def _normalise_element(element):
        if isinstance(element, tuple):
            pass
        elif hasattr(element, "__iter__"):
            element = tuple(element)
        else:
            element = (element,)
        return element

    def __iter__(self):
        if self.is_empty():
            values = {}
        elif self.is_dee():
            values = {tuple()}
        else:
            self._drop_duplicates_if_needed()
            values = self.itervalues()
        for v in values:
            yield tuple(v)

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        return next(self._container.itertuples(name=None, index=False))

    def __len__(self):
        if self._container is None:
            return 0
        self._drop_duplicates_if_needed()
        return len(self._container)

    @staticmethod
    def _drop_duplicates(container):
        if len(container) == 0 or len(container.columns) == 0:
            return container
        with log_performance(
            LOG,
            "Dropping duplicates from table %s", (str(container.head(1)),)
        ):
            container = container.drop_duplicates()
        return container

    @property
    def arity(self):
        if self._container is None:
            return 0
        return len(self._container.columns)

    @property
    def columns(self):
        if self._container is not None:
            return self._container.columns
        else:
            return tuple()

    def as_numpy_array(self):
        self._drop_duplicates_if_needed()
        res = self._container.values.view()
        res.setflags(write=False)
        return res

    def as_pandas_dataframe(self):
        self._drop_duplicates_if_needed()
        return self._container

    def _empty_set_same_structure(self):
        return type(self)()

    def projection(self, *columns):
        """
        Projects the set on the given list of columns.

        Examples
        --------
        >>> ras = RelationalAlgebraFrozenSet(
                [(i % 2, i, i * 2) for i in range(5)])
        >>> ras
           0  1  2
        0  0  0  0
        1  1  1  2
        2  0  2  4
        3  1  3  6
        4  0  4  8
        >>> ras.projection(1)
           0
        0  0
        1  1
        2  2
        3  3
        4  4
        """
        if self.is_empty():
            return self._empty_set_same_structure()
        new_container = self._container[list(columns)]
        new_container.columns = pd.RangeIndex(len(columns))
        output = self._empty_set_same_structure()
        output._container = new_container
        if (len(set(columns)) == self.arity):
            output._might_have_duplicates = self._might_have_duplicates
        return output

    def selection(
        self,
        select_criteria: Union[
            Callable,
            RelationalAlgebraStringExpression,
            Dict[int, Union[int, Callable]],
        ],
    ):
        """
        Select the elements based on the given selet_criteria.
        The select_criteria may be a callable function, a string expression
        or a Dict of columns -> value

        Parameters
        ----------
        select_criteria : Union[Callable, RelationalAlgebraStringExpression,
        Dict[int, Union[int, Callable]]]
            selection criteria

        Returns
        -------
        RelationalAlgebraFrozenSet
            A RelationalAlgebraFrozenSet with elements matching the criteria.
        """
        if self.is_empty():
            return self._empty_set_same_structure()

        if callable(select_criteria):
            ix = self._container.apply(select_criteria, axis=1)
        elif isinstance(
            select_criteria, RelationalAlgebraStringExpression
        ):
            ix = self._container.eval(select_criteria)
        else:
            ix = self._selection_dict(select_criteria)
        ix = ix.astype(bool)
        new_container = self._container[ix]

        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def _selection_dict(self, select_criteria):
        """
        Performs element-wise logical AND on the selection vector returned
        for each item in the select_criteria
        """
        it = iter(select_criteria.items())
        col, value = next(it)
        if callable(value):
            selector = self._container[col].apply(value)
        else:
            selector = self._container[col] == value
        ix = selector
        for col, value in it:
            if callable(value):
                selector = self._container[col].apply(value)
            else:
                selector = self._container[col] == value
            ix &= selector
        return ix

    def selection_columns(self, select_criteria: Dict[int, int]):
        """
        Select the elements where column pairs indicated as key, value
        items of the select_criteria are equal.
        The select_criteria must be a Dict of Int -> Int.

        Examples
        --------
        Select the elements where col0 == col1 and col1 == col2
        >>> ras = RelationalAlgebraFrozenSet(
                [(i % 2, i, i * 2) for i in range(5)])
        >>> ras.selection_columns({0:1, 1: 2})
           0  1  2
        0  0  0  0

        Parameters
        ----------
        select_criteria : Dict [Int, Int]
            selection criteria

        Returns
        -------
        RelationalAlgebraFrozenSet
            A RelationalAlgebraFrozenSet with elements matching the criteria.
        """
        if self.is_empty():
            return self._empty_set_same_structure()
        it = iter(select_criteria.items())
        col1, col2 = next(it)
        ix = self._container[col1] == self._container[col2]
        for col1, col2 in it:
            ix &= self._container[col1] == self._container[col2]

        new_container = self._container[ix]
        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def equijoin(self, other, join_indices):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self.is_empty() or other.is_empty():
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
            self._might_have_duplicates | other._might_have_duplicates
        )
        return output

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self.is_empty() or other.is_empty():
            return self._empty_set_same_structure()
        left = self._container.copy(deep=False)
        right = other._container.copy(deep=False)
        tmpcol = str(uuid1())
        left[tmpcol] = 1
        right[tmpcol] = 1
        col_types = self._extract_unified_column_types(other)
        new_container = pd.merge(
            left.astype(col_types),
            right.astype(col_types),
            on=tmpcol
        )
        del new_container[tmpcol]
        new_container.columns = range(new_container.shape[1])
        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = (
            self._might_have_duplicates | other._might_have_duplicates
        )
        return output

    def _extract_unified_column_types(self, other):
        col_types = dict()
        for col in self._container.columns.intersection(
            other._container.columns
        ):
            s_dtype = self._container[col].dtype
            o_dtype = other._container[col].dtype
            if s_dtype != o_dtype:
                if isinstance(s_dtype, pd.core.arrays.numeric.NumericDtype):
                    col_types[col] = s_dtype
                elif isinstance(o_dtype, pd.core.arrays.numeric.NumericDtype):
                    col_types[col] = o_dtype
        return col_types

    def copy(self):
        output = self._empty_set_same_structure()
        if len(self) > 0:
            output._container = self._container.copy()
            output._might_have_duplicates = self._might_have_duplicates
        return output

    def __repr__(self):
        if self.is_empty():
            return "{}"
        return repr(self._container.reset_index().drop("index", axis=1))

    def __sub__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if (
                self.arity > 0 and other.arity > 0
            ) and self.arity != other.arity:
                raise ValueError(
                    "Relational algebra set operators can only"
                    " be used on sets with same columns."
                )
            if self.is_empty() or other.is_empty():
                return self.copy()
            if self.is_dee():
                if other.is_dee():
                    return self.dum()
                return self.dee()
        return super().__sub__(other)

    def __or__(self, other):
        if self is other:
            return self.copy()
        elif isinstance(other, RelationalAlgebraFrozenSet):
            res = self._dee_dum_sum(other)
            if res is not None:
                return res
            ocont = other._container
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
        if self is other:
            return self.copy()
        if isinstance(other, RelationalAlgebraFrozenSet):
            res = self._dee_dum_product(other)
            if res is not None:
                return res
            new_container = pd.merge(
                left=self._container,
                right=other._container,
                how="inner",
            )
            output = self._empty_set_same_structure()
            output._container = new_container
            output._might_have_duplicates = (
                self._might_have_duplicates | other._might_have_duplicates
            )
            return output
        else:
            return super().__and__(other)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            scont = self._container
            ocont = other._container
            if self.is_empty() and other.is_empty():
                res = True
            elif self.arity != other.arity:
                res = False
            elif self.arity == 0 and other.arity == 0:
                res = self.is_dee() and other.is_dee()
            elif scont is not None and ocont is not None:
                intersection_dups = scont.merge(
                    ocont, how="outer", indicator=True
                ).iloc[:, -1]
                res = (intersection_dups == "both").all()
            else:
                res = False
            return res
        else:
            return super().__eq__(other)

    def groupby(self, columns):
        if not self.is_empty():
            if not isinstance(columns, Iterable):
                columns = [columns]
            for g_id, group in self._container.groupby(
                by=list(columns), dropna=False
            ):
                group_set = self._empty_set_same_structure()
                group_set._container = group
                yield g_id, group_set

    def itervalues(self):
        if self.is_empty():
            return iter([])
        else:
            return iter(self._container.itertuples(name=None, index=False))

    def __hash__(self):
        if self._container is None:
            return hash((tuple(), None))
        self._drop_duplicates_if_needed()
        v = self._container.values
        v.flags.writeable = False
        return hash((tuple(self._container.columns), v.data.tobytes()))


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):
    def __init__(self, columns, iterable=None, has_duplicates=True):
        if isinstance(columns, NamedRelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        # ensure there is no duplicated column
        self._check_for_duplicated_columns(columns)
        self._columns = tuple(columns)
        self._might_have_duplicates = has_duplicates
        if iterable is None:
            iterable = []

        if isinstance(iterable, NamedRelationalAlgebraFrozenSet):
            self._initialize_from_named_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        elif isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_unnamed_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        elif isinstance(iterable, pd.DataFrame):
            self._container = iterable.copy()
            self._container.columns = self._columns
        else:
            self._container = pd.DataFrame(iterable, columns=self._columns)

    def _initialize_from_named_ra_set(self, other):
        if not other.is_dum() and self.arity != other.arity:
            raise ValueError("Relations must have the same arity")

        if not other.is_empty():
            self._container = other._container[list(other.columns)].copy(
                deep=False
            )
        else:
            self._container = pd.DataFrame(columns=self._columns)

    def _initialize_from_unnamed_ra_set(self, other):
        if other.is_empty():
            self._container = pd.DataFrame(columns=self._columns)
        elif other._container is None:
            self._container = pd.DataFrame(list(other), columns=self._columns)
        else:
            if self.arity != other.arity:
                raise ValueError("Relations must have the same arity")
            self._container = other._container.copy(deep=False)
            self._container.columns = self._columns

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls(columns=tuple())
        output._container = other._container
        output._columns = other._columns
        output._might_have_duplicates = other._might_have_duplicates
        return output

    def _empty_set_same_structure(self):
        return type(self)(self.columns)

    @classmethod
    def dee(cls):
        output = cls(())
        output._container = pd.DataFrame([()])
        return output

    @classmethod
    def dum(cls):
        return cls(())

    def _light_init_same_structure(
        self, container, might_have_duplicates=True, columns=None
    ):
        if columns is None:
            columns = self.columns
        output = type(self)(columns)
        output._container = container
        if might_have_duplicates:
            LOG.info("Table %s might have duplicates", container.head(1))
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
        return super().__contains__(element)

    def projection(self, *columns):
        if self.is_empty():
            return type(self)(columns)
        if self.arity == 0:
            return self
        might_have_duplicates = (
            (len(set(columns)) < self.arity) or
            self._might_have_duplicates
        )
        new_container = self._container[list(columns)]
        return self._light_init_same_structure(
            new_container, might_have_duplicates=might_have_duplicates,
            columns=columns
        )

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = list(self.columns)
        columns = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*columns)

    def equijoin(self, other, join_indices):
        raise NotImplementedError()

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        on = [c for c in self.columns if c in other.columns]

        if len(on) == 0:
            return self.cross_product(other)

        new_columns = self.columns + tuple(
            c for c in other.columns if c not in self.columns
        )

        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()
        col_types = self._extract_unified_column_types(other)
        with log_performance(
            LOG, "About to natural join 2 tables of %s and %s",
            (self._container.shape, other._container.shape)
        ):
            new_container = (
                self._container
                .astype(col_types)
                .merge(other._container.astype(col_types))
            )
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates | other._might_have_duplicates
            ),
            columns=new_columns,
        )

    def left_naturaljoin(self, other):
        if self.is_dee():
            return self

        on = [c for c in self.columns if c in other.columns]

        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        if len(on) == 0:
            if other.is_empty():
                new_container = self._container.copy()
                for c in other.columns:
                    new_container[c] = pd.NA
            else:
                return self.cross_product(other)
        else:
            col_types = self._extract_unified_column_types(other)
            new_container = (
                self._container.astype(col_types)
                .merge(
                    other._container.astype(col_types),
                    how="left"
                )
            )

        new_columns = self.columns + tuple(
            c for c in other.columns if c not in self.columns
        )
        self._keep_column_types_after_join(
            other, new_container, new_columns, check_for_na=True
        )

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates | other._might_have_duplicates
            ),
            columns=new_columns,
        )

    def _keep_column_types_after_join(
        self, other, new_container, new_columns,
        check_for_na=False
    ):
        new_columns_types = {}
        for col in new_columns:
            if col in self.columns:
                new_columns_types[col] = self._container[col].dtype
            else:
                new_columns_types[col] = other._container[col].dtype
        self._keep_column_types_list(
            new_container, new_columns_types, check_for_na=check_for_na
        )

    def replace_null(self, column, value):
        if self._container is None:
            return self.copy()

        new_container = self._container.fillna({column: value})
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=self.columns,
        )

    def explode(self, src_column, dst_columns):
        if self._container is None:
            return self.copy()

        if dst_columns == src_column:
            # 1. replace original column by exploded one
            new_container = self._container.explode(
                src_column, ignore_index=True
            )
            new_columns = self.columns
        elif not isinstance(dst_columns, tuple):
            # 2. add new column with exploded values
            new_container = self._container.assign(
                **{dst_columns: self._container[src_column]}
            ).explode(dst_columns, ignore_index=True)
            new_columns = self.columns + (dst_columns,)
        else:
            # 3. explode values into multiple columns
            new_container = self._container.assign(
                **{dst_columns[-1]: self._container[src_column]}
            ).explode(dst_columns[-1], ignore_index=True)
            for c, v in zip(dst_columns, zip(*new_container[dst_columns[-1]])):
                new_container[c] = v
            new_columns = self.columns + dst_columns
            new_container = new_container[list(new_columns)]

        return self._light_init_same_structure(
            new_container.convert_dtypes(),
            might_have_duplicates=True,
            columns=new_columns,
        )

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res.copy()
        if len(self._container.columns.intersection(other.columns)) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        new_columns = self.columns + other.columns
        if self.is_empty() or other.is_empty():
            res = type(self)(new_columns)
        else:
            col_types = self._extract_unified_column_types(other)
            left = self._container.copy(deep=False).astype(col_types)
            right = other._container.copy(deep=False).astype(col_types)
            tmpcol = str(uuid1())
            left[tmpcol] = 1
            right[tmpcol] = 1
            new_container = pd.merge(left, right, on=tmpcol)
            del new_container[tmpcol]
            new_container.columns = tuple(self._container.columns) + tuple(
                other._container.columns
            )
            res = self._light_init_same_structure(
                new_container,
                might_have_duplicates=(
                    self._might_have_duplicates | other._might_have_duplicates
                ),
                columns=new_columns,
            )
        return res

    def rename_column(self, src, dst):
        if self.is_dum():
            return self
        if src not in self._columns:
            raise ValueError(f"{src} not in columns")
        if src == dst:
            return self
        if dst in self._columns:
            raise ValueError(f"{dst} cannot be in the columns")
        src_idx = self._columns.index(src)
        new_columns = (
            self._columns[:src_idx] + (dst,) + self._columns[src_idx + 1 :]
        )
        new_container = self._container.rename(columns={src: dst})
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=new_columns,
        )

    def rename_columns(self, renames):
        if self.is_dum():
            return self
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self._columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )
        new_columns = tuple(renames.get(col, col) for col in self._columns)
        new_container = self._container.rename(columns=renames)
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=new_columns,
        )

    def __eq__(self, other):
        scont = self._container
        ocont = other._container
        if len(scont.columns.symmetric_difference(ocont.columns)) > 0:
            res = False
        elif len(scont) == 0 and len(ocont) == 0:
            res = True
        elif len(scont.columns) == 0 and len(ocont.columns) == 0:
            res = len(scont) > 0 and len(ocont) > 0
        else:
            intersection_dups = scont.merge(
                ocont, how="outer", indicator=True
            ).iloc[:, -1]
            res = (intersection_dups == "both").all()
        return res

    def groupby(self, columns):
        if self.is_empty():
            return
        for g_id, group in self._container.groupby(
            by=list(columns), dropna=False
        ):
            group_set = self._light_init_same_structure(
                group,
                might_have_duplicates=self._might_have_duplicates,
                columns=self.columns,
            )
            yield g_id, group_set

    def aggregate(self, group_columns, aggregate_function):
        group_columns = list(group_columns)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")
        if self.is_dee():
            raise ValueError(
                "Aggregation on non-empty sets with arity == 0 is unsupported."
            )
        if self.is_empty():
            if isinstance(aggregate_function, dict):
                agg_columns = list(aggregate_function.keys())
            elif isinstance(aggregate_function, (tuple, list)):
                agg_columns = list(col for col, _, _ in aggregate_function)
            return NamedRelationalAlgebraFrozenSet(
                columns=group_columns + agg_columns
            )
        self._drop_duplicates_if_needed()
        if len(group_columns) > 0:
            groups = self._container.groupby(group_columns, dropna=False)
        else:
            groups = self._container.groupby(lambda x: 0, dropna=False)

        aggs, aggs_multi_column = self._classify_aggregations(
            group_columns, aggregate_function
        )

        new_containers = []
        if len(aggs) > 0:
            new_containers.append(groups.agg(**aggs))

        for dst, fun in aggs_multi_column.items():
            new_col = (
                self._apply_aggregation(groups, fun)
                .rename(dst)
                .to_frame()
            )
            new_containers.append(new_col)

        new_container = pd.concat(new_containers, axis=1)

        if len(group_columns) > 0:
            new_container = new_container.reset_index()

        self._keep_column_types(
            new_container, set(aggs) | set(aggs_multi_column)
        )

        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=False,
            columns=list(new_container.columns),
        )
        return output

    @staticmethod
    def _apply_aggregation(groups, fun):
        fun_dict = {
            len: 'count',
            max: 'max',
            min: 'min',
            sum: 'sum'
        }
        if fun in fun_dict:
            res = getattr(groups, fun_dict[fun])()
            res = res.iloc[:, 0]
        else:
            res = groups.apply(fun)
        return res

    def _keep_column_types(self, new_container, skip=None, check_for_na=True):
        column_types = {
            col: self._container[col].dtype
            for col in self._container.columns
        }

        return self._keep_column_types_list(
            new_container, column_types,
            skip=skip, check_for_na=check_for_na
        )

    def _keep_column_types_list(
        self, new_container, column_types,
        skip=None, check_for_na=False
    ):
        if self.is_empty():
            return

        if skip is None:
            skip = {}
        for col in new_container.columns:
            if col in skip:
                continue
            if (
                col in column_types
                and new_container[col].dtype != column_types[col]
            ):
                if (
                    check_for_na and
                    isinstance(column_types[col], np.dtype) and
                    column_types[col].kind == 'i' and
                    new_container[col].hasnans
                ):
                    column_type = getattr(
                        pd,
                        f"Int{column_types[col].itemsize * 8}Dtype"
                    )()
                else:
                    column_type = column_types[col]
                new_container[col] = new_container[col].astype(
                    column_type
                )

    def _classify_aggregations(self, group_columns, aggregate_function):
        aggs = OrderedDict()
        aggs_multi_columns = OrderedDict()
        if isinstance(aggregate_function, dict):
            arg_iterable = (
                (k, k if k in self.columns else None, v)
                for k, v in aggregate_function.items()
            )
        elif isinstance(aggregate_function, (tuple, list)):
            arg_iterable = aggregate_function
        else:
            raise ValueError(
                "Unsupported aggregate_function: {} of type {}".format(
                    aggregate_function, type(aggregate_function)
                )
            )

        for dst, src, fun in arg_iterable:
            if dst in group_columns:
                raise ValueError(
                    f"Destination column {dst} can't be part of the grouping"
                )
            if src in self.columns:
                aggs[dst] = pd.NamedAgg(src, fun)
            elif src is None or all(s in self.columns for s in src):
                aggs_multi_columns[dst] = fun
            else:
                raise ValueError(f"Source column {src} not in columns")

        return aggs, aggs_multi_columns

    def extended_projection(self, eval_expressions):
        proj_columns = list(eval_expressions.keys())
        if self.is_empty():
            return NamedRelationalAlgebraFrozenSet(
                columns=proj_columns,
                iterable=[],
            )
        new_container = self._container.copy()
        seen_pure_columns = set()
        for dst_column, operation in eval_expressions.items():
            if isinstance(operation, RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    new_container.eval(
                        "{}={}".format(str(dst_column), str(operation)),
                        engine="python",
                        inplace=True
                    )
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                seen_pure_columns.add(operation)
                new_container[dst_column] = new_container.loc[:, operation]
            elif callable(operation):
                new_container[dst_column] = new_container.apply(
                    operation, axis=1
                )
            else:
                new_container[dst_column] = operation
        new_container = new_container.loc[:, proj_columns]
        might_have_duplicates = not (
            (len(seen_pure_columns) == len(self.columns))
            and not self._might_have_duplicates
        )
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=might_have_duplicates,
            columns=proj_columns,
        )
        return output

    def __iter__(self):
        self._drop_duplicates_if_needed()
        if self.is_dee():
            return iter([tuple()])
        container = self._container[list(self.columns)]
        return container.itertuples(index=False, name="tuple")

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        container = self._container[list(self.columns)]
        return next(container.itertuples(index=False, name="tuple"))

    def to_unnamed(self):
        container = self._container[list(self.columns)].copy()
        container.columns = range(len(container.columns))
        output = RelationalAlgebraFrozenSet()
        output._container = container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def __sub__(self, other):
        if (self.arity > 0 and other.arity > 0) and len(
            self._container.columns.difference(other._container.columns)
        ) > 0:
            raise ValueError(
                "Difference defined only for sets with the same columns"
            )
        if self.is_empty() or other.is_empty():
            return self.copy()
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        col_types = self._extract_unified_column_types(other)

        new_container = (
            self._container
            .astype(col_types)
            .merge(
                other._container.astype(col_types),
                indicator=True, how="left"
            )
        )
        new_container = new_container[
            new_container.iloc[:, -1] == "left_only"
        ].iloc[:, :-1]

        self._keep_column_types(new_container)
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
        )
        return output

    def __or__(self, other):
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        elif set(self.columns) != set(other.columns):
            raise ValueError(
                "Union defined only for sets with the same columns"
            )

        col_types = self._extract_unified_column_types(other)

        new_container = pd.merge(
            left=self._container.astype(col_types),
            right=other._container.astype(col_types),
            how="outer",
        )

        self._keep_column_types(new_container)
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=True,
        )
        return output

    def __and__(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if set(self.columns) != set(other.columns):
            raise ValueError(
                "Union defined only for sets with the same columns"
            )
        if self.is_empty():
            return self.copy()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()
        col_types = self._extract_unified_column_types(other)
        new_container = pd.merge(
            left=self._container.astype(col_types),
            right=other._container.astype(col_types),
            how="inner",
        )
        self._keep_column_types(new_container)
        output = self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates & other._might_have_duplicates
            ),
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


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def add(self, value):
        value = self._normalise_element(value)
        e_hash = hash(value)
        if self.is_empty():
            self._container = pd.DataFrame([value], index=[e_hash])
        else:
            self._container.loc[e_hash] = value

    def discard(self, value):
        if not self.is_empty():
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
            if other.is_empty() or other.arity == 0:
                return self
            if self.is_empty():
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
        if self.is_empty():
            return self
        if isinstance(other, RelationalAlgebraSet):
            if other.is_empty() or other.arity == 0:
                if self.is_empty() or self.arity == 0:
                    self._container = self._container.iloc[:0]
            elif other.arity != self.arity:
                raise ValueError(
                    "Operation only valid for sets with the same arity"
                )
            else:
                new_container = pd.merge(
                    left=self._container,
                    right=other._container,
                    how="left",
                    indicator=True,
                )
                new_container = new_container[
                    new_container.iloc[:, -1] == "left_only"
                ].iloc[:, :-1]
                self._container = new_container
            return self
        else:
            return super().__isub__(other)
