from typing_extensions import get_origin
from .dask_helpers import (
    DaskContextManager,
    convert_type_to_pandas_dtype,
    convert_types_to_pandas_dtype,
    timeit,
    try_to_infer_type_of_operation,
)

import logging
import re
import types
import uuid
from typing import Tuple, Iterable

import dask.dataframe as dd
import numpy as np
from sqlalchemy import (
    and_,
    column,
    func,
    literal,
    literal_column,
    select,
    text,
)
from sqlalchemy.sql import FromClause, except_, intersect, table, union

import pandas as pd

from ...config import config
from ...type_system import Unknown, get_args, infer_type, is_parameterized
from . import abstract as abc

LOG = logging.getLogger(__name__)


class RelationalAlgebraStringExpression(str):
    """
    RelationalAlgebraStringExpression which replaces some pandas/python
    operators by SQL compatible ones.
    """

    def __new__(cls, content):
        instance = super().__new__(cls, content)
        instance = super().__new__(cls, cls._match_operations(str(instance)))
        return instance

    @classmethod
    def _match_operations(cls, content):
        clean_content = cls._match_not_equal(content)
        clean_content = cls._match_power(clean_content)
        clean_content = cls._match_log(clean_content)
        return clean_content

    @staticmethod
    def _match_not_equal(content):
        """
        Replace != by <>
        """
        return re.sub("!=", "<>", content)

    @staticmethod
    def _match_power(content):
        """
        Replace (d ** 2) by POWER(d, 2)
        """
        p = re.compile(r"\(?([^\*]+)\*\*([^\*\)]+)\)?")
        m = p.match(content)
        if m:
            return f"POWER({m.group(1)}, {m.group(2)})"
        return content

    @staticmethod
    def _match_log(content):
        """
        Replace log(x) by ln(x)
        """
        p = re.compile(r"log\((.+)\)")
        m = p.match(content)
        if m:
            return f"ln({m.group(1)})"
        return content

    def __repr__(self):
        return "{}{{ {} }}".format(self.__class__.__name__, super().__repr__())


def _new_name(prefix="table_"):
    if config["RAS"].get("tableIds", "uuid") == "uuid":
        return prefix + str(uuid.uuid4()).replace("-", "_")
    else:
        return prefix + next(DaskContextManager._id_generator_)


class DaskRelationalAlgebraBaseSet:
    """
    Base class for RelationalAlgebraSets relying on a Dask-SQL backend.
    This class defines no RA operations but has all the logic of creating /
    iterating / fetching of items in the sets.
    """

    _count = None
    _is_empty = None
    _table_name = None
    _table = None
    _container = None
    _init_columns = None
    row_types = None

    def __init__(self, iterable=None, columns=None):
        self._init_columns = columns
        if isinstance(iterable, DaskRelationalAlgebraBaseSet):
            if (
                columns is None
                or len(columns) == 0
                or columns == iterable.columns
            ):
                self._init_from(iterable)
            else:
                self._init_from_and_rename(iterable, columns)
        elif iterable is not None:
            self._create_insert_table(iterable, columns)

    def _init_from(self, other):
        self._table_name = other._table_name
        self._container = other._container
        self._table = other._table
        self._count = other._count
        self._is_empty = other._is_empty
        self._init_columns = other._init_columns
        self.row_types = other.row_types

    def _init_from_and_rename(self, other, columns):
        if other._table is not None:
            query = select(
                *[
                    c.label(str(nc))
                    for c, nc in zip(other.sql_columns, columns)
                ]
            ).select_from(other._table)
            self._table = query.subquery()
        self._is_empty = other._is_empty
        self._count = other._count
        if other.row_types is not None:
            self.row_types = other.row_types.rename(
                {other.row_types.index[i]: c for i, c in enumerate(columns)}
            )

    def _create_insert_table(self, data, columns=None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=columns)
        elif columns is not None:
            data.columns = list(columns)
        data.columns = data.columns.astype(str)
        data = data.drop_duplicates()
        if len(data) > 0 and len(data.columns) > 0:
            # partitions should not be too small, yet fit nicely in memory.
            # The amount of RAM available on the machine should be greater
            # than nb of core x partition size.
            # Here we create partitions of less than 500mb based on the
            # original df size.
            df_size = data.memory_usage(deep=True).sum() / (1 << 20)
            npartitions = 1 + int(df_size) // 500
            LOG.info(
                "Creating dask dataframe with {} partitions"
                " from {:0.2f} Mb pandas df.".format(npartitions, df_size)
            )
            ddf = dd.from_pandas(data, npartitions=npartitions)
            self._set_container(ddf, persist=True)
            self.row_types = pd.Series(
                get_args(infer_type(next(data.itertuples(index=False)))),
                index=data.columns,
            )
        self._count = len(data)
        self._is_empty = self._count == 0

    @timeit
    def _set_container(self, ddf, persist=True, prefix="table_", sql=None):
        self._container = ddf
        if persist:
            # Persist triggers an evaluation of the dask dataframe task-graph.
            # This evaluation is asynchronous (if using an asynchronous scheduler).
            # It will return a new dataframe with a shallow graph.
            # See https://distributed.dask.org/en/latest/memory.html#persisting-collections
            # Since we're calling persist here, make sure to pass persist=False to
            # create_table method to not call it twice.
            self._container = self._container.persist()
            self._table_name = _new_name(prefix=prefix)
            DaskContextManager.get_context().create_table(
                self._table_name, ddf, persist=False, sql=sql
            )
            self._table = table(
                self._table_name, *[column(c) for c in ddf.columns]
            )

    @property
    def set_row_type(self):
        """
        Return typing info for this set.
        """
        if self.arity > 0:
            if self.row_types is not None:
                return Tuple[tuple(self.row_types.values)]
            return Tuple[tuple(Unknown for _ in self.columns)]
        return Tuple

    @property
    def dtypes(self, default_type=np.float64):
        """
        return a pandas dtypes array containing numpy dtypes for each column.
        """
        if self.row_types is None:
            return None
        return convert_types_to_pandas_dtype(self.row_types, default_type)

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        if self._table is None:
            return [] if self._init_columns is None else self._init_columns
        return self._table.c.keys()

    @property
    def sql_columns(self):
        if self._table is None:
            return {}
        return self._table.c

    @property
    def container(self):
        """
        Accessing the container will evaluate the SQL query representing this set and
        persist the results in Dask.
        """
        if self._container is None:
            if self._table is not None and self.arity > 0:
                q = select(self._table)
                ddf = DaskContextManager.sql(q)
                query = DaskContextManager.compile_query(q)
                self._set_container(
                    ddf, persist=True, prefix="table_as_", sql=query
                )
        return self._container

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        output._is_empty = False
        return output

    @classmethod
    def dum(cls):
        output = cls()
        output._count = 0
        return output

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._init_from(other)
        return output

    def copy(self):
        if self.is_dee():
            return self.dee()
        elif self.is_dum():
            return self.dum()
        return type(self).create_view_from(self)

    def _create_view_from_query(
        self,
        query: FromClause,
        row_types: pd.Series = None,
        is_empty: bool = None,
    ):
        """
        This method is called to create a new set whenever a relational
        algebra operation is performed on this set.

        Parameters
        ----------
        query : FromClause
            The sql query for the new set
        row_types : pd.Series, optional
            typings row type information for the newset,
            if None self.row_types will be used
        is_empty : bool, optional
            is the new set empty, by default None

        Returns
        -------
        DaskRelationalAlgebraBaseSet
            A new set corresponding to the query
        """
        output = type(self)()
        output._table = query.subquery()
        output._container = None
        output._is_empty = is_empty
        if row_types is None:
            row_types = self.row_types
        output.row_types = row_types
        return output

    def is_empty(self):
        if self._is_empty is None:
            if self._count is not None:
                self._is_empty = self._count == 0
            else:
                self._is_empty = self.fetch_one() is None
        return self._is_empty

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    def __len__(self):
        if self._count is None:
            if self.container is None:
                self._count = 0
            else:
                self._count = len(self.container.drop_duplicates())
        return self._count

    def __contains__(self, element):
        if self.arity == 0:
            return False
        element = self._normalise_element(element)
        query = select(self._table)
        for c, v in element.items():
            query = query.where(self.sql_columns.get(c) == v)
        res = DaskContextManager.sql(query).head(1, npartitions=-1)
        return len(res) > 0

    def _normalise_element(self, element):
        if isinstance(element, dict):
            pass
        elif hasattr(element, "__iter__") and not isinstance(element, str):
            element = dict(zip(self.columns, element))
        else:
            element = dict(zip(self.columns, (element,)))
        return element

    def itervalues(self):
        if self.is_dee():
            return iter([tuple()])
        else:
            return iter(self._fetchall().itertuples(name=None, index=False))

    def __iter__(self):
        return self._do__iter__(named=False)

    def _do__iter__(self, named=False):
        if self.is_dee():
            return iter([tuple()])
        if named:
            try:
                return self._fetchall().itertuples(name="tuple", index=False)
            except ValueError:
                # Invalid column names for namedtuple, just return unnamed tuples
                pass
        return self._fetchall().itertuples(name=None, index=False)

    def as_numpy_array(self):
        return self._fetchall().to_numpy()

    def as_pandas_dataframe(self):
        df = self._fetchall()
        try:
            df.columns = df.columns.astype(int)
        except TypeError:
            pass
        return df

    def _fetchall(self):
        if self.container is None:
            if self._count == 1:
                return pd.DataFrame([()])
            else:
                return pd.DataFrame([])
        df = self.container.compute()
        return df

    def fetch_one(self):
        return self._fetch_one(named=False)

    def _fetch_one(self, named=False):
        if self.container is None:
            if self._count == 1:
                return tuple()
            return None
        if not hasattr(self, "_one_row"):
            name = "tuple" if named else None
            try:
                self._one_row = next(
                    self.container.head(1).itertuples(name=name, index=False)
                )
            except StopIteration:
                self._one_row = None
        return self._one_row

    def __eq__(self, other):
        if isinstance(other, DaskRelationalAlgebraBaseSet):
            if self.is_dee() or other.is_dee():
                res = self.is_dee() and other.is_dee()
            elif self.is_dum() or other.is_dum():
                res = self.is_dum() and other.is_dum()
            elif (
                self._table_name is not None
                and self._table_name == other._table_name
            ):
                res = True
            elif not self._equal_sets_structure(other):
                res = False
            else:
                scont = self.container
                ocont = other.container
                if (
                    scont is None
                    or len(scont) == 0
                    or ocont is None
                    or len(ocont) == 0
                ):
                    res = (scont is None or len(scont) == 0) and (
                        ocont is None or len(ocont) == 0
                    )
                else:
                    intersection_dups = (
                        scont.merge(ocont, how="outer", indicator=True)
                        .iloc[:, -1]
                        .compute()
                    )
                    res = (intersection_dups == "both").all()
            return res
        else:
            return super().__eq__(other)

    def _equal_sets_structure(self, other):
        if isinstance(self, abc.NamedRelationalAlgebraFrozenSet) or isinstance(
            other, abc.NamedRelationalAlgebraFrozenSet
        ):
            return set(self.columns) == set(other.columns)
        return self.arity == other.arity

    def __repr__(self):
        t = self._table
        return "{}({})".format(type(self), t)

    def __hash__(self):
        if self._table is None:
            return hash((tuple(), None))
        return hash((tuple(self.columns), self.as_numpy_array().tobytes()))


class RelationalAlgebraFrozenSet(
    DaskRelationalAlgebraBaseSet, abc.RelationalAlgebraFrozenSet
):
    def __init__(self, iterable=None, columns=None):
        super().__init__(iterable, columns=columns)

    def selection(self, select_criteria):
        if self._table is None:
            return self.copy()

        query = select(self._table)
        if callable(select_criteria):
            lambda_name = _new_name("lambda")
            params = [(c, self.dtypes[c]) for c in self.columns]
            DaskContextManager.register_function(
                select_criteria, lambda_name, params, np.bool8, True
            )
            f_ = getattr(func, lambda_name)
            query = query.where(f_(*self.sql_columns))
        elif isinstance(select_criteria, RelationalAlgebraStringExpression):
            # replace == used in python by = used in SQL
            query = query.where(text(re.sub("==", "=", str(select_criteria))))
        else:
            for k, v in select_criteria.items():
                if callable(v):
                    lambda_name = _new_name("lambda")
                    c_ = self.sql_columns.get(str(k))
                    DaskContextManager.register_function(
                        v,
                        lambda_name,
                        [(str(k), self.dtypes[str(k)])],
                        np.bool8,
                        False,
                    )
                    f_ = getattr(func, lambda_name)
                    query = query.where(f_(c_))
                elif isinstance(
                    select_criteria, RelationalAlgebraStringExpression
                ):
                    query = query.where(text(re.sub("==", "=", str(v))))
                else:
                    query = query.where(self.sql_columns.get(str(k)) == v)
        return self._create_view_from_query(query)

    def selection_columns(self, select_criteria):
        if self._table is None:
            return self.copy()
        query = select(*self.sql_columns).select_from(self._table)
        for k, v in select_criteria.items():
            query = query.where(
                self.sql_columns.get(str(k)) == self.sql_columns.get(str(v))
            )
        return self._create_view_from_query(query)

    def equijoin(self, other, join_indices):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        # Create an alias on the other table's name if we're joining on
        # the same table.
        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()

        join_cols = list(self.sql_columns) + [
            ot.c.get(str(i)).label(str(i + self.arity))
            for i in range(other.arity)
        ]
        query = select(*join_cols)

        if join_indices is not None and len(join_indices) > 0:
            on_clause = and_(
                *[
                    self.sql_columns.get(str(i)) == ot.c.get(str(j))
                    for i, j in join_indices
                ]
            )
            query = query.select_from(self._table.join(ot, on_clause))
        row_types = pd.concat(
            [
                self.row_types,
                other.row_types.rename(
                    {str(i): str(i + self.arity) for i in range(other.arity)}
                ),
            ]
        )
        return self._create_view_from_query(query, row_types=row_types)

    def cross_product(self, other):
        return self.equijoin(other, join_indices=None)

    def groupby(self, columns):
        if self.container is not None:
            if isinstance(columns, str) or not isinstance(columns, Iterable):
                columns = [columns]
            columns = list(map(str, columns))
            df = self.container.compute()
            for g_id, group in df.groupby(by=columns):
                group_set = type(self)(iterable=group)
                yield g_id, group_set

    def projection(self, *columns):
        return self._do_projection(*columns, reindex=True)

    def _do_projection(self, *columns, reindex=True):
        if len(columns) == 0 or self.arity == 0:
            new = type(self)()
            if len(self) > 0:
                new._count = 1
            return new

        if self._table is None:
            return type(self)(columns=columns, iterable=[])

        row_types = self.row_types[[str(c) for c in columns]]
        if reindex:
            proj_columns = [
                self.sql_columns.get(str(c)).label(str(i))
                for i, c in enumerate(columns)
            ]
            row_types.index = [str(i) for i in range(len(columns))]
        else:
            proj_columns = [self.sql_columns.get(str(c)) for c in columns]
        query = select(proj_columns).select_from(self._table)
        if len(set(proj_columns)) != len(set(self.sql_columns)):
            query = query.distinct()

        return self._create_view_from_query(
            query, row_types=row_types, is_empty=self._is_empty
        )

    def _do_set_operation(self, other, sql_operator):
        if not self._equal_sets_structure(other):
            raise ValueError(
                "Relational algebra set operators can only be used on sets"
                " with same columns."
            )

        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()
        query = sql_operator(
            select(self._table),
            select([ot.c.get(c) for c in self.columns]).select_from(ot),
        )
        if sql_operator is union:
            query = select(query.subquery()).distinct()
        return self._create_view_from_query(query)

    def __and__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self._table is None or other._table is None:
            return type(self)(columns=self.columns, iterable=[])
        return self._do_set_operation(other, intersect)

    def __or__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__or__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        if self._table is None:
            return other.copy()
        elif other._table is None:
            return self.copy()
        return self._do_set_operation(other, union)

    def __sub__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        if self.arity == 0 or other.arity == 0:
            if self.is_dee() and other.is_dee():
                return self.dum()
            return self.copy()
        if self._table is None or other._table is None:
            return self.copy()
        return self._do_set_operation(other, except_)


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):
    def __init__(self, columns=None, iterable=None):
        if isinstance(columns, RelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        self._check_for_duplicated_columns(columns)
        super().__init__(iterable, columns)

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if columns is not None and len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    @property
    def columns(self):
        return tuple(super().columns)

    def fetch_one(self):
        return super()._fetch_one(named=True)

    def __iter__(self):
        return super()._do__iter__(named=True)

    def projection(self, *columns):
        return super()._do_projection(*columns, reindex=False)

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if len(set(self.columns).intersection(set(other.columns))) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        if self._table is None or other._table is None:
            return self._do_empty_join(other, isouter=False)

        query = select(self._table, other._table)
        return self._create_view_from_query(
            query,
            row_types=self.row_types.append(other.row_types),
            is_empty=self._is_empty,
        )

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self.cross_product(other)

        if self._table is None or other._table is None:
            return self._do_empty_join(other, isouter=False)
        return self._do_join(other, on, isouter=False)

    def left_naturaljoin(self, other):
        """
        Same as naturaljoin with outer=True
        """
        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self
        if self._table is None or other._table is None:
            return self._do_empty_join(other, isouter=True)
        return self._do_join(other, on, isouter=True)

    def _do_join(self, other, on, isouter=False):
        """
        Performs the join on the two sets.

        Parameters
        ----------
        other : NamedRelationalAlgebraFrozenSet
            The other set
        on : Iterable[sqlalchemy.Columns]
            The columns to join on
        isouter : bool, optional
            If True, performs a left outer join, by default False

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            The joined set
        """
        # Create an alias on the other table's name if we're joining on
        # the same table.
        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()

        on_clause = and_(
            *[self._table.c.get(col) == ot.c.get(col) for col in on]
        )
        other_cols = list(set(other.columns) - set(self.columns))
        select_cols = [self._table] + [ot.c.get(col) for col in other_cols]
        query = select(*select_cols).select_from(
            self._table.join(ot, on_clause, isouter=isouter)
        )
        row_types = self.row_types.append(other.row_types[other_cols])
        empty = self._is_empty if isouter else None
        return self._create_view_from_query(
            query, row_types=row_types, is_empty=empty
        )

    def _do_empty_join(self, other, isouter=False):
        """
        Perform a join when self._table or other._table is None.
        Result will be an empty set with combined columns unless
        it's a left join and self._table is not None.
        """
        other_cols = tuple(set(other.columns) - set(self.columns))
        new_cols = self.columns + other_cols
        if not isouter or self._table is None:
            return type(self)(columns=new_cols, iterable=[])

        projections = {
            col: abc.RelationalAlgebraColumnStr(col) for col in self.columns
        }
        for col in other_cols:
            projections[col] = None
        return self.extended_projection(projections)

    def replace_null(self, dst_column, value):
        columns = self.columns
        if len(columns) == 0 or self.arity == 0:
            new = type(self)()
            if len(self) > 0:
                new._count = 1
            return new

        if self._table is None:
            return type(self)(columns=columns, iterable=[])

        sql_dst_column = self.sql_columns.get(dst_column)
        columns_without_change = [
            c for c in self.sql_columns if c != sql_dst_column
        ]
        if isinstance(value, RelationalAlgebraStringExpression):
            if str(value) != str(dst_column):
                value = literal_column(value)
            else:
                value = self.sql_columns.get(str(value))
        elif isinstance(value, abc.RelationalAlgebraColumn):
            value = self.sql_columns.get(str(value))
        else:
            value = literal(value)

        proj_column = func.coalesce(sql_dst_column, value).label(dst_column)
        query = (
            select(
                columns_without_change +
                [proj_column]
            )
            .select_from(self._table)
        )

        return self._create_view_from_query(
            query, row_types=self.row_types
        )

    def explode(self, src_column: str):
        """
        Transform each element of a list-like column to a row.
        
        Since explode is not a standard SQL statement but is an operation
        implemented on dask dataframes, this relational algebra operation
        evaluates the dask container for the set on which it is applied, and
        then calls the `explode` method on it.

        Parameters
        ----------
        src_column : str
            The column to explode

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            The set with exploded column
        
        Examples
        --------
        >>> ras = NamedRelationalAlgebraFrozenSet(
        ...        columns=("x", "y", "z"),
        ...        iterable=[
        ...            (5, frozenset({1, 2, 5, 6}), "foo"),
        ...            (10, frozenset({5, 9}), "bar"),
        ...        ])
        >>> ras.explode('y')
            x  y    z
        0   5  1  foo
        1   5  2  foo
        2   5  5  foo
        3   5  6  foo
        4  10  9  bar
        5  10  5  bar
        """
        if self.is_dum() or self.is_dee():
            return self

        if self._table is None:
            return type(self)(columns=self.columns, iterable=[])

        q = select(self._table)
        ddf = DaskContextManager.sql(q)
        exploded_ddf = ddf.explode(src_column)

        output = type(self)(columns=self.columns)
        output._is_empty = self._is_empty
        output._count = None
        col_type = self.row_types[src_column]
        if (
            is_parameterized(col_type)
            and issubclass(get_origin(col_type), Iterable)
            and len(get_args(col_type)) == 1
        ):
            col_type = get_args(col_type)[0]
            if col_type != Unknown:
                try:
                    exploded_ddf[src_column] = exploded_ddf[src_column].astype(
                        col_type
                    )
                except TypeError:
                    LOG.warn(
                        f"Unable to cast exploded column to type {col_type}"
                    )
        else:
            col_type = Unknown
        output.row_types = self.row_types.copy()
        output.row_types[src_column] = col_type
        output._set_container(exploded_ddf, persist=True, prefix="table_as_")
        return output

    def equijoin(self, other, join_indices):
        raise NotImplementedError()

    def rename_column(self, src, dst):
        if src == dst:
            return self
        if dst in self.columns:
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"{dst} is already a column name."
            )
        if self._table is None:
            new_cols = [dst if c == src else c for c in self.columns]
            return type(self)(columns=new_cols, iterable=[])
        query = select(
            *[
                c.label(str(dst)) if c.name == src else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        row_types = self.row_types.rename({str(src): str(dst)})
        return self._create_view_from_query(
            query, row_types=row_types, is_empty=self._is_empty
        )

    def rename_columns(self, renames):
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())

        if self.is_dum():
            return self

        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self.columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )

        if self._table is None:
            new_cols = [
                renames.get(c) if c in renames else c for c in self.columns
            ]
            return type(self)(columns=new_cols, iterable=[])

        query = select(
            *[
                c.label(str(renames.get(c.name))) if c.name in renames else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        row_types = self.row_types.rename(renames)
        return self._create_view_from_query(
            query, row_types=row_types, is_empty=self._is_empty
        )

    def aggregate(self, group_columns, aggregate_function):
        group_columns = list(group_columns)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")
        if self.is_dee():
            raise ValueError(
                "Aggregation on non-empty sets with arity == 0 is unsupported."
            )
        distinct_sub_query = select(self._table).subquery()
        agg_cols, agg_types = self._build_aggregate_functions(
            group_columns, aggregate_function, distinct_sub_query
        )

        if self._table is None:
            new_cols = group_columns + list(col.name for col in agg_cols)
            return type(self)(columns=new_cols, iterable=[])

        groupby = [distinct_sub_query.c.get(str(c)) for c in group_columns]
        query = select(*(groupby + agg_cols)).group_by(*groupby)
        row_types = self.row_types[list(group_columns)].append(agg_types)
        return self._create_view_from_query(
            query, row_types=row_types, is_empty=self._is_empty
        )

    def _build_aggregate_functions(
        self, group_columns, aggregate_function, distinct_view
    ):
        """
        Create the list of aggregated destination columns.
        """
        if isinstance(aggregate_function, dict):
            agg_iter = ((k, k, v) for k, v in aggregate_function.items())
        elif isinstance(aggregate_function, (tuple, list)):
            agg_iter = aggregate_function
        else:
            raise ValueError(
                "Unsupported aggregate_function: {} of type {}".format(
                    aggregate_function, type(aggregate_function)
                )
            )
        un_grouped_cols = [
            c_ for c_ in distinct_view.c if c_.name not in group_columns
        ]
        agg_cols = []
        agg_types = pd.Series(dtype="object")
        for dst, src, f in agg_iter:
            if src in distinct_view.c.keys():
                # call the aggregate function on only one column
                c_ = [distinct_view.c.get(src)]
            else:
                # call the aggregate function on all the non-grouped columns
                c_ = un_grouped_cols
            if isinstance(f, types.BuiltinFunctionType):
                f = f.__name__
            if callable(f):
                lambda_name = _new_name("lambda")
                params = [(c.name, self.dtypes[c.name]) for c in c_]
                rtype = try_to_infer_type_of_operation(f, self.row_types)
                DaskContextManager.register_aggregation(
                    f, lambda_name, params, convert_type_to_pandas_dtype(rtype)
                )
                f_ = getattr(func, lambda_name)
            elif isinstance(f, str):
                if f == "first":
                    # first is registered as a postgresql function which
                    # behaves differently from pandas first.
                    # Use single_value instead.
                    f_ = getattr(func, "single_value")
                    rtype = self.row_types[src]
                elif f == "sum":
                    # sum is problematic since sqlalchemy has it return
                    # NullType so we force the return type to be the that
                    # of the src col.
                    f_ = getattr(func, f)
                    rtype = (
                        self.row_types[src]
                        if self.row_types is not None
                        else Unknown
                    )
                else:
                    f_ = getattr(func, f)
                    rtype = try_to_infer_type_of_operation(f, self.row_types)
            else:
                raise ValueError(
                    f"Aggregate function for {src} needs "
                    "to be callable or a string"
                )
            agg_cols.append(f_(*c_).label(str(dst)))
            agg_types[str(dst)] = rtype
        return agg_cols, agg_types

    def extended_projection(self, eval_expressions):
        if self.is_dee():
            return self._extended_projection_on_dee(eval_expressions)
        elif self._table is None:
            return type(self)(
                columns=list(eval_expressions.keys()), iterable=[]
            )

        proj_columns = []
        row_types = pd.Series(dtype="object")
        col_names = set()
        for dst_column, operation in eval_expressions.items():
            if callable(operation):
                lambda_name = _new_name("lambda")
                params = [(c, self.dtypes[c]) for c in self.columns]
                rtype = try_to_infer_type_of_operation(
                    operation, self.row_types
                )
                DaskContextManager.register_function(
                    operation,
                    lambda_name,
                    params,
                    convert_type_to_pandas_dtype(rtype),
                    True,
                )
                f_ = getattr(func, lambda_name)
                proj_columns.append(
                    f_(*self.sql_columns).label(str(dst_column))
                )
                row_types[str(dst_column)] = rtype
            elif isinstance(operation, RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    proj_columns.append(
                        literal_column(operation).label(str(dst_column))
                    )
                    rtype = try_to_infer_type_of_operation(
                        operation, self.row_types
                    )
                    row_types[str(dst_column)] = rtype
                else:
                    proj_columns.append(self.sql_columns.get(str(operation)))
                    row_types[str(dst_column)] = self.row_types[
                        str(dst_column)
                    ]
                    col_names.add(str(operation))
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                proj_columns.append(
                    self.sql_columns.get(str(operation)).label(str(dst_column))
                )
                row_types[str(dst_column)] = self.row_types[str(operation)]
                col_names.add(str(operation))
            else:
                proj_columns.append(literal(operation).label(str(dst_column)))
                rtype = try_to_infer_type_of_operation(
                    operation, self.row_types
                )
                row_types[str(dst_column)] = rtype

        query = select(proj_columns).select_from(self._table)
        if set(self.columns) != col_names:
            query = query.distinct()
        return self._create_view_from_query(
            query, row_types=row_types, is_empty=self._is_empty
        )

    def _extended_projection_on_dee(self, eval_expressions):
        """
        Extended projection when called on Dee to create set with
        constant values.
        """
        return type(self)(
            columns=eval_expressions.keys(),
            iterable=[eval_expressions.values()],
        )

    def to_unnamed(self):
        if self._table is not None:
            query = select(
                *[c.label(str(i)) for i, c in enumerate(self.sql_columns)]
            ).select_from(self._table)
            row_types = self.row_types.rename(
                {c: str(i) for i, c in enumerate(self.columns)}
            )
            return RelationalAlgebraFrozenSet()._create_view_from_query(
                query, row_types=row_types, is_empty=self._is_empty
            )
        return RelationalAlgebraFrozenSet(
            columns=[str(i) for i in range(self.arity)]
        )

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = self.columns
        columns = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*columns)


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def _update_self_with_ddf(
        self, ddf, _count=None, _is_empty=None, reset_row=False
    ):
        self._set_container(ddf, persist=True)
        self._count = _count
        self._is_empty = _is_empty
        if reset_row and hasattr(self, "_one_row"):
            delattr(self, "_one_row")

    def add(self, value):
        if self.container is None:
            value = [self._normalise_row(value)]
            self._create_insert_table(value, self._init_columns)
        else:
            value = self._normalise_element(value)
            ddf = self.container.append(pd.DataFrame([value], index=[0]))
            ddf = ddf.drop_duplicates()
            self._update_self_with_ddf(ddf, _is_empty=False)

    @staticmethod
    def _normalise_row(value):
        if isinstance(value, tuple):
            pass
        elif hasattr(value, "__iter__"):
            value = tuple(value)
        else:
            value = (value,)
        return value

    def discard(self, value):
        if self.container is not None:
            value = self._normalise_element(value)
            mask = (
                self.container[list(value.keys())] == list(value.values())
            ).all(axis=1)
            ddf = self.container[~mask]
            self._update_self_with_ddf(ddf)

    def __ior__(self, other):
        res = self.__or__(other)
        self._init_from(res)
        return self

    def __isub__(self, other):
        res = self.__sub__(other)
        self._init_from(res)
        return self
