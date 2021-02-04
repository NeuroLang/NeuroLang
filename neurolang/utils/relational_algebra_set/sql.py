from collections import namedtuple
from collections.abc import Iterable

from neurolang.type_system import Unknown
from typing import Tuple

import logging
import numpy as np
import types
from neurolang.utils.relational_algebra_set.sql_helpers import (
    CreateTableAs,
    CreateView,
    map_dtype_to_sqla,
    SQLAEngineFactory,
)
import uuid

from . import abstract as abc
from ..various import log_performance
import pandas as pd

from sqlalchemy import (
    Table,
    MetaData,
    Index,
    func,
    and_,
    select,
    text,
    tuple_,
    literal_column,
    literal,
)
from sqlalchemy.sql import table, intersect, union, except_

LOG = logging.getLogger(__name__)


class RelationalAlgebraFrozenSet(abc.RelationalAlgebraFrozenSet):

    _is_view = False
    _count = None
    _table_name = None
    _table = None
    _parent_tables = {}

    def __init__(self, iterable=None):
        self._table_name = self._new_name()
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._init_from(iterable)
        else:
            self._create_insert_table(iterable)

    @staticmethod
    def _new_name(prefix="table_"):
        return prefix + str(uuid.uuid4()).replace("-", "_")

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._init_from(other)
        return output

    def _init_from(self, other):
        self._table_name = other._table_name
        self._count = other._count
        self._table = other._table
        self._is_view = other._is_view
        self._parent_tables = other._parent_tables
        if hasattr(other, "_one_row"):
            self._one_row = other._one_row

    def _create_insert_table(self, data):
        """
        Initialise the set with the provided data collection.
        We use pandas to infer datatypes from the data and create
        the appropriate sql statement.
        We then read the table metadata and store it in self._table.

        Parameters
        ----------
        data : Iterable[Any]
            The initial data for the set.
        """
        if data is not None:
            data = pd.DataFrame(data).drop_duplicates()
            if len(data.columns) > 0:
                self._table = self._insert_dataframe_into_db(data)
                self._parent_tables = {self._table}
            self._count = len(data)
        else:
            self._count = 0

    def _insert_dataframe_into_db(self, df):
        column_names_and_types = {
            df.columns[i]: map_dtype_to_sqla(df.iloc[:, i])
            for i in range(len(df.columns))
        }
        df.to_sql(
            self._table_name,
            SQLAEngineFactory.get_engine(),
            index=False,
            dtype=column_names_and_types,
        )
        _table = Table(
            self._table_name,
            MetaData(),
            autoload=True,
            autoload_with=SQLAEngineFactory.get_engine(),
        )
        return _table

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    def is_empty(self):
        if self._count is not None:
            return self._count == 0
        elif self._table is None:
            return True
        else:
            return self.fetch_one() is None

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    @property
    def dummy_row_type(self):
        """
        Return dummy row_type of Tuple[Unknown * arity] to avoid calls
        to the DB.
        TODO: implement this for real.
        """
        if self.arity > 0:
            return Tuple[tuple(Unknown for _ in range(self.arity))]
        return Tuple

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        """
        List of columns as string identifiers.

        Returns
        -------
        Iterable[str]
            Set of column names.
        """
        if self._table is None:
            return []
        return self._table.c.keys()

    @property
    def sql_columns(self):
        """
        List of columns as sqlalchemy.schema.Columns collection.

        Returns
        -------
        Iterable[sqlalchemy.schema.Columns]
            Set of columns.
        """
        if self._table is None:
            return []
        return self._table.c

    def __len__(self):
        """
        Length of the set is equal to distinct values in the view or table.

        Returns
        -------
        int
            length.
        """
        if self._count is None:
            if self._table is not None:
                query = select([func.count()]).select_from(
                    select(self.sql_columns)
                    .select_from(self._table)
                    .distinct()
                )
                with SQLAEngineFactory.get_engine().connect() as conn:
                    res = conn.execute(query).scalar()
                    self._count = res
            else:
                self._count = 0
        return self._count

    def __iter__(self):
        if self.arity > 0 and len(self) > 0:
            query = (
                select(self.sql_columns).select_from(self._table).distinct()
            )
            with SQLAEngineFactory.get_engine().connect() as conn:
                res = conn.execute(query)
                for t in res:
                    yield tuple(t)
        elif self.arity == 0 and len(self) > 0:
            yield tuple()

    def __contains__(self, element):
        """
        Check whether the set contains the element.

        Parameters
        ----------
        element : Any
            the element to check

        Returns
        -------
        bool
            True if the set contains the element.
        """
        if self.arity == 0:
            return False
        element = self._normalise_element(element)
        query = select(self.sql_columns).select_from(self._table).limit(1)
        for c, v in element.items():
            query = query.where(self.sql_columns.get(c) == v)
        with SQLAEngineFactory.get_engine().connect() as conn:
            res = conn.execute(query)
            return res.first() is not None

    def _normalise_element(self, element):
        """
        Returns a dict representation of the element as col -> value.

        Parameters
        ----------
        element : Iterable[Any]
            the element to normalize

        Returns
        -------
        Dict[str, Any]
            the dict reprensentation of the element
        """
        if isinstance(element, dict):
            pass
        elif hasattr(element, "__iter__") and not isinstance(element, str):
            element = dict(zip(self.columns, element))
        else:
            element = dict(zip(self.columns, (element,)))
        return element

    @classmethod
    def create_view_from_query(cls, query, parent_tables, _count=None):
        """
        Create a new RelationalAlgebraFrozenSet backed by an underlying
        VIEW representation.

        Parameters
        ----------
        query : sqlachemy.selectable.select
            View expression.
        parent_tables: Set(RelationalAlgebraFrozenSet)
            Set of parent tables.
        _count: int, optional
            The count of the query


        Returns
        -------
        RelationalAlgebraFrozenSet
            A new set.
        """
        output = cls()
        view_name = cls._new_name("view_")
        view = CreateView(view_name, query)
        with SQLAEngineFactory.get_engine().connect() as conn:
            conn.execute(view)
        t = table(view_name)
        for c in query.c:
            c._make_proxy(t)
        output._table_name = view_name
        output._table = t
        output._is_view = True
        output._parent_tables = parent_tables
        output._count = _count
        return output

    def deep_copy(self):
        """
        Creates a new deepcopy of the set. This creates a new table in the DB
        with the same elements.

        Returns
        -------
        RelationalAlgebraSet
            Same set with new table representation.
        """
        if self.is_dee():
            return self.dee()
        if self._table is None:
            return type(self)()
        else:
            query = select(self.sql_columns)
            output = type(self)()
            new_table = CreateTableAs(output._table_name, query)
            with SQLAEngineFactory.get_engine().connect() as conn:
                conn.execute("PRAGMA optimize;")
                with log_performance(
                    LOG, "Deep copy", end_message="\tDeep copy %2.2fs"
                ):
                    conn.execute(new_table)
                conn.execute("PRAGMA optimize;")
            self._explain_long_query(query)
            output._table = Table(
                output._table_name,
                MetaData(),
                autoload=True,
                autoload_with=SQLAEngineFactory.get_engine(),
            )
            output._is_view = False
            output._parent_tables = {output._table}
            output._count = self._count
            return output

    def selection(self, select_criteria):
        """
        Select elements from the set matching selection criteria.

        Parameters
        ----------
        select_criteria : Union[callable, RelationalAlgebraStringExpression,
        Dict[int, str]]
            selection criteria

        Returns
        -------
        RelationalAlgebraFrozenSet
            The set with elements matching the given criteria
        """
        if self._table is None:
            return self.copy()

        query = select(self.sql_columns).select_from(self._table)
        if callable(select_criteria):
            lambda_name = self._new_name("lambda")
            SQLAEngineFactory.register_function(
                lambda_name,
                len(self.sql_columns),
                select_criteria,
                params=self.sql_columns.values(),
            )
            f_ = getattr(func, lambda_name)
            query = query.where(f_(*self.sql_columns))
        elif isinstance(
            select_criteria, abc.RelationalAlgebraStringExpression
        ):
            query = query.where(text(select_criteria))
        else:
            for k, v in select_criteria.items():
                if callable(v):
                    lambda_name = self._new_name("lambda")
                    c_ = self.sql_columns.get(str(k))
                    SQLAEngineFactory.register_function(
                        lambda_name, 1, v, [c_]
                    )
                    f_ = getattr(func, lambda_name)
                    query = query.where(f_(c_))
                elif isinstance(
                    select_criteria, abc.RelationalAlgebraStringExpression
                ):
                    query = query.where(text(v))
                else:
                    query = query.where(self.sql_columns.get(str(k)) == v)
        return type(self).create_view_from_query(query, self._parent_tables)

    def selection_columns(self, select_criteria):
        """
        Select elements in the set with equal column values.

        Parameters
        ----------
        select_criteria : Dict[Union[int, str], Union[int, str]]
            The dict of column names to check equality on.

        Returns
        -------
        RelationalAlgebraFrozenSet
            The set with selected elements
        """
        if self._table is None:
            return self.copy()
        query = select(self.sql_columns).select_from(self._table)
        for k, v in select_criteria.items():
            query = query.where(
                self.sql_columns.get(str(k)) == self.sql_columns.get(str(v))
            )

        return type(self).create_view_from_query(query, self._parent_tables)

    def copy(self):
        if self.is_dee():
            return self.dee()
        elif self.is_dum():
            return self.dum()
        return type(self).create_view_from_query(
            select(self.sql_columns).select_from(self._table),
            self._parent_tables,
            self._count,
        )

    def itervalues(self):
        raise NotImplementedError()

    def as_numpy_array(self):
        return np.array(self._fetchall())

    def as_pandas_dataframe(self):
        return pd.DataFrame(self._fetchall(), columns=self.columns)

    def _fetchall(self):
        if self.arity > 0 and self._table is not None:
            query = (
                select(self.sql_columns).select_from(self._table).distinct()
            )
            with SQLAEngineFactory.get_engine().connect() as conn:
                res = conn.execute(query)
                return res.fetchall()
        elif self._count == 1:
            return [()]
        return []

    def equijoin(self, other, join_indices=None):
        """
        Join sets on column equality.

        Parameters
        ----------
        other : RelationalAlgebraFrozenSet
            The other set to join on
        join_indices : Iterable[tuple(int)], optional
            The column indices to join on, by default None

        Returns
        -------
        RelationalAlgebraFrozenSet
            The set with elements where all columns in join_indices are equal.
        """
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        query = select(
            self.sql_columns
            + [
                other.sql_columns.get(str(i)).label(str(i + self.arity))
                for i in range(other.arity)
            ]
        )

        if join_indices is not None and len(join_indices) > 0:
            on_clause = and_(
                *[
                    self.sql_columns.get(str(i))
                    == other.sql_columns.get(str(j))
                    for i, j in join_indices
                ]
            )
            # Create an alias on the other table's name if we're joining on
            # the same table.
            other_join_table = other._table
            if other._table_name == self._table_name:
                other_join_table = other_join_table.alias()
            query = query.select_from(
                self._table.join(other_join_table, on_clause)
            )

        return type(self).create_view_from_query(
            query, self._parent_tables | other._parent_tables
        )

    def cross_product(self, other):
        """
        Cross product with other set.

        Parameters
        ----------
        other : RelationalAlgebraFrozenSet
            The other set for the join.

        Returns
        -------
        RelationalAlgebraFrozenSet
            Resulting set of cross product.
        """
        return self.equijoin(other)

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        if hasattr(self, "_one_row"):
            res = self._one_row
        else:
            query = select(self.sql_columns).select_from(self._table).limit(1)
            with SQLAEngineFactory.get_engine().connect() as conn:
                conn.execute("PRAGMA optimize;")
                with log_performance(
                    LOG, "Fetch one", end_message="\tFetch one %2.2fs"
                ):
                    res = conn.execute(query).fetchone()

            self._one_row = res
            self._explain_long_query(
                query,
            )
        return None if res is None else tuple(res)

    @staticmethod
    def _explain_long_query(query, *params):
        qp = "EXPLAIN QUERY PLAN %s" % (
            str(query.compile(compile_kwargs={"literal_binds": True}))
        )
        with SQLAEngineFactory.get_engine().connect() as conn:
            eqp = conn.execute(qp, params).fetchall()
            tabs = {0: ""}
            tree = "\nQUERY PLAN\n"
            for r in eqp:
                line = tabs[r[1]] + "|--"
                if r[0] not in tabs:
                    tabs[r[0]] = tabs[r[1]] + "  "
                tree += line + r[3] + "\n"
            LOG.info(
                "EXPLAIN QUERY PLAN {}".format(
                    str(query.compile(compile_kwargs={"literal_binds": True}))
                )
            )
            LOG.info(tree)

    def groupby(self, columns):
        """
        Apply group_by to a subset of columns.

        Parameters
        ----------
        columns : Iterable[int, str]
            The list of columns to group on.

        Yields
        -------
        tuple(Union[int, str], RelationalAlgebraFrozenSet)
            The different values for the group_by clause with the
            associated result set.
        """
        if self._table is not None:
            single_column = False
            if isinstance(columns, (str, int)):
                single_column = True
                columns = (columns,)

            groupby = [self.sql_columns.get(str(c)) for c in columns]
            query = (
                select(self.sql_columns)
                .select_from(self._table)
                .group_by(*groupby)
            )
            with SQLAEngineFactory.get_engine().connect() as conn:
                res = conn.execute(query).fetchall()
            for row in res:
                group = self.selection(dict(zip(columns, row)))
                if single_column:
                    t_out = row[0]
                else:
                    t_out = tuple(row)
                yield t_out, group

    def projection(self, *columns, reindex=True):
        """
        Project the set on the specified columns. Creates a view with only the
        specified columns.

        Returns
        -------
        RelationalAlgebraFrozenSet
            The projected set.
        """
        if len(columns) == 0 or self.arity == 0:
            new = type(self)()
            if len(self) > 0:
                new._count = 1
            return new

        if reindex:
            proj_columns = [
                self.sql_columns.get(str(c)).label(str(i))
                for i, c in enumerate(columns)
            ]
        else:
            proj_columns = [self.sql_columns.get(str(c)) for c in columns]
        query = select(proj_columns).select_from(self._table)
        return type(self).create_view_from_query(query, self._parent_tables)

    def __repr__(self):
        t = self._table_name
        return "{}({})".format(type(self), t)

    def __eq__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee() or other.is_dee():
                res = self.is_dee() and other.is_dee()
            elif self.is_dum() or other.is_dum():
                res = self.is_dum() and other.is_dum()
            elif self._table_name == other._table_name:
                res = True
            elif not self._equal_sets_structure(other):
                res = False
            else:
                select_left = select(columns=self.sql_columns).select_from(
                    self._table
                )
                select_right = select(
                    [other.sql_columns.get(c) for c in self.columns]
                ).select_from(other._table)
                diff_left = select_left.except_(select_right)
                diff_right = select_right.except_(select_left)
                with SQLAEngineFactory.get_engine().connect() as conn:
                    if conn.execute(diff_left).fetchone() is not None:
                        res = False
                    elif conn.execute(diff_right).fetchone() is not None:
                        res = False
                    else:
                        res = True
            return res
        else:
            return super().__eq__(other)

    def _equal_sets_structure(self, other):
        return set(self.columns) == set(other.columns)

    def _do_set_operation(self, other, sql_operator):
        if not self._equal_sets_structure(other):
            raise ValueError(
                "Relational algebra set operators can only be used on sets"
                " with same columns."
            )
        query = sql_operator(
            select(self.sql_columns).select_from(self._table),
            select(
                [other.sql_columns.get(c) for c in self.columns]
            ).select_from(other._table),
        )
        return type(self).create_view_from_query(
            query, self._parent_tables | other._parent_tables
        )

    def __and__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        return self._do_set_operation(other, intersect)

    def __or__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__or__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        return self._do_set_operation(other, union)

    def __sub__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        if self._table is None or other._table is None:
            return self.copy()
        return self._do_set_operation(other, except_)

    def __hash__(self):
        if self._table is None:
            return hash((tuple(), None))
        return hash(tuple(self.columns, self.as_numpy_array().tobytes()))


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):
    """
    A RelationalAlgebraFrozenSet with an underlying SQL representation.
    Data for this set is either stored in a table in an SQL database,
    or stored as a view which represents a query to be executed by the SQL
    server.
    """

    def __init__(self, columns=None, iterable=None):
        if isinstance(columns, RelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        self._table_name = self._new_name()
        self._count = None
        self._table = None
        self._is_view = False
        self._parent_tables = {}
        self._check_for_duplicated_columns(columns)
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            if columns is None or columns == iterable.columns:
                self._init_from(iterable)
            else:
                self._init_from_and_rename(iterable, columns)
        elif columns is not None and len(columns) > 0:
            self._create_insert_table(iterable, columns)

    def _create_insert_table(self, data, columns=None):
        """
        Initialise the set with the provided data collection.
        We use pandas to infer datatypes from the data and create
        the appropriate sql statement.
        We then read the table metadata and store it in self._table.

        Parameters
        ----------
        data : Union[pd.DataFrame, Iterable[Any]]
            The initial data for the set.
        columns : List[str], optional
            The column names, by default None.
        """
        if data is None:
            data = []
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=columns)
        else:
            data.columns = list(columns)
        data = data.drop_duplicates()
        self._table = self._insert_dataframe_into_db(data)
        self._parent_tables = {self._table}
        self._count = len(data)

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if columns is not None and len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    def _init_from_and_rename(self, other, columns):
        """
        Initialize this set using the other set's values while also
        renaming the columns. Called on init when a new
        list of columns is passed along with a set to init from.
        This method creates a view pointing to the other table
        with new column names.

        Parameters
        ----------
        other : NamedRelationalAlgebraFrozenSet
            The set to initialize from
        columns : List[str]
            The list of new column names
        """
        if other._table is not None:
            query = select(
                [c.label(str(nc)) for c, nc in zip(other.sql_columns, columns)]
            ).select_from(other._table)
            view = CreateView(self._table_name, query)
            with SQLAEngineFactory.get_engine().connect() as conn:
                conn.execute(view)
            t = table(self._table_name)
            for c in query.c:
                c._make_proxy(t)
            self._table = t
            self._is_view = True
        self._count = other._count
        self._parent_tables = other._parent_tables

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    @property
    def dummy_row_type(self):
        """
        Return dummy row_type of Tuple[Unknown * arity] to avoid calls
        to the DB.
        TODO: implement this for real.
        """
        if self.arity > 0:
            return Tuple[tuple(Unknown for _ in range(self.arity))]
        return Tuple

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        """
        List of columns as string identifiers.

        Returns
        -------
        Iterable[str]
            Set of column names.
        """
        if self._table is None:
            return tuple()
        return tuple(self._table.c.keys())

    @property
    def sql_columns(self):
        """
        List of columns as sqlalchemy.schema.Columns collection.

        Returns
        -------
        Iterable[sqlalchemy.schema.Columns]
            Set of columns.
        """
        if self._table is None:
            return []
        return self._table.c

    def projection(self, *columns):
        return super().projection(*columns, reindex=False)

    def cross_product(self, other):
        """
        Cross product with other set.

        Parameters
        ----------
        other : NamedRelationalAlgebraFrozenSet
            The other set for the join.

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            Resulting set of cross product.

        Raises
        ------
        ValueError
            Raises ValueError when cross product is done on sets with
            intersecting columns.
        """
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if len(set(self.columns).intersection(set(other.columns))) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        query = select([self._table, other._table])
        return type(self).create_view_from_query(
            query, self._parent_tables | other._parent_tables
        )

    def naturaljoin(self, other):
        """
        Natural join on the two sets. Natural join creates a view
        representing the join query. Indexes are created on all parent tables
        for the join column tuple if needed.

        Parameters
        ----------
        other : NamedRelationalAlgebraSet
            The other set to join with.

        Returns
        -------
        NamedRelationalAlgebraSet
            The joined set.
        """
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self.cross_product(other)
        return self._do_join(other, on, isouter=False)

    def left_naturaljoin(self, other):
        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self
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
        self._try_to_create_index(on)
        other._try_to_create_index(on)
        on_clause = and_(
            *[self._table.c.get(col) == other._table.c.get(col) for col in on]
        )
        select_cols = [self._table] + [
            other._table.c.get(col)
            for col in set(other.columns) - set(self.columns)
        ]
        # Create an alias on the other table's name if we're joining on the
        # same table.
        other_join_table = other._table
        if other._table_name == self._table_name:
            other_join_table = other_join_table.alias()
        query = select(select_cols).select_from(
            self._table.join(other_join_table, on_clause, isouter=isouter)
        )
        return type(self).create_view_from_query(
            query, self._parent_tables | other._parent_tables
        )

    def _try_to_create_index(self, on):
        """
        Create an index on this set's parent tables and specified columns.
        Index is only created for each table if it does not already have an
        index on the same set of columns.

        Parameters
        ----------
        on : List[str]
            List of columns to create the index on.
        """
        for table in self._parent_tables:
            if not self.has_index(table, on):
                table_idx_cols = [table.c.get(c) for c in on if c in table.c]
                if len(table_idx_cols) > 0:
                    # Create an index on the columns
                    i = Index(
                        "idx_{}_{}".format(table, "_".join(on)),
                        *table_idx_cols,
                    )
                    i.create(SQLAEngineFactory.get_engine())
                    # Analyze the table
                    with SQLAEngineFactory.get_engine().connect() as conn:
                        conn.execute("ANALYZE {}".format(table))

    @staticmethod
    def has_index(table, columns):
        """
        Checks whether the SQL table already has an index on the specified
        columns

        Parameters
        ----------
        table : sqlalchemy.schema.Table
            The SQLA table representation.
        columns : List[str]
            List of column identifiers.

        Returns
        -------
        bool
            True if _table has an index with the same set of columns.
        """
        if table is not None and table.indexes is not None:
            for index in table.indexes:
                if set(index.columns.keys()) == set(columns):
                    return True
        return False

    def __iter__(self):
        """
        Iterate over set values. Values are returned as namedtuples.

        Yields
        -------
        Iterator[NamedTuple]
            Set values.
        """
        named_tuple_type = namedtuple("tuple", self.columns)
        if self.arity > 0 and len(self) > 0:
            query = (
                select(self.sql_columns).select_from(self._table).distinct()
            )
            with SQLAEngineFactory.get_engine().connect() as conn:
                res = conn.execute(query)
                for t in res:
                    yield named_tuple_type(**t)
        elif self.arity == 0 and len(self) > 0:
            yield tuple()

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def rename_column(self, src, dst):
        if (dst) in self.columns:
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"{dst} is already a column name."
            )
        query = select(
            [
                c.label(str(dst)) if c.name == src else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        return type(self).create_view_from_query(
            query, self._parent_tables, self._count
        )

    def rename_columns(self, renames):
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self.columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )
        query = select(
            [
                c.label(str(renames.get(c.name))) if c.name in renames else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        return type(self).create_view_from_query(
            query, self._parent_tables, self._count
        )

    def aggregate(self, group_columns, aggregate_function):
        """
        Group by set values on group_columns, while applying aggregate
        functions.

        Parameters
        ----------
        group_columns : List[str, int]
            List of columns to group on
        aggregate_function : Union[Dict[str, Union[callable, str]],
                    List[tuple(str, str, Union[callable, str])]]
            dict of destination column name -> aggregate function

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            New set with aggregated values as columns

        Raises
        ------
        ValueError
            Raised on unsupported aggregate function
        """
        if isinstance(group_columns, str) or not isinstance(
            group_columns, Iterable
        ):
            group_columns = (group_columns,)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")

        distinct_sub_query = (
            select(self.sql_columns)
            .select_from(self._table)
            .distinct()
            .alias()
        )
        agg_cols = self._build_aggregate_functions(
            group_columns, aggregate_function, distinct_sub_query
        )
        groupby = [distinct_sub_query.c.get(str(c)) for c in group_columns]

        query = select(groupby + agg_cols).group_by(*groupby)
        return type(self).create_view_from_query(query, self._parent_tables)

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
                lambda_name = self._new_name("lambda")
                SQLAEngineFactory.register_aggregate(
                    lambda_name,
                    len(c_),
                    f,
                    params=c_,
                )
                f_ = getattr(func, lambda_name)
            elif isinstance(f, str):
                f_ = getattr(func, f)
            else:
                raise ValueError(
                    f"Aggregate function for {src} needs "
                    "to be callable or a string"
                )
            agg_cols.append(f_(*c_).label(str(dst)))
        return agg_cols

    def extended_projection(self, eval_expressions):
        if self.is_dee():
            return self._extended_projection_on_dee(eval_expressions)
        elif self._table is None:
            return type(self)(
                columns=list(eval_expressions.keys()), iterable=[]
            )

        proj_columns = []
        for dst_column, operation in eval_expressions.items():
            if callable(operation):
                lambda_name = self._new_name("lambda")
                SQLAEngineFactory.register_function(
                    lambda_name,
                    len(self.sql_columns),
                    operation,
                    params=self.sql_columns.values(),
                )
                f_ = getattr(func, lambda_name)
                proj_columns.append(
                    f_(*self.sql_columns).label(str(dst_column))
                )
            elif isinstance(operation, abc.RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    proj_columns.append(
                        literal_column(operation).label(str(dst_column))
                    )
                else:
                    proj_columns.append(self.sql_columns.get(str(operation)))
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                proj_columns.append(
                    self.sql_columns.get(str(operation)).label(str(dst_column))
                )
            else:
                proj_columns.append(literal(operation).label(str(dst_column)))

        query = select(proj_columns).select_from(self._table)
        return type(self).create_view_from_query(query, self._parent_tables)

    def _extended_projection_on_dee(self, eval_expressions):
        """
        Extended projection when called on Dee to create set with
        constant values.
        """
        return type(self)(
            columns=eval_expressions.keys(),
            iterable=[eval_expressions.values()],
        )

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        try:
            named_tuple_type = namedtuple("tuple", self.columns)
        except ValueError:
            # Invalid column names, just return a tuple
            return super().fetch_one()
        if hasattr(self, "_one_row"):
            res = self._one_row
        else:
            query = select(self.sql_columns).select_from(self._table).limit(1)
            with SQLAEngineFactory.get_engine().connect() as conn:
                res = conn.execute(query).fetchone()
            self._one_row = res
        return None if res is None else named_tuple_type(*res)

    def to_unnamed(self):
        if self._table is not None:
            query = select(
                [c.label(str(i)) for i, c in enumerate(self.sql_columns)]
            ).select_from(self._table)
            return RelationalAlgebraFrozenSet.create_view_from_query(
                query, self._parent_tables, self._count
            )
        return RelationalAlgebraFrozenSet()

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = self.columns
        columns = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*columns)


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def __init__(self, iterable=None):
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            iterable = iterable.deep_copy()
        super().__init__(iterable=iterable)

    def _reset_cached(self):
        self._count = None
        if hasattr(self, "_one_row"):
            delattr(self, "_one_row")

    def copy(self):
        return self.deep_copy()

    def add(self, value):
        self._reset_cached()
        if self._table is None:
            self._create_insert_table((value,))
        else:
            query = self._table.insert().values(self._normalise_element(value))
            with SQLAEngineFactory.get_engine().connect() as conn:
                conn.execute(query)
        self._count = None

    def discard(self, value):
        self._reset_cached()
        value = self._normalise_element(value)
        query = self._table.delete()
        for c, v in value.items():
            query = query.where(self.sql_columns.get(c) == v)
        with SQLAEngineFactory.get_engine().connect() as conn:
            conn.execute(query)
        self._count = None

    def __ior__(self, other):
        self._reset_cached()
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee() or other.is_dee() or other._table is None:
                return self
            elif self._table is None:
                self._table_name = self._new_name()
                self._create_insert_table(other)
                self._count = None
                return self
            else:
                if not self._equal_sets_structure(other):
                    raise ValueError(
                        "Relational algebra set operators can only be used on sets"
                        " with same columns."
                    )
                query = self._table.insert().from_select(
                    self.columns,
                    select(
                        [other.sql_columns.get(c) for c in self.columns]
                    ).select_from(other._table),
                )
                with SQLAEngineFactory.get_engine().connect() as conn:
                    with log_performance(LOG, "IOR"):
                        conn.execute(query)
                self._count = None
                return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        self._reset_cached()
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee() and other.is_dee():
                return self.dum()
            if other._table is None or self._table is None:
                return self
            if not self._equal_sets_structure(other):
                raise ValueError(
                    "Relational algebra set operators can only be used on sets"
                    " with same columns."
                )
            query = self._table.delete().where(
                tuple_(*self.sql_columns).in_(
                    select(
                        [other.sql_columns.get(c) for c in self.columns]
                    ).select_from(other._table)
                )
            )
            with SQLAEngineFactory.get_engine().connect() as conn:
                with log_performance(LOG, "ISUB"):
                    conn.execute(query)
            self._count = None
            return self
        else:
            return super().__isub__(other)
