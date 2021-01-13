from neurolang.utils.relational_algebra_set.sql_helpers import CreateView
import uuid

from . import abstract as abc
import pandas as pd
from sqlalchemy import Table, func, MetaData, and_, select, create_engine
from sqlalchemy.sql import table

# Create the engine to connect to the PostgreSQL database
# engine = sqlalchemy.create_engine('sqlite:///neurolang.db', echo=True)
# metadata = MetaData()


class SQLAEngineFactory:
    engine = create_engine("sqlite:///neurolang.db", echo=True)

    @classmethod
    def get_engine(cls):
        return cls.engine


class SQLARelationalAlgebraFrozenSet(abc.RelationalAlgebraFrozenSet):
    """
    A RelationalAlgebraFrozenSet with an underlying SQL representation.
    Data for this set is either stored in a table in an SQL database,
    or constructed as a query which will then be evaluated on the
    database tables.
    """

    def __init__(self, engine, iterable=None):
        self._table_name = str(uuid.uuid4())
        self._might_have_duplicates = True
        self._dtypes = None
        self._len = 0
        self._table = None
        if iterable is not None:
            if isinstance(iterable, SQLARelationalAlgebraFrozenSet):
                self._copy_from(iterable)
            else:
                if not isinstance(iterable, pd.DataFrame):
                    iterable = pd.DataFrame(iterable)
                iterable.to_sql(self._table_name, engine, index=False)
                self._table = Table(
                    self._table_name,
                    MetaData(),
                    autoload=True,
                    autoload_with=engine,
                )

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._copy_from(other)
        return output

    def _copy_from(self, other):
        """
        Copy the underlying attributes of the other set on this one.

        Parameters
        ----------
        other : [type]
            [description]
        """
        self._table_name = other._table_name
        self._count = other._count
        self._dtypes = other._dtypes
        self._might_have_duplicates = other._might_have_duplicates

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    def is_empty(self):
        return self._count == 0

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    @property
    def arity(self):
        if self._dtypes is None:
            return 0
        return len(self._dtypes)

    @property
    def columns(self):
        return self._dtypes.index

    def _empty_set_same_structure(self):
        return type(self)()

    def projection(self, *columns):
        if self.is_empty():
            return self._empty_set_same_structure()
        self
        new_container = self._container[list(columns)]
        new_container.columns = pd.RangeIndex(len(columns))
        output = self._empty_set_same_structure()
        output._container = new_container
        return output


class NamedSQLARelationalAlgebraFrozenSet(abc.NamedRelationalAlgebraFrozenSet):
    """
    A RelationalAlgebraFrozenSet with an underlying SQL representation.
    Data for this set is either stored in a table in an SQL database,
    or constructed as a query which will then be evaluated on the
    database tables.
    """

    def __init__(self, engine, columns=None, data=None):
        self._table_name = self._new_name()
        self._count = None
        self._table = None
        self.engine = engine
        self._check_for_duplicated_columns(columns)
        if isinstance(data, SQLARelationalAlgebraFrozenSet):
            self._init_from(data)
        elif data is not None:
            self._create_insert_table(data, columns)

    def _create_insert_table(self, data, columns=None):
        """
        We use pandas to infer datatypes from the data and create
        the appropriate sql statement.
        We then read the table metadata and store it in self._table
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=columns)
        data.to_sql(self._table_name, self.engine, index=False)
        self._table = Table(
            self._table_name,
            MetaData(),
            autoload=True,
            autoload_with=self.engine,
        )

    def _init_from(self, other):
        self._table_name = other._table_name
        self._count = other._count
        self._table = other._table
        self._query = other._query

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if columns is not None and len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    @staticmethod
    def _new_name():
        return "table_" + str(uuid.uuid4()).replace("-", "_")

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._init_from(other)
        return output

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    def is_empty(self):
        return len(self) == 0

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    @property
    def arity(self):
        if self._table is None:
            return 0
        return len(self._table.c)

    @property
    def columns(self):
        if self._table is None:
            return []
        return self._table.c.keys()

    @property
    def sql_columns(self):
        if self._table is None:
            return []
        return self._table.c

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if len(set(self.columns).intersection(set(other.columns))) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        query = select([self._table, other._table])
        return self._create_view_from_query(query)

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self.cross_product(other)

        on_clause = and_(
            *[self._table.c.get(col) == other._table.c.get(col) for col in on]
        )
        select_cols = [self._table] + [
            other._table.c.get(col)
            for col in set(other.columns) - set(self.columns)
        ]
        query = select(select_cols).select_from(
            self._table.join(other._table, on_clause)
        )
        return self._create_view_from_query(query)

    def _create_view_from_query(self, query):
        output = NamedSQLARelationalAlgebraFrozenSet(self.engine)
        view = CreateView(output._table_name, query)
        with self.engine.connect() as conn:
            conn.execute(view)
        t = table(output._table_name)
        for c in query.c:
            c._make_proxy(t)
        output._table = t
        return output

    def __len__(self):
        if self._count is None:
            query = select([func.count()]).select_from(self._table).distinct()
            with self.engine.connect() as conn:
                res = conn.execute(query).scalar()
                self._count = res
        return self._count

    def __iter__(self):
        if self.arity > 0 and len(self) > 0:
            query = (
                select(self.sql_columns).select_from(self._table).distinct()
            )
            with self.engine.connect() as conn:
                res = conn.execute(query)
                for t in res:
                    yield tuple(t)
        elif self.arity == 0 and len(self) > 0:
            yield tuple()

    def __contains__(self, element):
        pass

    def projection(self, *columns):
        pass

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def left_naturaljoin(self, other):
        pass

    def rename_column(self, src, dst):
        pass

    def rename_columns(self, renames):
        pass

    def groupby(self, columns):
        pass

    def aggregate(self, group_columns, aggregate_function):
        pass

    def extended_projection(self, eval_expressions):
        pass

    def fetch_one(self):
        query = select(self.sql_columns).select_from(self._table).limit(1)
        with self.engine.connect() as conn:
            res = conn.execute(query)
        return res.fetchone()

    def to_unnamed(self):
        pass

    def selection(self, select_criteria):
        pass

    def selection_columns(self, select_criteria):
        pass

    def copy(self):
        pass

    def itervalues(self):
        pass

    def as_numpy_array():
        pass

    def __repr__(self):
        t = self._table_name
        d = {}
        if self._table is not None and not self.is_empty():
            with self.engine.connect() as conn:
                d = pd.read_sql(
                    select(self.sql_columns)
                    .select_from(self._table)
                    .distinct(),
                    conn,
                )
        return "{}({},\n {})".format(type(self), t, d)
