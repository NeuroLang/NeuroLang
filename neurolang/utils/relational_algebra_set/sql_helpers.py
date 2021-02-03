"""
See https://github.com/sqlalchemy/sqlalchemy/wiki/Views for information
on how to integrate views into SQLA.

"""
import os
import re
import pickle
import datetime
from typing import Iterable
import pandas as pd
from collections import namedtuple
from abc import ABC
from pandas.api.types import infer_dtype
from sqlalchemy import create_engine, event, Table, BLOB
import sqlalchemy
from sqlalchemy.schema import DDLElement
from sqlalchemy.ext import compiler
from sqlalchemy.event import listen
from sqlalchemy.types import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    SmallInteger,
    Text,
    Time,
    PickleType,
)


class CreateTableAs(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def compile(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropView)
def compile(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


@compiler.compiles(CreateTableAs)
def _create_table_as(element, compiler, **kw):
    return "CREATE TABLE %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@event.listens_for(Table, "column_reflect")
def _setup_pickletype(inspector, table, column_info):
    """
    This listener converts all BLOB column types to PickleType
    when using SQLA Table reflection to get table info.
    See : https://docs.sqlalchemy.org/en/13/core/custom_types.html
    #working-with-custom-types-and-reflection
    """
    if isinstance(column_info["type"], BLOB):
        column_info["type"] = PickleType()


VALID_SQL_PYTHON_TYPES = set(
    [
        int,
        str,
        float,
        datetime.datetime,
        datetime.date,
        datetime.time,
        bytes,
        bool,
    ]
)


def _sql_result_processor(value):
    """
    Process the result value of a custom function or aggregate function
    to be returned to SQL. SQL can only handle specific python types. If
    the value is not an instance of one of VALID_SQL_PYTHON_TYPES, we
    return the value as bytes using pickle.

    Parameters
    ----------
    value : ANY
        the value to process
    """
    if value is None:
        return value
    for type_ in VALID_SQL_PYTHON_TYPES:
        if isinstance(value, type_):
            return value
    return pickle.dumps(value)


def _sql_params_processor(
    params: Iterable[sqlalchemy.Column], *values, as_namedtuple=False
):
    """
    Process the params passed by SQL to a registered function or aggregate
    function by unpacking the values and applying pickle.loads to those
    that are of type PickleType.

    Parameters
    ----------
    params : Iterable[sqlalchemy.Column]
        param column description
    as_namedtuple: bool, optional
        if True, return the values as namedtuple
    values: Any
        param values
    """
    # Unpickle PickleType values
    unpickled = []
    for i, p in enumerate(params):
        if isinstance(p.type, PickleType) or p.type == PickleType:
            unpickled.append(pickle.loads(values[i]))
        else:
            unpickled.append(values[i])

    if len(unpickled) > 1:
        param_names = [p.name for p in params]
        if as_namedtuple and _check_namedtuple_names(param_names):
            # return arguments as namedtuple
            named_tuple_type = namedtuple("LambdaArgs", param_names)
            named_v = named_tuple_type(*unpickled)
            return named_v
        # return arguments as tuple
        return unpickled
    # return single argument value
    return unpickled[0]


def _check_namedtuple_names(field_names):
    """
    Checks that the given field_names are valid field names for
    namedtuples. Namedtuples do not accept field names which start
    with a digit or an underscore.

    Parameters
    ----------
    field_names : List[str]
        field names

    Returns
    -------
    bool
        True if all field names are valid
    """
    valid_name = re.compile("^[^0-9_]")
    return all(valid_name.match(n) is not None for n in field_names)


class CustomAggregateClass(object):
    """
    AggregateClass for use with sqlite3.Connection.create_aggregate
    This Class should have two attributes set on it when created:
    params: List[sqlalchemy.Column] and agg_func : callable.
    Such as :

    >>> type(
            'AggSum',
            (CustomAggregateClass,),
            {"params": [Column('x', Integer), Column('y', Integer)],
             "agg_func": staticmethod(
                lambda r: sum(r.x + r.y)
            )},
        )

    SQLite will call step on this class with each value to aggregate,
    then call finalize to return the aggregated value. In order to be
    compatible with pandas aggregation mecanism, the step method stores
    all the values in an array, then calls the given agg_func on the
    final result.
    """

    def __init__(self):
        self.values = []
        self.multi_value = len(self.params) > 1

    def step(self, *value):
        self.values.append(
            _sql_params_processor(self.params, *value, as_namedtuple=False)
        )

    def finalize(self):
        # call the agg_func with the aggregated values
        if self.multi_value:
            agg = pd.DataFrame(
                self.values, columns=[c.name for c in self.params]
            )
            res = self.agg_func(agg)
        else:
            res = self.agg_func(self.values)
        return res


class PandasGroupbyFirstAggregateClass(CustomAggregateClass):
    """
    Datalog Chase resolution algorithm makes extensive use of pandas
    built-in `first` function (https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.core.groupby.GroupBy.first.html).
    This class replicates the behaviour of this function to make it available
    in SQLite.
    """

    def __init__(self):
        self.values = []
        self.multi_value = False

    def step(self, *value):
        self.values.append(value[0])

    def finalize(self):
        res = self.values[0]
        return res


class SQLAEngineFactory(ABC):
    """
    Singleton class to store the SQLAlchemy engine.
    The engine is initialised using environment variables.

    Returns
    -------
    None
        Abstract class. Do not instantiate.
    """

    _engine = None
    _in_memory_sqlite = False
    _funcs = {}
    _aggregates = {}

    @classmethod
    def register_aggregate(cls, name, num_params, func, params):
        """
        Register a custom aggregate function. This function will be added
        to each new connection to the SQLite db and can be called from
        groupby queries.

        Parameters
        ----------
        name : str
            a unique function name
        num_params : int
            the number of params for the function
        func : callable
            the aggregate function to be called
        params : List[sqlalchemy.Column]
            the Column desc for the params
        """
        agg_class = type(
            name,
            (CustomAggregateClass,),
            {"params": params, "agg_func": staticmethod(func)},
        )
        cls._aggregates[(name, num_params)] = agg_class
        if cls._in_memory_sqlite:
            cls._engine.raw_connection().create_aggregate(name, num_params, agg_class)

    @classmethod
    def register_function(cls, name, num_params, func, params):
        """
        Register a custom function on the engine factory. This function
        will be added to each new connection to the SQLite database and can
        be called from queries.
        See https://docs.python.org/3/library/
        sqlite3.html#sqlite3.Connection.create_function

        The sqlite3.create_function method is scoped on each connection and
        must be invoked with each new connection. Hence registered
        functions are stored on the SQLAEngineFactory and passed to each
        new connection with a listener.

        Parameters
        ----------
        name : str
            a unique function name
        num_params : int
            the number of params for the function
        func : callable
            the function to be called
        params : List[sqlalchemy.Column]
            list of sqlalchemy Column describing the params that will be
            passed to the function when called by SQL. This list is used
            to process PickleType columns.

        Example
        -------
        Registering the following function will let you call the my_sum
        function from an SQL query
        >>> mysum = lambda r: r.x + r.y
        >>> SQLAEngineFactory.register_function(
                "my_sum", 2, mysum,
                [Column("x", Integer), Column("y", Integer)]
            )

        You can then use the my_sum function in SQL:
        >>> SELECT * FROM users WHERE my_sum(users.parents, users.kids) > 5
        """

        def _transform_value(*v):
            if v is None:
                return v
            return func(_sql_params_processor(params, *v, as_namedtuple=True))

        cls._funcs[(name, num_params)] = _transform_value
        if cls._in_memory_sqlite:
            cls._engine.raw_connection().create_function(name, num_params, _transform_value)

    @classmethod
    def _on_connect(cls, dbapi_con, connection_record):
        """
        Listener callback. Called each time cls._engine.connect() is called.
        See https://docs.sqlalchemy.org/en/14/core/event.html on events in
        SQLAlchemy.
        Note: this event is only usefull when using connection pooling with
        a persistent database. When using an in-memory SQLite database,
        SQLAlchemy uses only one connection which is kept open all the time,
        hence the _on_connect listener is called only once.
        See https://docs.sqlalchemy.org/en/14/dialects/
        sqlite.html#threading-pooling-behavior

        Parameters
        ----------
        dbapi_con : sqlite3.Connection
            a DBAPI connection
        connection_record : sqlalchemy.pool._ConnectionRecord
            the _ConnectionRecord managing the DBAPI connection.
        """
        dbapi_con.create_aggregate(
            "first", 1, PandasGroupbyFirstAggregateClass
        )
        for (name, num_params), func in cls._funcs.items():
            dbapi_con.create_function(name, num_params, func)
        for (name, num_params), klass in cls._aggregates.items():
            dbapi_con.create_aggregate(name, num_params, klass)

    @classmethod
    def get_engine(cls):
        """
        Get the SQLA engine.
        Registers the _on_connect listener on engine creation.

        Returns
        -------
        sqlalchemy.engine.Engine
            The singleton engine.
        """
        if cls._engine == None:
            cls._engine = cls._create_engine(echo=False)
            listen(cls._engine, "connect", cls._on_connect)
        return cls._engine

    @classmethod
    def _create_engine(cls, echo=False):
        """
        Create the engine for the singleton.
        The engine is created with the db+dialect string found in the
        `NEURO_SQLA_DIALECT` environment variable.
        """
        dialect = os.getenv("NEURO_SQLA_DIALECT", "sqlite://")
        # dialect = os.getenv("NEURO_SQLA_DIALECT", "sqlite:///neurolang.db")
        print(
            (
                "Creating SQLA engine with {} uri."
                " Set the $NEURO_SQLA_DIALECT environment"
                " var to change it.".format(dialect)
            )
        )
        cls._in_memory_sqlite = (
            (dialect == "sqlite://")
            or (dialect == "sqlite:///")
            or (dialect == ":memory:")
        )
        return create_engine(dialect, echo=echo)


def pickle_comparator(x, y):
    if x == y:
        return True
    if type(x) == bytes and type(y) == bytes:
        return pickle.loads(x) == pickle.loads(y)
    return False


def map_dtype_to_sqla(col):
    """
    Map a pandas.DataFrame column to an SQLA Type by infering data type.
    Adapted from pandas.io.sql.SQLTable._sqlalchemytype.
    (https://github.com/pandas-dev/pandas/blob/
    b8890eb33b40993da00656f16c65070c42429f0d/pandas/io/sql.py#L1117)

    Defaults to PickleType when "generic" types cannot be infered.
    """
    col_type = infer_dtype(col, skipna=True)

    if col_type == "datetime64" or col_type == "datetime":
        try:
            if col.dt.tz is not None:
                return TIMESTAMP(timezone=True)
        except AttributeError:
            # The column is actually a DatetimeIndex
            # GH 26761 or an Index with date-like data e.g. 9999-01-01
            if getattr(col, "tz", None) is not None:
                return TIMESTAMP(timezone=True)
        return DateTime
    if col_type == "timedelta64":
        return BigInteger
    elif col_type == "floating":
        if col.dtype == "float32":
            return Float(precision=23)
        else:
            return Float(precision=53)
    elif col_type == "integer":
        # GH35076 Map pandas integer to optimal SQLAlchemy integer type
        if col.dtype.name.lower() in ("int8", "uint8", "int16"):
            return SmallInteger
        elif col.dtype.name.lower() in ("uint16", "int32"):
            return Integer
        elif col.dtype.name.lower() == "uint64":
            raise ValueError(
                "Unsigned 64 bit integer datatype is not supported"
            )
        else:
            return BigInteger
    elif col_type == "boolean":
        return Boolean
    elif col_type == "date":
        return Date
    elif col_type == "time":
        return Time
    elif col_type == "complex":
        raise ValueError("Complex datatypes not supported")
    elif col_type == "string":
        return Text

    return PickleType(comparator=pickle_comparator)