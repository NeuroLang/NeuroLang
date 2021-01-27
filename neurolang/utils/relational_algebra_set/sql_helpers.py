"""
See https://github.com/sqlalchemy/sqlalchemy/wiki/Views for information
on how to integrate views into SQLA.

"""
import os
import re
import pandas as pd
from collections import namedtuple
from abc import ABC
from pandas.api.types import infer_dtype
from sqlalchemy import create_engine, event, Table, BLOB
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


class CustomAggregateClass(object):
    """
    AggregateClass for use with sqlite3.Connection.create_aggregate
    This Class should have two attributes set on it when created:
    field_names : List[str] and agg_func : callable.
    Such as :

    >>> type(
            'AggSum',
            (CustomAggregateClass,),
            {"field_names": ['x', 'y'], "agg_func": staticmethod(
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
        self.multi_value = len(self.field_names) > 1

    def step(self, *value):
        if self.multi_value:
            self.values.append(value)
        else:
            self.values.append(value[0])

    def finalize(self):
        # call the agg_func with the aggregated values
        if self.multi_value:
            agg = pd.DataFrame(self.values, columns=self.field_names)
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
    _funcs = {}
    _aggregates = {}

    @classmethod
    def register_aggregate(cls, name, num_params, func, param_names):
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
        param_names : List[str]
            the string identifiers for the params
        """
        agg_class = type(
            name,
            (CustomAggregateClass,),
            {"field_names": param_names, "agg_func": staticmethod(func)},
        )
        cls._aggregates[(name, num_params)] = agg_class

    @classmethod
    def register_function(cls, name, num_params, func, param_names=None):
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
        param_names : List[str], optional
            if given, the function will be called with the arguments casted
            as a namedtuple with given field_names, by default None

        Example
        -------
        Registering the following function will let you call the my_sum
        function from an SQL query
        >>> mysum = lambda r: r.x + r.y
        >>> SQLAEngineFactory.register_function(
                "my_sum", 2, mysum, ["x", "y"]
            )

        You can then use the my_sum function in SQL:
        >>> SELECT * FROM users WHERE my_sum(users.parents, users.kids) > 5
        """

        def _transform_value(*v):
            if v is None:
                return v
            if len(v) > 1:
                if param_names is not None and cls._check_namedtuple_names(
                    param_names
                ):
                    # Pass arguments as namedtuple
                    named_tuple_type = namedtuple(name, param_names)
                    named_v = named_tuple_type(*v)
                    return func(named_v)
                # Pass arguments as tuple
                return func(v)
            # Pass single argument value
            return func(v[0])

        cls._funcs[(name, num_params)] = _transform_value

    @staticmethod
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

    @classmethod
    def _on_connect(cls, dbapi_con, connection_record):
        """
        Listener callback. Called each time cls._engine.connect() is called.
        See https://docs.sqlalchemy.org/en/14/core/event.html on events in
        SQLAlchemy.

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

    @staticmethod
    def _create_engine(echo=False):
        """
        Create the engine for the singleton.
        The engine is created with the db+dialect string found in the
        `NEURO_SQLA_DIALECT` environment variable.
        """
        dialect = os.getenv("NEURO_SQLA_DIALECT", "sqlite:///neurolang.db")
        print(
            (
                "Creating SQLA engine with {} uri."
                " Set the $NEURO_SQLA_DIALECT environment"
                " var to change it.".format(dialect)
            )
        )
        return create_engine(dialect, echo=echo)


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

    return PickleType()