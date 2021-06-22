import ast
import inspect
import logging
import threading
import time
import types
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Type, Union

import numpy as np
from neurolang.type_system import (
    Unknown,
    get_args,
    infer_type_builtins,
    typing_callable_from_annotated_function,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import functions

import pandas as pd
from pandas.api.types import pandas_dtype

from ...utils import config

if config["RAS"].getboolean("Synchronous", False):
    import dask

    dask.config.set(scheduler="single-threaded")

from dask.distributed import Client

from dask_sql import Context
from dask_sql.mappings import sql_to_python_type

LOG = logging.getLogger(__name__)


def timeit(func):
    """
    This decorator logs the execution time for the decorated function.
    Log times will not be acurate for Dask operations if using asynchronous
    scheduler.
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LOG.debug("======================================")
        LOG.debug(f"{func.__name__} took {elapsed:2.4f} s")
        LOG.debug("======================================")
        return result

    return wrapper


def _id_generator():
    lock = threading.RLock()
    i = 0
    while True:
        with lock:
            fresh = f"{i:08}"
            i += 1
        yield fresh


class DaskContextManager(ABC):
    """
    Singleton class to manage Dask-related objects, mainly
    Dask's Client and Dask-SQL's Context.
    """

    _context = None
    _client = None
    _id_generator_ = _id_generator()

    @abstractmethod
    def _do_not_instantiate_singleton_class(self):
        pass

    @classmethod
    def _create_client(cls):
        if cls._client is None:
            cls._client = Client(processes=False)

    @classmethod
    def get_context(cls, new=False):
        if cls._context is None or new:
            if not config["RAS"].getboolean("Synchronous", False):
                cls._create_client()
            cls._context = Context()
            # We register an aggregate function called len which applies to string columns
            # Used for example in `test_probabilistic_frontend:test_postprob_conjunct_with_wlq_result`
            cls._context.register_aggregation(
                len, "len", [("x", pd.StringDtype())], np.int32
            )
        return cls._context

    @classmethod
    def sql(cls, query):
        compiled_query = cls.compile_query(query)
        LOG.info(f"Executing SQL query :\n{compiled_query}")
        return cls.get_context().sql(compiled_query)

    @staticmethod
    def compile_query(query):
        return str(
            query.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

    @classmethod
    def register_function(cls, f_, fname, params, return_type, wrapped):
        func_to_register = f_
        if wrapped:
            func_to_register = cls.wrap_function_with_dataframe(
                f_, params, return_type
            )
        cls.get_context().register_function(
            func_to_register, fname, params, return_type
        )

    @classmethod
    def register_aggregation(cls, f_, fname, params, return_type):
        # FIXME: We should preferably try to use GroupBy-aggregations
        # instead of GroupBy-apply when doing aggregations. I.e.
        # create a dd.Aggregation from the given function and register it
        # on the context. But this doesnt work in all cases, since dask
        # applies GroupBy-aggregations first on each chunk, then again to
        # the results of all the chunk aggregations.
        # So transformative aggregation will not work properly, for
        # instance sum(x) - 1 will result in sum(x) - 2 in the end.
        # agg = dd.Aggregation(
        #     fname, lambda chunk: chunk.agg(f_), lambda total: total.agg(f_)
        # )

        func_to_register = f_
        if len(params) > 1:
            func_to_register = cls.wrap_function_with_param_names(f_, params)
        cls.get_context().register_aggregation(
            func_to_register, fname, params, return_type
        )

    @staticmethod
    def wrap_function_with_param_names(f_, params):
        try:
            pnames = [name for (name, _) in params]
            named_tuple_type = namedtuple("LambdaTuple", pnames)
        except ValueError:
            # Invalid column names, just use a tuple instead.
            named_tuple_type = None

        def wrapped_custom_function(*values):
            if named_tuple_type:
                return f_(named_tuple_type(*values))
            else:
                return f_(tuple(values))

        return wrapped_custom_function

    @staticmethod
    def wrap_function_with_dataframe(f_, params, return_type):
        """
        The way a function is called in dask_sql is by calling it with a list
        of its parameters (dask series,see dask_sql.physical.rex.core.call.py)
        Also, dask_sql is changing the names of the columns internally.
        What we want to do is transform these Series into a dataframe with the
        expected column names (params) and call apply on it.

        This is what wrapped_custom_function does.

        Concatenating Series into a Dask DataFrame is complicated because of
        unknown partitions, so we turn the first Series into a DataFrame and
        then assign the other Series as columns.
        """
        pnames = [name for (name, _) in params]

        def wrapped_custom_function(*values):
            s0 = values[0]
            s0.name = pnames[0]
            ddf = values[0].to_frame()
            for name, col in zip(pnames[1:], values[1:]):
                ddf[name] = col
            return ddf.apply(f_, axis=1, meta=(None, return_type))

        return wrapped_custom_function


def convert_types_to_pandas_dtype(
    types: pd.Series, default_type: np.dtype = np.float64
) -> pd.Series:
    """
    Convert a series of native python types and typing type annotations
    into a series of pandas/numpy dtypes.

    Parameters
    ----------
    types : pd.Series
        the types
    default_type : np.dtype, optional
        the default type for when a type is Unknown, by default np.float64

    Returns
    -------
    pd.Series
        the converted types
    """
    nptypes = []
    for t in types:
        nptypes.append(convert_type_to_pandas_dtype(t, default_type))
    return pd.Series(nptypes, index=types.index)


def convert_type_to_pandas_dtype(
    type_: Union[type, Type], default_type: np.dtype = np.float64
) -> np.dtype:
    """
    Convert a native python type or typing type annotation to a numpy dtype.

    Parameters
    ----------
    type_ : Union[type, Type]
        the type to convert
    default_type : np.dtype, optional
        default dtype used for when type is Unknown, by default np.float64

    Returns
    -------
    np.dtype
        the converted type
    """
    if type_ is Unknown:
        nptype = default_type
    elif isinstance(type_, type) and issubclass(type_, str):
        nptype = pd.StringDtype()
    else:
        try:
            nptype = pandas_dtype(type_)
        except TypeError:
            # assume it's an object type
            nptype = np.object_
    return nptype


def try_to_infer_type_of_operation(
    operation, column_types, default_type=np.float64
):
    """
    Tries to infer the return type for an operation passed to aggregate
    or extended_projection methods.
    In order to work with dask-sql, the return type should be a pandas
    or numpy type.

    Parameters
    ----------
    operation : Union[Callable, str]
        The operation to infer the type for
    column_types : pd.Series
        The dtypes series mapping the dtype for each column.
        Used if operation references a known column.
    default_type : Type, optional
        The return value if type cannot be infered, by default np.float64

    Returns
    -------
    Type
        An infered return type for the operation.
    """
    try:
        # 1. First we try to guess the return type of the operation
        if isinstance(operation, (types.FunctionType, types.MethodType)):
            # operation is a custom function
            rtype = typing_callable_from_annotated_function(operation)
            rtype = get_args(rtype)[1]
        elif isinstance(operation, types.BuiltinFunctionType):
            # operation is something like 'sum'
            rtype = infer_type_builtins(operation)
            rtype = get_args(rtype)[1]
        else:
            if isinstance(operation, str):
                default_type = str
                # check if it's one of SQLAlchemy's known functions, like count
                if hasattr(functions, operation):
                    rtype = getattr(functions, operation).type
                    if inspect.isclass(rtype):
                        rtype = rtype()
                    rtype = sql_to_python_type(rtype.compile())
                else:
                    # otherwise operation is probably a str or
                    # RelationalAlgebraStringExpression representing a column
                    # literal, like 'col_a + 1', or a constant like '0'.
                    # We try to parse the expression to get type of variable or
                    # constant.
                    rtype = type_of_expression(
                        ast.parse(operation, mode="eval").body, column_types
                    )
            else:
                rtype = type(operation)
    except (ValueError, TypeError, NotImplementedError, SyntaxError):
        LOG.warning(
            f"Unable to infer type of operation {operation}"
            f", assuming default {default_type} type instead."
        )
        rtype = default_type
    return rtype


def type_of_expression(node, column_types):
    if isinstance(node, ast.Num):  # <number>
        return type(node.n)
    elif isinstance(node, ast.Constant):  # Constant
        return type(node.value)
    elif isinstance(node, ast.Name):
        if node.id in column_types.index:  # Col name
            return column_types[node.id]
        else:
            return str
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return type_of_expression(node.left, column_types)
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return type_of_expression(node.operand, column_types)
    else:
        return Unknown
