import logging
import nibabel
import types

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from threading import RLock, get_ident
from typing import AbstractSet, Callable, Dict, Union

from .engines import (
    NeurolangEngineConfiguration,
    NeurolangEngineSet,
)
from ..relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from ...commands import CommandsMixin
from ...expressions import Command
from ...frontend.query_resolution_expressions import Symbol
from ...type_system import get_args, is_leq_informative


LOG = logging.getLogger(__name__)


class LRUCacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, cache_len: int = 15, *args, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)
        return val


class NeurolangQueryManager:
    """
    Class to manage execution of queries and keep track of results.

    This class is initialized with a dict of `NeurolangEngineConfiguration`
    that defines which Neurolang engine to create and how many.
    The NeurolangQueryManager creates a pool of thread workers (as many as
    there are engines) to execute the queries that are submitted.

    It also keeps track of results in memory, in a results_cache which is a
    dict of uuid -> Future.
    """

    def __init__(
        self,
        options: Dict[NeurolangEngineConfiguration, int],
        cache_len: int = 15,
    ) -> None:
        """
        By default the `NeurolangQueryManager` creates
        `nb_engines = sum(options.values())` workers, but no engines. It will
        call the `_init_engines` method to create the engines asynchronously.

        Parameters
        ----------
        options : Dict[NeurolangEngineConfiguration, int]
            a dictionary defining the types of engines and the number of each
            type to create.
        cache_len : int, optional
            the maximum size for the results cache, by default 15
        """
        self.engines = {}
        self._lock = RLock()
        self.results_cache = LRUCacheDict(cache_len=cache_len)
        self.configs = options

        nb_engines = sum(options.values())
        LOG.debug("Creating query manager with %d workers.", nb_engines)
        self.executor = ThreadPoolExecutor(max_workers=nb_engines)
        self._init_engines(options)

    def _init_engines(
        self, options: Dict[NeurolangEngineConfiguration, int]
    ) -> None:
        """
        Dispatch a series of tasks to create `nb_engines` engines using the
        `NeurolangEngineConfiguration.create` functions.

        Parameters
        ----------
        options : Dict[NeurolangEngineConfiguration, int]
            a dictionary defining the types of engines and the number of each
            type to create.
        """

        def create_wrapper(config: NeurolangEngineConfiguration):
            engine = config.create()
            return (config.key, engine)

        # Dispatch the engine create tasks to the workers
        for config, nb in options.items():
            LOG.debug("Starting creation of %d %s engines...", nb, config.key)
            for _ in range(nb):
                futures = [self.executor.submit(create_wrapper, config)]
                for future in as_completed(futures):
                    future.add_done_callback(self._engine_created)

    def _engine_created(self, future: Future) -> None:
        """
        Callback called when an engine creation task has finished.

        Parameters
        ----------
        future : Future
            the future holding the result of the engine creation task.
        """
        key, engine = future.result()
        with self._lock:
            engine_set = self.engines.get(key, None)
            if engine_set is None:
                engine_set = NeurolangEngineSet(engine)
                self.engines[key] = engine_set
            else:
                engine_set.add_engine(engine)
            LOG.debug(
                "Added a created engine of type %s, got %d of this type.",
                key, engine_set.counter
            )

    def _execute_neurolang_query(
        self, query: str, engine_type: str
    ) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Function executed on a ThreadPoolExecutor worker to execute a query
        against a neurolang engine.

        Parameters
        ----------
        query : str
            the query to execute
        engine_type : str
            the type of engine on which to execute the query

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            the result of the query execution
        """
        LOG.debug("[Thread - %s] - Executing query...", get_ident())
        LOG.debug("[Thread - %s] - Query :\n%s", get_ident(), query)
        engine_set = self.engines[engine_type]
        with engine_set.engine() as engine:
            LOG.debug(
                "[Thread - %s] - Engine of type %s acquired.",
                get_ident(), engine_type
            )
            try:
                with engine.scope:
                    LOG.debug("[Thread - %s] - Solving query...", get_ident())
                    res = engine.execute_datalog_program(query)
                    if res is None:
                        res = engine.solve_all()
                    else:
                        res = {"ans": res}
                    LOG.debug("[Thread - %s] - Query solved.", get_ident())
                    for s in engine.program_ir.probabilistic_predicate_symbols:
                        try:
                            res[s]._is_probabilistic = True
                        except KeyError:
                            pass
                    if engine.current_program:
                        try:
                            last_parsed_symbol = engine.current_program[-1].expression.consequent.functor
                            res[last_parsed_symbol.name]._last_parsed_symbol = True
                        except Exception:
                            pass
                    return res
            except Exception as e:
                LOG.debug(
                    "[Thread - %s] - Query execution raised %s.", get_ident(), e
                )
                raise e
            finally:
                LOG.debug(
                    "[Thread - %s] - Engine of type %s released.", get_ident(), engine_type
                )

    def submit_query(self, uuid: str, query: str, engine_type: str) -> Future:
        """
        Submit a query to one of the available engines / workers.

        Parameters
        ----------
        uuid : str
            the uuid for the query.
        query : str
            the datalog query to execute.
        engine_type : str
            the type of engine to use for solving the query.

        Returns
        -------
        Future
            a future result for the query execution.
        """
        LOG.debug(
            "Submitting query with uuid %s to executor pool of %s engines.",
            uuid, engine_type
        )

        future_res = self.executor.submit(
            self._execute_neurolang_query, query, engine_type
        )
        self.results_cache[uuid] = future_res
        return future_res

    def get_result(self, uuid: str) -> Future:
        """
        Fetch the results for a query execution.

        Parameters
        ----------
        uuid : str
            the query id

        Returns
        -------
        Future
            the Future results of the execution task.
        """
        return self.results_cache[uuid]

    def cancel(self, uuid: str) -> bool:
        """
        Attempt to cancel the execution. If the call is currently being
        executed or finished running and cannot be cancelled then the method
        will return False, otherwise the call will be cancelled and the
        method will return True.

        Parameters
        ----------
        uuid : str
            the task execution id

        Returns
        -------
        bool
            True if cancelled
        """
        return self.results_cache[uuid].cancel()

    def get_atlas(self, engine_type: str) -> nibabel.Nifti1Image:
        """
        Get the atlas for a given engine type.

        Parameters
        ----------
        engine_type : str
            the engine type

        Returns
        -------
        nibabel.Nifti1Image
            the mni atlas
        """
        config = [c for c in self.configs.keys() if c.key == engine_type]
        return config[0].atlas

    def get_symbols(self, engine_type: str) -> Future:
        """
        Request the symbols table for a given engine. If the symbols table for
        this engine type is present in cache, then this is returned, otherwise
        a job to fetch the symbols table is dispatched to the threadpool
        executor.

        Parameters
        ----------
        engine_type : str
            the engine type.

        Returns
        -------
        Future
            the Future result for the symbols query
        """
        LOG.debug("Requesting symbols for %s engine...", engine_type)
        key = f"ENGINE_SYMBOLS_{engine_type}"
        if key in self.results_cache:
            LOG.debug("Returning cached symbols for %s engine.", engine_type)
            return self.results_cache[key]

        # Raise a KeyError if engines are not yet available
        self.engines[engine_type]

        # Submit task
        future_res = self.executor.submit(
            self._get_engine_symbols, engine_type
        )
        self.results_cache[key] = future_res
        return future_res

    def _get_engine_symbols(
        self, engine_type: str
    ) -> Dict[str, Union[RelationalAlgebraFrozenSet, Symbol]]:
        """
        Function executed on a ThreadPoolExecutor worker to get the available symbols
        on a neurolang engine.

        Parameters
        ----------
        engine_type : str
            the type of engine for which to get the symbols

        Returns
        -------
        Dict[str, Union[NamedRelationalAlgebraFrozenSet, Symbol]]
            the result of the query execution
        """
        LOG.debug(
            "[Thread - %s] - Fetching symbols for %s engine...",
            get_ident(), engine_type
        )
        engine_set = self.engines[engine_type]
        with engine_set.engine() as engine:
            try:
                LOG.debug(
                    "[Thread - %s] - Engine of type %s acquired.",
                    get_ident(), engine_type
                )
                LOG.debug(
                    "[Thread - %s] - Returning symbol_table.",
                    get_ident()
                )
                symbols = {}
                for name in engine.symbols:
                    if not name.startswith("_"):
                        symbol = engine.symbols[name]
                        if is_leq_informative(symbol.type, Callable):
                            symbols[name] = symbol
                        elif is_leq_informative(symbol.type, AbstractSet):
                            ras = symbol.value
                            ras.row_type = get_args(symbol.type)[0]
                            symbols[name] = ras
                symbols.update(_get_commands(engine))
                return symbols
            finally:
                LOG.debug(
                    "[Thread - %s] - Engine of type %s released.",
                    get_ident(), engine_type
                )


def _get_commands(engine) -> Dict[str, Dict]:
    """
    Add the command processing functions in CommandsMixin to the list
    of symbols available on an engine. This is useful to display the
    docstring of these commands in the frontend.

    Parameters
    ----------
    engine : NeurolangPDL
        the engine with a program that extends CommandsMixin

    Returns
    -------
    Dict[str, Dict]
        a mapping of command name -> command info
    """
    commands = {}
    for name, command in CommandsMixin.__dict__.items():
        if (
            not name.startswith("_")
            and isinstance(command, types.FunctionType)
            and hasattr(engine.program_ir, name)
        ):
            commands[name] = {
                "type": str(Command),
                "doc": command.__doc__,
                "command": True,
            }
    return commands
