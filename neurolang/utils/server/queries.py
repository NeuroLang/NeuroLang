import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock, get_ident
from typing import Dict

import nibabel
import tornado.websocket
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
from neurolang.utils.server.engines import (
    NeurolangEngineConfiguration,
    NeurolangEngineSet,
)


LOG = logging.getLogger(__name__)


class NeurolangQueryManager:
    """
    Class to manage execution of queries and keep track of results.

    This class is initialized with a dict of `NeurolangEngineConfiguration`
    that defines which Neurolang engine to create and how many.
    The NeurolangQueryManager creates a pool of thread workers (as many as
    there are engines) to execute the queries that are submited.

    It also keeps track of results in memory, in a results_cache which is a
    dict of uuid -> Future.
    """

    def __init__(
        self, options: Dict[NeurolangEngineConfiguration, int]
    ) -> None:
        """
        By default the `NeurolangQueryManager` creates
        `nb_engines = sum(options.values())` workers, but no engines. It will
        call the `_init_engines` method to create the engines asynchronously.

        Parameters
        ----------
        options : Dict[NeurolangEngineConfiguration, int]
            a dictionnary defining the types of engines and the number of each
            type to create.
        """
        self.engines = {}
        self._lock = RLock()
        self.results_cache = {}
        self.configs = options

        nb_engines = sum(options.values())
        LOG.debug(f"Creating query manager with {nb_engines} workers.")
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
            a dictionnary defining the types of engines and the number of each
            type to create.
        """

        def create_wrapper(config: NeurolangEngineConfiguration):
            engine = config.create()
            return (config.key, engine)

        # Dispatch the engine create tasks to the workers
        for config, nb in options.items():
            LOG.debug(f"Starting creation of {nb} {config.key} engines...")
            for _ in range(nb):
                future = self.executor.submit(create_wrapper, config)
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
                f"Added a created engine of type {key}, got {engine_set.counter} of this type."
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
        LOG.debug(f"[Thread - {get_ident()}] - Executing query...")
        LOG.debug(f"[Thread - {get_ident()}] - Query :\n{query}")
        engine_set = self.engines[engine_type]
        with engine_set.engine() as engine:
            LOG.debug(
                f"[Thread - {get_ident()}] - Engine of type {engine_type} acquired."
            )
            try:
                with engine.scope:
                    LOG.debug(f"[Thread - {get_ident()}] - Solving query...")
                    res = engine.execute_datalog_program(query)
                    if res is None:
                        res = engine.solve_all()
                    else:
                        res = {"ans": res}
                    LOG.debug(f"[Thread - {get_ident()}] - Query solved.")
                    return res
            except Exception as e:
                LOG.debug(
                    f"[Thread - {get_ident()}] - Query execution raised {e}."
                )
                raise e
            finally:
                LOG.debug(
                    f"[Thread - {get_ident()}] - Engine of type {engine_type} released."
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
            f"Submitting query with uuid {uuid} to executor pool of {engine_type} engines."
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

    def get_mni_mask(self, engine_type: str) -> nibabel.Nifti1Image:
        """
        Get the MNI mask for a given engine type.

        Parameters
        ----------
        engine_type : str
            the engine type

        Returns
        -------
        nibabel.Nifti1Image
            the mni mask
        """
        config = [c for c in self.configs.keys() if c.key == engine_type]
        return config[0].mni_mask
