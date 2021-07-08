import json
import logging
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
import os.path
from concurrent.futures import Future, ThreadPoolExecutor
from threading import get_ident, RLock
from typing import Dict
from uuid import uuid4

import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
from neurolang.utils.server.engines import (
    NeurolangEngineConfiguration,
    NeurolangEngineSet,
    NeurosynthEngineConf,
)
from neurolang.utils.server.responses import (
    CustomQueryResultsEncoder,
    QueryResults,
)
from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
LOG.addHandler(ch)
LOG.propagate = False

logger = logging.getLogger("neurolang")
logger.propagate = False


class NeurolangQueryManager:
    """
    Class to manage execution of queries and keep track of results.
    """

    executor = None
    engines = {}
    _lock = RLock()
    results_cache = {}

    def __init__(
        self, options: Dict[NeurolangEngineConfiguration, int]
    ) -> None:
        """
        By default, the query manager has nb_engines workers, but no engines.
        It will call the _init_engines to create `nb_engines` engines
        asynchronously.

        Parameters
        ----------
        options : Dict[NeurolangEngineConfiguration, int]
            a dictionnary defining the types of engines and the number of each
            type to create.
        """
        nb_engines = sum(options.values())
        LOG.debug(f"Creating query manager with {nb_engines} engines.")
        self.executor = ThreadPoolExecutor(max_workers=nb_engines)
        self._init_engines(options)

    def _init_engines(self, options: Dict[NeurolangEngineConfiguration, int]):
        """
        Dispatch a series of tasks to create `nb_engines` engines using the
        `engine_create_func` function.

        Parameters
        ----------
        nb_engines : int
            the nb of engines to create. Also the number of workers.
        engine_create_func : callable
            a function to create a new engine. Should return a Neurolang instance.
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

    def _engine_created(self, future: Future):
        """
        Callback called when an engine creation task has finished.

        Parameters
        ----------
        future : concurrent.futures.Future
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

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            the result of the query execution
        """
        LOG.debug(f"[Thread - {get_ident()}] - Executing query...")
        LOG.debug(f"[Thread - {get_ident()}] - Query : {query}")
        engine_set = self.engines[engine_type]
        with engine_set.sema:
            engine = engine_set.pop()
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
                engine_set.add(engine)
                LOG.debug(
                    f"[Thread - {get_ident()}] - Engine of type {engine_type} released."
                )

    def submit_query(self, uuid: str, query: str, engine_type: str) -> Future:
        """
        Submit the query to one of the available engines / workers.

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
        return self.results_cache[uuid]


class Application(tornado.web.Application):
    """
    tornado Application. Defines the routes which the application listens on.
    Holds the NeurolangQueryManager which manages queries / results.

    Parameters
    ----------
    nqm : NeurolangQueryManager
        the query manager
    """

    def __init__(self, njm):
        self.njm = njm
        uuid_pattern = (
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        )
        handlers = [
            (r"/", MainHandler),
            (r"/v1/empty", EmptyHandler),
            (
                r"/v1/cancel/({uuid})".format(uuid=uuid_pattern),
                CancelHandler,
            ),
            (
                r"/v1/status/({uuid})".format(uuid=uuid_pattern),
                StatusHandler,
            ),
            (
                r"/v1/statement",
                QueryHandler,
            ),
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            debug=True,
        )
        super().__init__(handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    """
    Serve the index page.
    """

    def get(self):
        self.render("index.html")


class EmptyHandler(tornado.web.RequestHandler):
    """
    Helper endpoint returning an empty
    result.
    """

    def get(self):
        return self.write({"id": str(uuid4())})


class CancelHandler(tornado.web.RequestHandler):
    """
    Cancel an already running computation
    """

    def delete(self, uuid: str):
        LOG.debug(f"Canceling the request with uuid {uuid}")
        return self.write({"status": "ok"})


class JSONRequestHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    def write_json_reponse(self, data=None, status: str = "ok"):
        response = {"status": status}
        if data is not None:
            response["data"] = data
        return self.write(json.dumps(response, cls=CustomQueryResultsEncoder))


class StatusHandler(JSONRequestHandler):
    """
    Return the status (or the result) of an already running calculation.
    """

    async def get(self, uuid: str):
        LOG.debug(f"Accessing status for request {uuid}.")
        try:
            future = self.application.njm.get_result(uuid)
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=404, log_message="uuid not found"
            )
        return self.write_json_reponse(QueryResults(uuid, future))


class QueryHandler(JSONRequestHandler):
    """
    Main endpoint to submit a query.
    """

    async def post(self):
        query = self.get_argument("query")
        engine = self.get_argument("engine", "neurosynth")
        uuid = str(uuid4())
        LOG.debug(f"Submitting query with uuid {uuid}.")
        self.application.njm.submit_query(uuid, query, engine)
        return self.write_json_reponse({"query": query, "uuid": uuid})


def main():
    opts = {NeurosynthEngineConf(): 2}
    njm = NeurolangQueryManager(opts)

    tornado.options.parse_command_line()
    print(f"Tornado application starting on port {options.port}")
    app = Application(njm)
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
