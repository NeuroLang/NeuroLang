from neurolang.utils.server.responses import CustomQueryResultsEncoder, QueryResults
import threading
from neurolang.utils.server.engines import init_frontend, load_neurosynth_data
from threading import BoundedSemaphore
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import logging
import json
from uuid import uuid4
import concurrent.futures
from tornado.options import define, options


define("port", default=8888, help="run on the given port", type=int)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class NeurolangJobManager:

    executor = None
    engines = set()
    engines_sema = None
    results_cache = {}

    def __init__(self, engines) -> None:
        LOG.debug(
            f"Initializing NeurolangJobManager with {len(engines)} engines."
        )
        for engine in engines:
            self.engines.add(engine)
        self.engines_sema = BoundedSemaphore(value=len(engines))
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(engines)
        )

    def execute_neurolang_query(self, query):
        LOG.debug(f"[{threading.get_ident()}] - Executing query...")
        with self.engines_sema:
            engine = self.engines.pop()
            LOG.debug(f"[{threading.get_ident()}] - Engine acquired.")
            try:
                with engine.scope:
                    LOG.debug(f"[{threading.get_ident()}] - Solving query...")
                    res = engine.execute_datalog_program(query)
                    if res is None:
                        res = engine.solve_all()
                    LOG.debug(f"[{threading.get_ident()}] - Query solved.")
                    return res
            finally:
                self.engines.add(engine)
                LOG.debug(f"[{threading.get_ident()}] - Engine released.")

    def submit_query(self, uuid, query):
        LOG.debug(f"Submitting query with uuid {uuid} to executor pool.")
        future_res = self.executor.submit(self.execute_neurolang_query, query)
        self.results_cache[uuid] = future_res

    def get_result(self, uuid):
        return self.results_cache[uuid]


class Application(tornado.web.Application):
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
    Return the status (or the result) of an already running calculation
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
        query = self.get_argument("body")
        uuid = str(uuid4())
        LOG.debug(f"Submitting query with uuid {uuid}.")
        self.application.njm.submit_query(uuid, query)
        return self.write_json_reponse({"query": query, "uuid": uuid})


def setup_job_manager(nb_engines):
    engines = []
    for _ in range(nb_engines):
        nl = init_frontend()
        load_neurosynth_data(nl)
        engines.append(nl)

    njm = NeurolangJobManager(engines)
    return njm


def main():
    tornado.options.parse_command_line()

    njm = setup_job_manager(2)

    print(f"Tornado application starting on port {options.port}")
    app = Application(njm)
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
