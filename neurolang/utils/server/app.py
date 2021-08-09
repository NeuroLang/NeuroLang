import json
import logging
from neurolang.utils.server.queries import NeurolangQueryManager
import os.path
from concurrent.futures import Future
from uuid import uuid4

import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
from neurolang.utils.server.engines import (
    DestrieuxEngineConf,
    NeurosynthEngineConf,
)
from neurolang.utils.server.responses import (
    CustomQueryResultsEncoder,
    QueryResults,
    base64_encode_nifti,
)
from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)

LOG = logging.getLogger(__name__)


class Application(tornado.web.Application):
    """
    tornado Application. Defines the routes which the application listens on.
    Holds the NeurolangQueryManager which manages queries / results.

    Parameters
    ----------
    nqm : NeurolangQueryManager
        the query manager
    """

    def __init__(self, nqm: NeurolangQueryManager):
        self.nqm = nqm
        uuid_pattern = (
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        )
        static_path = os.path.join(
            os.path.dirname(__file__), "neurolang-web/dist"
        )

        handlers = [
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
                r"/v1/symbol/(.+)",
                SymbolsHandler,
            ),
            (
                r"/v1/statement",
                QueryHandler,
            ),
            (
                r"/v1/statementsocket",
                QuerySocketHandler,
            ),
            (
                r"/v1/atlas",
                NiftiiImageHandler,
            ),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": static_path, "default_filename": "index.html"},
            ),
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=static_path,
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
    Cancel an already running computation.
    """

    def delete(self, uuid: str):
        LOG.debug(f"Canceling the request with uuid {uuid}.")
        result = self.application.nqm.cancel(uuid)
        return self.write_json_reponse({"cancelled": result})


def query_results_to_json(data=None, status: str = "ok"):
    response = {"status": status}
    if data is not None:
        response["data"] = data
    return json.dumps(response, cls=CustomQueryResultsEncoder)


class JSONRequestHandler(tornado.web.RequestHandler):
    """
    Base Handler for writing JSON responses.
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/json")

    def write_json_reponse(self, data=None, status: str = "ok"):
        return self.write(query_results_to_json(data, status))


class StatusHandler(JSONRequestHandler):

    async def get(self, uuid: str):
        """
        Return the status (or the result) of an already running calculation.
        Optional query parameters are :
            * symbol : return only the values for this symbol
            * start : index of the first row to return
            * length : number of rows to return
            * sort : the index of the column to sort by
            * asc : sort by ascending (true) or descending (false)

        Parameters
        ----------
        uuid : str
            The query id

        Returns
        -------
        QueryResults
            The query results.

        Raises
        ------
        tornado.web.HTTPError
            raises 404 if query id does not exist.
        """
        LOG.debug(f"Accessing status for request {uuid}.")
        try:
            future = self.application.nqm.get_result(uuid)
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=404, log_message="uuid not found"
            )
        symbol = self.get_argument("symbol", None)
        start = int(self.get_argument("start", 0))
        length = int(self.get_argument("length", 50))
        sort = int(self.get_argument("sort", -1))
        asc = bool(int(self.get_argument("asc", 1)))
        return self.write_json_reponse(
            QueryResults(uuid, future, symbol, start, length, sort, asc)
        )


class SymbolsHandler(JSONRequestHandler):

    def get(self, engine: str):
        """
        Return the symbols available on an engine.
        This method is syncronous and will block until an engine is
        available to get its symbols.

        Optional query parameters are :
            * symbol : return only the values for this symbol
            * start : index of the first row to return
            * length : number of rows to return
            * sort : the index of the column to sort by
            * asc : sort by ascending (true) or descending (false)

        Parameters
        ----------
        engine : str
            The engine id

        Returns
        -------
        QueryResults
            A dict of query results

        Raises
        ------
        tornado.web.HTTPError
            raises 404 if engine id does not exist
        """
        LOG.debug(f"Accessing symbols for engine {engine}.")
        try:
            symbols = self.application.nqm.get_symbols(engine)
            # Block until results are available
            symbols.result()
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=404, log_message="engine not found"
            )
        uuid = f"ENGINE_SYMBOLS_{engine}"
        symbol = self.get_argument("symbol", None)
        start = int(self.get_argument("start", 0))
        length = int(self.get_argument("length", 50))
        sort = int(self.get_argument("sort", -1))
        asc = bool(int(self.get_argument("asc", 1)))
        return self.write_json_reponse(
            QueryResults(uuid, symbols, symbol, start, length, sort, asc)
        )


class QuerySocketHandler(tornado.websocket.WebSocketHandler):
    """
    Main handler to submit a query using a websocket.
    """

    def check_origin(self, origin):
        """
        Allow cross-origin websockets.
        """
        return True

    async def on_message(self, message):
        """
        Upon receiving a message, extract the query from it and generate
        a uuid. Submit the query to the NeurolangQueryManager and set
        `send_query_update` as the callback for when the query execution
        completes.

        Parameters
        ----------
        message : str
            a serialized JSON representation of an object containing a query.
        """
        parsed = tornado.escape.json_decode(message)
        query = parsed["query"]
        engine = parsed.get("engine", "neurosynth")
        self.uuid = str(uuid4())

        LOG.debug(f"Submitting query with uuid {self.uuid}.")
        future = self.application.nqm.submit_query(self.uuid, query, engine)
        tornado.ioloop.IOLoop.current().add_future(
            future, self.send_query_update
        )
        self.send_query_update(future)

    def send_query_update(self, future: Future, status: str = "ok"):
        """
        Upon completion of a query, send a message with the results

        Parameters
        ----------
        future : Future
            the query results.
        status : str, optional
            the status of the query, by default "ok"
        """
        self.write_message(
            query_results_to_json(QueryResults(self.uuid, future), status)
        )


class QueryHandler(JSONRequestHandler):
    """
    Main endpoint to submit a query using a POST request.
    """

    async def post(self):
        query = self.get_argument("query")
        engine = self.get_argument("engine", "neurosynth")
        uuid = str(uuid4())
        LOG.debug(f"Submitting query with uuid {uuid}.")
        self.application.nqm.submit_query(uuid, query, engine)
        return self.write_json_reponse({"query": query, "uuid": uuid})


class NiftiiImageHandler(JSONRequestHandler):
    """
    Return the atlas image to be used by the Papaya viewer.
    Currently returns the MNI_MASK used by the engine configuration.

    Image is returned as base64 encoded.
    """

    def get(self):
        engine = self.get_argument("engine", "neurosynth")
        mni_mask = self.application.nqm.get_mni_mask(engine)

        return self.write_json_reponse(
            {"image": base64_encode_nifti(mni_mask)}
        )


def setup_logs():
    logger = logging.getLogger("neurolang.utils.server")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    logger = logging.getLogger("neurolang")
    logger.propagate = False


def main():
    setup_logs()
    opts = {NeurosynthEngineConf(resolution=2): 2, DestrieuxEngineConf(): 2}
    nqm = NeurolangQueryManager(opts)

    tornado.options.parse_command_line()
    print(
        f"Tornado application starting on http://localhost:{options.port}/ ..."
    )
    app = Application(nqm)
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
