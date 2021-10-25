import gzip
import json
import logging
import os
import os.path
import sys
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import matplotlib
import pandas as pd
import tornado.ioloop
import tornado.iostream
import tornado.options
import tornado.web
import tornado.websocket
import yaml
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from tornado.options import define, options

from .engines import DestrieuxEngineConf, NeurosynthEngineConf
from .queries import NeurolangQueryManager
from .responses import (
    CustomQueryResultsEncoder,
    QueryResults,
    base64_encode_nifti,
)

define("port", default=8888, help="run on the given port", type=int)
define(
    "data_dir",
    default=str(Path.home() / "neurolang_data"),
    help="path of a directory where the downloaded datasets are stored",
    type=str,
)
define(
    "npm_build",
    default=False,
    help="force a build of the frontend app",
    type=bool,
)

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
        static_path = str(
            Path(__file__).resolve().parent / "neurolang-web" / "dist"
        )
        print(f"Serving static files from {static_path}")

        handlers = [
            (r"/v1/empty", EmptyHandler),
            (r"/v1/engines", EnginesHandler),
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
                r"/v1/download/(.+)",
                DownloadsHandler,
            ),
            (
                r"/v1/figure/({uuid})".format(uuid=uuid_pattern),
                MpltFigureHandler,
            ),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": static_path, "default_filename": "index.html"},
            ),
        ]
        dev = (
            os.environ.get("NEUROLANG_SERVER_MODE") is None
            or os.environ.get("NEUROLANG_SERVER_MODE") == "dev"
        )
        cookie_secret = (
            os.environ.get("NEUROLANG_COOKIE_SECRET")
            or "WWdY+9ILT/u/7hZ24ubLYDA5I1lfgEQMuwRVMp9PY4U="
        )
        settings = dict(
            cookie_secret=cookie_secret,
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=static_path,
            xsrf_cookies=not dev,
            debug=dev,
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
        LOG.debug("Canceling the request with uuid %s.", uuid)
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
        LOG.debug("Accessing status for request %s.", uuid)
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
        LOG.debug("Accessing symbols for engine %s.", engine)
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

        LOG.debug("Submitting query with uuid %s.", self.uuid)
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
        LOG.debug("Submitting query with uuid %s.", uuid)
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
        atlas = self.application.nqm.get_atlas(engine)
        return self.write_json_reponse({"image": base64_encode_nifti(atlas)})


class DownloadsHandler(tornado.web.RequestHandler):
    """
    Handle requests for file downloads.
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/octet-stream")

    def save_image_to_gzip(
        self, df: pd.DataFrame, col: str, idx: str
    ) -> BytesIO:
        """
        Save an image as nii.gz to a bytes array. The image should be an
        ExplicitVBR or ExplicitVBROverlay object located in the given
        dataframe at idx, col position.

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe containing the images
        col : str
            the col index for the image in the df
        idx : str
            the row index for the image in the df

        Returns
        -------
        BytesIO
            the bytes buffer with the compressed image data

        Raises
        ------
        tornado.web.HTTPError
            404 if object at given col, row indices is not the right type.
        """
        image = df.iat[int(idx), int(col)]
        if not isinstance(image, (ExplicitVBR, ExplicitVBROverlay)):
            raise tornado.web.HTTPError(
                status_code=404, log_message="Invalid file format to download"
            )
        data = BytesIO(gzip.compress(image.spatial_image().dataobj.tobytes()))
        return data

    async def get(self, key: str):
        """
        Serve the file for the given key. Required query parameters are:
        - symbol: the symbol to download
        Optional query parameters:
        - col
        - idx
        the col and row index when a specific image file should be downloaded
        instead of the whole dataframe.

        Parameters
        ----------
        key : str
            the unique id key, either for a query or for an engine type

        Raises
        ------
        tornado.web.HTTPError
            404 if key is invalid, or results are not available
        """
        # 1. Get the symbol to download
        symbol = self.get_argument("symbol")
        col = self.get_argument("col", None)
        idx = self.get_argument("idx", None)
        try:
            future = self.application.nqm.get_result(key)
            LOG.debug(
                "Preparing image file for query %s and symbol %s.", key, symbol
            )
        except KeyError:
            try:
                future = self.application.nqm.get_symbols(key)
                LOG.debug(
                    f"Preparing image file for engine %s and symbol %s.",
                    key,
                    symbol,
                )
            except KeyError:
                raise tornado.web.HTTPError(
                    status_code=404, log_message="uuid not found"
                )
        if not (future.done() and future.result()):
            raise tornado.web.HTTPError(
                status_code=404, log_message="query does not have results"
            )
        results = future.result()
        ras = results[symbol]
        df = ras.as_pandas_dataframe()

        # 2. Write the object as gzip bytes
        if col is not None and idx is not None:
            data = self.save_image_to_gzip(df, col, idx)
            filename = symbol + "_" + idx + ".nii.gz"
        else:
            data = BytesIO(
                gzip.compress(
                    df.to_csv(index=False, compression="gzip").encode()
                )
            )
            filename = symbol + ".csv.gz"

        # 3. Stream the data in chunks to avoid blocking the server
        self.set_header(
            "Content-Disposition", "attachment; filename=" + filename
        )
        chunk_size = 1024 * 1024 * 1  # 1 MiB
        while True:
            chunk = data.read(chunk_size)
            if not chunk:
                break
            try:
                self.write(chunk)  # write the chunk to response
                await self.flush()  # send the chunk to client
            except tornado.iostream.StreamClosedError:
                break
            finally:
                del chunk


class MpltFigureHandler(tornado.web.RequestHandler):
    """
    Handle requests for matplotlib figures. Streams them as svg.
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "image/svg+xml")

    async def get(self, uuid: str):
        """
        Serve the matplotlib figure requested. Required query parameters are:
        - symbol: the symbol containing the figures
        - col & row: the col and row index of the figure in the symbol's dataframe.

        Parameters
        ----------
        uuid : str
            the uuid of the query

        Raises
        ------
        tornado.web.HTTPError
            404 if uuid is invalid, or results are not available
        """
        # 1. Get the result dataframe for the query params
        symbol = self.get_argument("symbol")
        row = self.get_argument("row")
        col = self.get_argument("col")
        format = self.get_argument("format", "svg")
        LOG.debug("Accessing figure result for request %s.", uuid)
        try:
            future = self.application.nqm.get_result(uuid)
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=404, log_message="uuid not found"
            )
        if not (future.done() and future.result()):
            raise tornado.web.HTTPError(
                status_code=404, log_message="query does not have results"
            )
        results = future.result()
        ras = results[symbol]
        df = ras.as_pandas_dataframe()
        figure = df.iat[int(row), int(col)]
        if not isinstance(figure, matplotlib.figure.Figure):
            raise tornado.web.HTTPError(
                status_code=404, log_message="Invalid figure format"
            )

        # 2. Stream the figure in requested format. If format is not svg
        # we serve it as an attached file
        if format != "svg":
            filename = f"{symbol}_{row}_{col}.{format}"
            self.set_header(
                "Content-Disposition", "attachment; filename=" + filename
            )
        data = BytesIO()
        figure.savefig(data, format=format)
        data.seek(0)
        chunk_size = 1024 * 1024 * 1  # 1 MiB
        while True:
            chunk = data.read(chunk_size)
            if not chunk:
                break
            try:
                self.write(chunk)  # write the chunk to response
                await self.flush()  # send the chunk to client
            except tornado.iostream.StreamClosedError:
                break
            finally:
                del chunk


class EnginesHandler(JSONRequestHandler):
    def get(self):
        engines = [e.key for e in self.application.nqm.configs.keys()]

        dirs = [
            os.path.join(sys.prefix, "queries"),
            os.path.dirname(os.path.realpath(__file__)),
        ]
        for d in dirs:
            queries_file = os.path.join(d, "queries.yaml")
            if os.path.isfile(queries_file):
                LOG.info(
                    "Reading queries configuration file for Neurolang: %s",
                    queries_file,
                )
                with open(queries_file, "r") as stream:
                    queries = yaml.safe_load(stream)
                break

        data = []
        for engine in engines:
            res = {"engine": engine}
            if engine in queries:
                res["queries"] = queries[engine]
            data.append(res)
        self.write_json_reponse(data)


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
    LOG.setLevel(logging.DEBUG)


def main():
    tornado.options.parse_command_line()
    setup_logs()
    data_dir = Path(options.data_dir)
    LOG.info(f"Neurolang data directory set to {data_dir}")
    opts = {
        NeurosynthEngineConf(data_dir, resolution=2): 2,
        DestrieuxEngineConf(data_dir): 2,
    }
    nqm = NeurolangQueryManager(opts)

    print(
        f"Tornado application starting on http://localhost:{options.port}/ ..."
    )
    app = Application(nqm)
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
