import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import logging
from uuid import uuid4

from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)

LOG = logging.getLogger(__name__)


class Application(tornado.web.Application):
    def __init__(self):
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
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
            debug=True
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


class StatusHandler(tornado.web.RequestHandler):
    """
    Return the status (or the result) of an already running calculation
    """

    async def get(self, uuid: str):
        LOG.debug(f"Accessing the request with uuid {uuid}")
        return self.write({"status": "ok"})


class QueryHandler(tornado.web.RequestHandler):
    """
    Main endpoint to submit a query.
    """

    async def post(self):
        query = self.get_argument("body")
        uuid = str(uuid4())
        LOG.debug(f"Registering {query} with uuid {uuid}.")
        return self.write({"query": query, "uuid": uuid})


def main():
    tornado.options.parse_command_line()
    print(f"Tornado application starting on port {options.port}")
    app = Application()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
