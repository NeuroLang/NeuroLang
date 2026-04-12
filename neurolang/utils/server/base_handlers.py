"""
Base handlers shared by app.py and v2_handlers.py.

Keeping shared handler base classes in a separate module avoids circular
import issues: v2_handlers.py imports from here, and app.py also imports
from here (and from v2_handlers.py).
"""
import json

import tornado.web

from .responses import CustomQueryResultsEncoder


def query_results_to_json(data=None, status: str = "ok") -> str:
    """Serialise *data* and *status* to a JSON string.

    Parameters
    ----------
    data : Any, optional
        Response payload (must be JSON-serialisable by
        :class:`~.responses.CustomQueryResultsEncoder`).
    status : str, optional
        Status string, by default ``"ok"``.

    Returns
    -------
    str
        JSON-encoded response string.
    """
    response = {"status": status}
    if data is not None:
        response["data"] = data
    return json.dumps(response, cls=CustomQueryResultsEncoder)


class JSONRequestHandler(tornado.web.RequestHandler):
    """Base handler that sets CORS / JSON headers and provides a
    :meth:`write_json_reponse` helper.

    Note: the method name preserves the (intentional) typo present in the
    existing codebase for backward compatibility.
    """

    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header(
            "Access-Control-Allow-Methods", "POST, GET, OPTIONS"
        )
        self.set_header("Content-Type", "application/json")

    def write_json_reponse(self, data=None, status: str = "ok"):
        """Write a JSON response with the given *data* payload.

        Parameters
        ----------
        data : Any, optional
            Response payload.
        status : str, optional
            Status string, by default ``"ok"``.
        """
        return self.write(query_results_to_json(data, status))
