"""
Links that open the REPL in the browser with a query in its box.

A link's fragment is made as ``encodeShare`` of
``doc/_static/repl/share.js`` makes it, which the page reads on load; the
page runs nothing until Run. No dependency, so that the tests of the
packages check it against the page.
"""

import base64
from urllib.parse import urlencode

REPL_PAGE = "_static/repl/index.html"


def share_fragment(code: str, mode: str = "datalog") -> str:
    """Return the fragment of a link with *code* in *mode* in the REPL's box."""
    query = base64.urlsafe_b64encode(code.encode()).decode().rstrip("=")
    params = {"q": query}
    if mode != "datalog":
        params["mode"] = mode
    return urlencode(params)


def repl_link(code: str, mode: str, engine: str, depth: int = 0) -> str:
    """
    Return the link, from a page *depth* folders deep, to run *code* on
    *engine*, named even when it is the page's default, which may change.
    """
    page = "../" * depth + REPL_PAGE + "?" + urlencode({"engine": engine})
    return f"{page}#{share_fragment(code, mode)}"
