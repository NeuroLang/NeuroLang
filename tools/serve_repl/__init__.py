"""
Serve the REPL in the browser on this machine.

The page (``doc/_static/repl``) is static: Python runs in the browser, on
Pyodide. This script puts it, with the logo, in SITE, builds its bundle
there with ``make_repl_bundle.py`` (NeuroLang's wheels and the engine's
data; once, or again with ``--rebuild``), and serves SITE on this machine,
telling the browser not to cache, so that a rebuild is seen on reload.
The browser still loads Pyodide from its CDN.

Usage, from a checkout::

    uv run --extra apps neurolang-repl-serve [--port 8766] [--rebuild]
        [--embeddings DIR] [--site SITE] [--host 0.0.0.0] [--no-browser]

the command of the workspace's root project, or ``python -m serve_repl``.
It opens the page in the browser. The text embeddings of
``query_similarity`` are kept in ``--embeddings`` (by default
``~/.cache/neurolang/repl-embeddings``): the bundle has them if they are
there, or if torch is installed to make them (``--extra embeddings``),
which takes a while the first time; otherwise the page goes without.
"""

import argparse
import http.server
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "doc"
BUNDLE_SCRIPT = DOC / "make_repl_bundle.py"
PAGE = DOC / "_static" / "repl"
LOGO = DOC / "_static" / "logo"
DEFAULT_SITE = Path("~/.cache/neurolang/repl-site").expanduser()
# Where the text embeddings are kept between builds: making them takes
# torch, the abstracts of the studies from PubMed and a while.
DEFAULT_EMBEDDINGS = Path("~/.cache/neurolang/repl-embeddings").expanduser()


def build_bundle(repl: Path, embeddings) -> None:
    """Build the page's bundle in *repl*, with the text embeddings of the
    data directory *embeddings* if given."""
    command = [sys.executable, str(BUNDLE_SCRIPT), str(repl)]
    if embeddings:
        command.append(str(embeddings))
    subprocess.check_call(command)


def prepare_site(site: Path, rebuild: bool = False, embeddings=None) -> Path:
    """
    Put the page and the logo in *site*, build the bundle if it is not
    there or *rebuild*, and return the path of the page.

    The page is copied each time, so that it is that of the checkout.
    """
    repl = site / "_static" / "repl"
    repl.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOGO, site / "_static" / "logo", dirs_exist_ok=True)
    if rebuild or not (repl / "bundle.json").exists():
        build_bundle(repl, embeddings)
    for path in PAGE.iterdir():
        if path.is_file():
            shutil.copy(path, repl / path.name)
    return repl / "index.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves files uncached, the page's scripts as JavaScript modules."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".whl": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # the downloads of the page are many and of no interest


class Server(http.server.ThreadingHTTPServer):
    """An HTTP server that does not look up the host's name, which
    ``HTTPServer`` does when it binds and may take half a minute."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(site: Path, host: str, port: int):
    """Return the HTTP server of *site* on *host*:*port*."""
    return Server(
        (host, port),
        lambda *args, **kwargs: Handler(*args, directory=str(site), **kwargs),
    )


def parse_args(argv=None):
    """Return the options of the command line *argv*."""
    parser = argparse.ArgumentParser(
        prog="neurolang-repl-serve", description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="0.0.0.0 to serve the other machines of the network too",
    )
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--rebuild", action="store_true",
                        help="build the bundle again, from this checkout")
    parser.add_argument(
        "--embeddings", type=Path, default=DEFAULT_EMBEDDINGS,
        help="data directory keeping the text embeddings (default: "
        "%(default)s)",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    page = prepare_site(args.site, args.rebuild, args.embeddings)
    server = make_server(args.site, args.host, args.port)
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = (f"http://{host}:{server.server_address[1]}/"
           f"{page.relative_to(args.site).as_posix()}")
    print(f"The REPL is at {url} (Ctrl+C stops it)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
