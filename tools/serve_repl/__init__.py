"""
Serve the REPL in the browser on this machine.

The page (``doc/_static/repl``) is static: Python runs in the browser, on
Pyodide. This script puts it, with the logo, in SITE, builds its bundle
there with ``make_repl_bundle.py`` (NeuroLang's wheels and the engine's
data; once, or again with ``--rebuild``, or from nothing with ``--clear``,
which first removes the bundle built before), and serves SITE on this
machine, telling the browser not to cache, so that a rebuild is seen on
reload.
The browser still loads Pyodide from its CDN.

Usage, from a checkout::

    uv run --extra apps neurolang-repl-serve [--port 8766] [--rebuild]
        [--engine NAME ...] [--clear]
        [--embeddings DIR] [--site SITE] [--host 0.0.0.0] [--no-browser]

the command of the workspace's root project, or ``python -m serve_repl``.
The page offers the engines ``neurosynth_iit`` (Neurosynth with the tracts
of the IIT atlas, which it starts with), ``neurosynth`` and ``neurostore`` (a
select at its top, or ``?engine=neurostore``): the bundle has the data of
all, unless
``--engine`` names the ones to build, and is built again when it lacks one that
is asked for. It opens the page in the browser. The text embeddings of
``query_similarity`` are kept in ``--embeddings`` (by default
``~/.cache/neurolang/repl-embeddings``): the bundle has them if they are
there, or if torch is installed to make them (``--extra embeddings``),
which takes a while the first time; otherwise the page goes without.

The REPL with NeuroLang in the Python of this machine::

    uv run --extra apps neurolang-repl-serve --local [--data-dir DIR]
        [--resolution MM] [--light] [--engine NAME ...]

builds no bundle: the page's worker sends what it is asked to this server,
which builds the engine from DIR (``neurolang_data`` by default, as
``neurolang-query``), with all of its analyses unless ``--light``, and runs
the REPL (``neurolang_apps.utils.tui.local_backend``). The page is the same,
in SITE/local, and it is as fast, and has as much memory, as this machine.
It serves this machine only, unless ``--host``: anyone it serves can run
queries, and save files, in its Python.

The REPL of a pull request, built by its CI::

    uv run --extra apps neurolang-repl-serve --pr N
    uv run --extra apps neurolang-repl-serve --run-id ID

builds nothing: it downloads the site (the documentation and the REPL) that
the documentation's workflow uploaded for the latest run of pull request N
(the artifact ``repl-pr-N``), or for the run ID, into SITE/pr-N (or
SITE/run-ID), unless it is there already, and serves it. It needs the
GitHub CLI, ``gh``, logged in (``gh auth login``).
"""

import argparse
import http.server
import importlib.util
import json
import queue
import shutil
import socketserver
import subprocess
import sys
import urllib.parse
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
# The repository whose CI uploads the site of each pull request, and the
# artifacts holding it (.github/workflows/docs.yml).
REPOSITORY = "NeuroLang/NeuroLang"
ARTIFACTS = ("repl-pr-", "documentation-html")
# The run whose site a downloaded site is, in that site.
RUN_FILE = ".neurolang-run"


def bundle_script():
    """Return the module of the bundle script."""
    spec = importlib.util.spec_from_file_location(
        "make_repl_bundle", BUNDLE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle_engines() -> tuple:
    """Return the engines the bundle script builds by default."""
    return bundle_script().ENGINES


def built_engines(repl: Path) -> set:
    """Return the engines of the bundle in *repl* (none if it is not there)."""
    try:
        bundle = json.loads((repl / "bundle.json").read_text())
    except (OSError, ValueError):
        return set()
    return set(bundle.get("engines") or ["neurosynth"])


def build_bundle(repl: Path, embeddings, engines=None) -> None:
    """Build the page's bundle in *repl*, with the text embeddings of the
    data directory *embeddings* if given, and the data of the *engines*
    (by default all those of the page)."""
    command = [sys.executable, str(BUNDLE_SCRIPT), str(repl)]
    if embeddings:
        command.append(str(embeddings))
    for engine in engines or ():
        command += ["--engine", engine]
    subprocess.check_call(command)


def prepare_site(
    site: Path, rebuild: bool = False, embeddings=None, clear: bool = False,
    engines=None,
) -> Path:
    """
    Put the page and the logo in *site*, build the bundle if it is not
    there, lacks one of the *engines* (by default all of the page's) or
    *rebuild*, and return the path of the page.

    The page is copied each time, so that it is that of the checkout.
    *clear* removes the page and the bundle built before (wheels, data of
    the engine and its kept tables, embeddings files), then builds it
    again; the text embeddings kept in *embeddings* stay.
    """
    repl = site / "_static" / "repl"
    if clear:
        shutil.rmtree(repl, ignore_errors=True)
        rebuild = True
    repl.mkdir(parents=True, exist_ok=True)
    wanted = set(engines or bundle_engines())
    if rebuild or not wanted <= built_engines(repl):
        build_bundle(repl, embeddings, engines)
    return copy_page(site)


def copy_page(site: Path) -> Path:
    """Copy the page of the checkout and the logo into *site*; return the
    path of the page."""
    repl = site / "_static" / "repl"
    repl.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOGO, site / "_static" / "logo", dirs_exist_ok=True)
    for path in PAGE.iterdir():
        if path.is_file():
            shutil.copy(path, repl / path.name)
    return repl / "index.html"


def prepare_local_site(site: Path, engines=None) -> Path:
    """
    Put the page in *site* with a ``bundle.json`` telling it to have this
    machine's Python answer it (``"backend": "local"``), offering the
    *engines* (by default all of the page's), and return the path of the
    page.
    """
    script = bundle_script()
    engines = list(engines or script.ENGINES)
    page = copy_page(site)
    (page.parent / "bundle.json").write_text(json.dumps({
        "backend": "local",
        "engines": {
            name: {"label": script.LABELS.get(name, name)}
            for name in engines
        },
        "default": engines[0],
    }) + "\n")
    script.write_highlight_words(page.parent)
    return page


def feedback_target() -> tuple:
    """Return where the page's rated answers go (its ``feedback.json``),
    or None."""
    try:
        target = json.loads((PAGE / "feedback.json").read_text())
    except (OSError, ValueError):
        return None
    if not target.get("url"):
        return None
    return target["url"], target.get("doc", "")


def gh(*args) -> str:
    """Return the output of the GitHub CLI run with *args*; stop, telling
    why, if it is not installed or fails."""
    if shutil.which("gh") is None:
        sys.exit("--pr and --run-id need the GitHub CLI, gh "
                 "(https://cli.github.com), logged in: gh auth login")
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def latest_run(pr: int, repository: str = REPOSITORY):
    """Return the run and the commit of the latest site uploaded for the
    pull request *pr* that has not expired."""
    found = gh(
        "api", f"repos/{repository}/actions/artifacts"
        f"?name={ARTIFACTS[0]}{pr}&per_page=100",
        "--jq", "[.artifacts[] | select(.expired | not)] | sort_by("
        ".created_at) | last | values | "
        "\"\\(.workflow_run.id) \\(.workflow_run.head_sha)\"",
    )
    if not found:
        sys.exit(
            f"No site of pull request {pr} in {repository}: its "
            "documentation's run has not finished, or its artifact expired "
            f"(see https://github.com/{repository}/pull/{pr}/checks)."
        )
    run, commit = found.split()
    return int(run), commit


def run_artifact(run: int, repository: str = REPOSITORY) -> str:
    """Return the name of the artifact holding the site of the run *run*."""
    names = gh(
        "api", f"repos/{repository}/actions/runs/{run}/artifacts",
        "--jq", ".artifacts[] | select(.expired | not) | .name",
    ).split()
    name = next((n for n in names if n.startswith(ARTIFACTS)), None)
    if name is None:
        sys.exit(f"The run {run} of {repository} has no site: "
                 f"{', '.join(names) or 'no artifacts'}.")
    return name


def download_site(
    site: Path, run: int, name: str, repository: str = REPOSITORY
) -> Path:
    """Download the artifact *name* of the run *run* into *site*, unless it
    is there already, and return the path of the page."""
    if not (site / RUN_FILE).is_file() or \
            (site / RUN_FILE).read_text() != f"{run}":
        shutil.rmtree(site, ignore_errors=True)
        print(f"Downloading the artifact {name} of the run {run} into {site}",
              flush=True)
        gh("run", "download", str(run), "-R", repository, "-n", name,
           "-D", str(site))
        (site / RUN_FILE).write_text(f"{run}")
    page = site / "_static" / "repl" / "index.html"
    if not (site / "_static" / "repl" / "bundle.json").is_file():
        print(f"The run {run} built no bundle for the REPL: its page will "
              "say that it cannot start.", flush=True)
    return page


def fetch_site(args) -> Path:
    """Download the site of the pull request or the run of *args* into a
    directory of its own in the site directory, which becomes
    ``args.site``; return the page."""
    run, name = args.run_id, None
    if run is None:
        run, commit = latest_run(args.pr, args.repo)
        name = f"{ARTIFACTS[0]}{args.pr}"
        print(f"The REPL of pull request {args.pr}, built from {commit} "
              f"(https://github.com/{args.repo}/actions/runs/{run})",
              flush=True)
    else:
        name = run_artifact(run, args.repo)
    args.site = args.site / (
        f"pr-{args.pr}" if args.run_id is None else f"run-{run}"
    )
    return download_site(args.site, run, name, args.repo)


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


class LocalHandler(Handler):
    """
    Serves the page, and answers its worker (``local_worker.js``) with the
    REPL of the server's *backend*, a
    :class:`~neurolang_apps.utils.tui.local_backend.LocalBackend`:
    ``POST /api/ACTION`` with a JSON body, and ``GET /api/events``, the
    messages of the REPL while it works, as server-sent events.
    """

    backend = None
    # How often an idle stream of events tells it is alive, in seconds.
    KEEPALIVE_SECONDS = 15

    def _send(self, status: int, body: bytes, mime: str, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status: int = 200):
        self._send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        if self.path != "/api/events":
            return super().do_GET()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        listener = self.backend.listen()
        try:
            while True:
                try:
                    text = listener.get(timeout=self.KEEPALIVE_SECONDS)
                    self.wfile.write(f"data: {text}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.backend.forget(listener)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        action = self.path[len("/api/"):]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self._answer(action, body)
        except Exception as error:  # told to the page
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def _answer(self, action: str, body: dict):
        backend = self.backend
        if action == "stop":
            return self._json({"stopped": backend.stop()})
        if action == "start":
            return self._json(backend.start(body["engine"]))
        if action == "submit":
            return self._json(backend.submit(body["text"]))
        if action == "llm":
            backend.set_llm(**body)
            return self._json({})
        if action == "rate":
            return self._json(backend.rate(**body))
        if action == "download":
            file = backend.download(**body)
            return self._send(200, file.data, file.mime, [
                ("X-File-Name", urllib.parse.quote(file.name)),
            ])
        return self._json({"error": f"no action {action}"}, 404)


class Server(http.server.ThreadingHTTPServer):
    """An HTTP server that does not look up the host's name, which
    ``HTTPServer`` does when it binds and may take half a minute."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(site: Path, host: str, port: int, backend=None):
    """Return the HTTP server of *site* on *host*:*port*, answering the
    page with the REPL of *backend* if given."""
    handler = Handler
    if backend is not None:
        handler = type("Handler", (LocalHandler,), {"backend": backend})
    return Server(
        (host, port),
        lambda *args, **kwargs: handler(*args, directory=str(site), **kwargs),
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
        "--clear", action="store_true",
        help="remove the bundle built before, then build it again (the "
        "text embeddings in --embeddings are kept)",
    )
    parser.add_argument(
        "--embeddings", type=Path, default=DEFAULT_EMBEDDINGS,
        help="data directory keeping the text embeddings (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--engine", action="append", choices=bundle_engines(), dest="engines",
        help="an engine to put in the bundle (repeat for several; default: "
        "all of them)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--local", action="store_true",
        help="run NeuroLang in the Python of this machine rather than in "
        "the browser: no bundle is built",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("neurolang_data"),
        help="with --local, the data directory of the engines (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--resolution", type=float, default=2.0, metavar="MM",
        help="with --local, the resolution of the engines' MNI grid "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="with --local, leave out the analyses a browser cannot hold, "
        "as the page does",
    )
    built = parser.add_mutually_exclusive_group()
    built.add_argument(
        "--pr", type=int,
        help="serve the REPL that CI built for this pull request",
    )
    built.add_argument(
        "--run-id", type=int, help="serve the REPL that this CI run built",
    )
    parser.add_argument(
        "--repo", default=REPOSITORY,
        help="the repository of --pr and --run-id (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if (args.pr or args.run_id) and (args.rebuild or args.clear):
        parser.error("--pr and --run-id build nothing: no --rebuild nor "
                     "--clear")
    if args.local and (args.pr or args.run_id or args.rebuild
                       or args.clear):
        parser.error("--local builds no bundle: no --pr, --run-id, "
                     "--rebuild nor --clear")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    backend = None
    if args.local:
        from neurolang_apps.utils.tui.local_backend import LocalBackend

        args.site = args.site / "local"
        page = prepare_local_site(args.site, args.engines)
        backend = LocalBackend(
            args.data_dir, args.resolution, light=args.light,
            feedback=feedback_target(),
        )
        print(f"NeuroLang runs in {sys.executable}, on the data of "
              f"{args.data_dir.resolve()}", flush=True)
    elif args.pr is not None or args.run_id is not None:
        page = fetch_site(args)
    else:
        page = prepare_site(
            args.site, args.rebuild, args.embeddings, clear=args.clear,
            engines=args.engines,
        )
    server = make_server(args.site, args.host, args.port, backend)
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
