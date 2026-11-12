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
        [--embeddings DIR] [--site SITE] [--no-browser]
        [--host 0.0.0.0 --server-name NAME ...]

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

    uv run --extra apps --extra tui --extra llm neurolang-repl-serve --local
        [--data-dir DIR]
        [--resolution MM] [--light] [--engine NAME ...]

builds no bundle: the page's worker sends what it is asked to this server,
which builds the engine from DIR (``neurolang_data`` by default, as
``neurolang-query``), with all of its analyses unless ``--light``, and runs
the REPL (``neurolang_apps.utils.tui.local_backend``). The page is the same,
in SITE/local, and it is as fast, and has as much memory, as this machine.
It serves this machine only, unless ``--host``: anyone it serves can run
queries, and save files, in its Python.

Without a password, the server answers only the requests sent to
``localhost``, ``127.0.0.1``, ``::1``, ``--host`` (unless ``0.0.0.0`` or
``::``) and the names of ``--server-name NAME`` (repeated for several): a
page of another site, whose name it made lead to this machine (DNS
rebinding), is refused. Those of the network need ``--server-name`` with the
name or the address they use (``--host 0.0.0.0 --server-name
repl.lab.org``).

The REPL of a pull request, built by its CI::

    uv run --extra apps neurolang-repl-serve --pr N
    uv run --extra apps neurolang-repl-serve --run-id ID

builds nothing: it downloads the site (the documentation and the REPL) that
the documentation's workflow uploaded for the latest run of pull request N
(the artifact ``repl-pr-N``), or for the run ID, into SITE/pr-N (or
SITE/run-ID), unless it is there already, and serves it. It needs the
GitHub CLI, ``gh``, logged in (``gh auth login``).

HTTPS, with a certificate and its private key in PEM files (the key may be
in the certificate's file; its passphrase, if any, on the first line of
``--key-password-file``, else asked in a terminal)::

    uv run --extra apps neurolang-repl-serve --cert CERT.pem [--key KEY.pem]
        [--key-password-file FILE]

A password, on the first line of FILE (not on the command line; as it is,
but for its line ending, as the passphrase of ``--key-password-file``), that
the browser asks every visitor for (HTTP Basic, any user name; with ``--cert``
unless on this machine only); with ``--local``, the key of the AI assistant
that those with the password use, unless their page gives one: the
NAME=VALUE lines of ENV (the variable that the provider's ``api_key_env``
names, in ``~/.config/neurolang/llm.yaml``, e.g. ``ALBERT_API_KEY=...``),
put in the server's environment, so that the key never reaches a browser::

    uv run --extra apps --extra tui --extra llm neurolang-repl-serve --local
        --cert CERT.pem --key KEY.pem --password-file FILE [--env-file ENV]

The site's main page, ``/``, leads to the REPL (unless the site has its own,
the documentation's).

As a service of this user, started now and at each login, serving as the
other options say::

    uv run --extra apps neurolang-repl-serve --daemon install [OPTIONS]
    uv run --extra apps neurolang-repl-serve --daemon reinstall [OPTIONS]
    uv run --extra apps neurolang-repl-serve --daemon remove

a unit of systemd's user manager (``~/.config/systemd/user/``; ``loginctl
enable-linger`` keeps it running after logout), or a launchd agent on macOS
(``~/Library/LaunchAgents/``). ``install`` builds the bundle first, here;
the service runs ``python -m serve_repl`` with the Python of the environment
installing it (``uv run``'s ``.venv/bin/python``), or ``--python PATH``
(a name is looked for in the PATH), and its PATH, with no browser and the
paths made absolute; the other variables of this shell (API keys...) it
does not get. The service installed before is removed by ``reinstall``
only once the new one's bundle is built and its Python can serve; if the
service manager fails, the service is removed and the error told.
``remove`` needs no other option. A port below 1024 needs a service of the
system, which this does not install.

A connection that says nothing for a minute (its TLS handshake, its request)
is closed, and the body of a request to ``/api/`` is a JSON object of 1 MB at
most.
"""

import argparse
import base64
import getpass
import hmac
import http.server
import importlib.util
import json
import os
import plistlib
import queue
import re
import shlex
import shutil
import socketserver
import ssl
import subprocess
import sys
import time
import traceback
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
    # The engine the page starts with: that of the bundle's, if offered.
    default = getattr(script, "DEFAULT_ENGINE", engines[0])
    page = copy_page(site)
    (page.parent / "bundle.json").write_text(json.dumps({
        "backend": "local",
        "engines": {
            name: {"label": script.LABELS.get(name, name)}
            for name in engines
        },
        "default": default if default in engines else engines[0],
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

    #: The password every request must give (HTTP Basic, any user name),
    #: as bytes; None: none.
    password = None
    #: Without a password, the names of the host a request may be sent to
    #: (its ``Host``): another one may be a page of another site whose name
    #: now leads here (DNS rebinding).
    names = frozenset()
    #: How long a connection may say nothing (its TLS handshake, its
    #: request, its body), in seconds.
    timeout = 60
    #: The longest body of a request, in bytes.
    MAX_BODY = 1 << 20

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # the downloads of the page are many and of no interest

    def parse_request(self):
        """Read the request, and answer 401 to one without the password,
        or, if there is none, 403 to one sent to a host not of
        ``names``: then it goes no further, whatever its method."""
        if not super().parse_request():
            return False
        if self.password is None:
            try:
                host = urllib.parse.urlsplit(
                    "//" + (self.headers.get("Host") or "")).hostname
            except ValueError:  # an IPv6 address without its ]
                host = None
            if host in self.names:
                return True
            self.send_error(403, "Unknown host", (
                f"This server does not answer to {host}: if it is its name, "
                f"serve it with --server-name {host}, or with a password"
            ))
            return False
        if self._authorised():
            return True
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate", 'Basic realm="NeuroLang REPL", charset="UTF-8"'
        )
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _authorised(self) -> bool:
        scheme, _, encoded = (
            self.headers.get("Authorization") or ""
        ).partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            credentials = base64.b64decode(encoded.strip(), validate=True)
        except ValueError:
            return False
        given = credentials.partition(b":")[2]
        return hmac.compare_digest(given, self.password)

    def _body(self) -> dict:
        """Return the JSON object of the request's body, {} if it has none;
        raise ValueError, telling why, if it is not one (TooLarge if over
        MAX_BODY)."""
        length = self.headers.get("Content-Length")
        if length is None:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("a body needs a Content-Length")
            return {}
        size = int(length) if length.isascii() and length.isdigit() else -1
        if size < 0:
            raise ValueError(f"Content-Length {length!r} is not a length")
        if size > self.MAX_BODY:
            raise TooLarge(f"a body of {size} bytes is over "
                           f"{self.MAX_BODY} bytes")
        try:
            data = self.rfile.read(size)
        except TimeoutError:
            data = b""
        if len(data) < size:
            raise ValueError("the body is shorter than its Content-Length")
        try:
            body = json.loads(data or b"{}")  # a ValueError if not JSON
        except RecursionError:
            raise ValueError("the body nests too deep") from None
        if not isinstance(body, dict):
            raise ValueError("the body is not a JSON object")
        return body


class TooLarge(ValueError):
    """The body of a request is over ``Handler.MAX_BODY``."""


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
        except OSError:  # gone, or reads nothing for Handler.timeout
            pass
        finally:
            self.backend.forget(listener)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        action = self.path[len("/api/"):]
        try:
            body = self._body()
        except ValueError as error:
            return self._json({"error": str(error)},
                              413 if isinstance(error, TooLarge) else 400)
        try:
            self._answer(action, body)
        except Exception as error:  # told to the page, and in the log
            traceback.print_exc()
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

    def handle_error(self, request, client_address):
        """Log a client that fails its handshake (plain HTTP on the HTTPS
        port), goes or says nothing in one line, others with their
        traceback."""
        error = sys.exc_info()[1]
        if isinstance(error, (ssl.SSLError, ConnectionError, TimeoutError)):
            print(f"{client_address[0]}: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
        else:
            super().handle_error(request, client_address)


def tls_context(cert: Path, key=None, password_file=None,
                ask: bool = True) -> ssl.SSLContext:
    """
    Return the TLS context of a server with the certificate *cert*, a PEM
    file (its chain after it), and its private key, in *key* or else in
    *cert*. A key with a passphrase is opened with the first line of
    *password_file*, else the passphrase asked in a terminal, if *ask*.
    Stop, telling why, if they cannot serve.
    """
    asked = []  # whether the key asked its passphrase

    def passphrase():
        asked.append(True)
        if password_file:
            return read_password(Path(password_file))
        if ask and sys.stdin.isatty():
            return getpass.getpass(f"The passphrase of the key of {cert}: ")
        sys.exit(f"The key of {cert} has a passphrase: give it in a file, "
                 "--key-password-file FILE")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(cert, key, password=passphrase)
    except OSError as error:  # ssl.SSLError is one
        sys.exit("HTTPS cannot be served: "
                 + _tls_reason(cert, key, error, bool(asked)))
    return context


def _tls_reason(cert: Path, key, error: OSError, asked: bool) -> str:
    """Say why the certificate *cert* and its *key* cannot serve, the key
    having *asked* its passphrase or not (OpenSSL says "PEM lib" of most
    of these)."""
    text = str(error)
    if not isinstance(error, ssl.SSLError):
        return f"{error.strerror or text}: {error.filename or cert}"
    if "KEY_VALUES_MISMATCH" in text or "key values mismatch" in text:
        return f"the key {key or cert} is not that of the certificate {cert}"
    if asked or "BAD_DECRYPT" in text or "bad decrypt" in text:
        return f"the passphrase of the key {key or cert} is not that one"
    try:
        has_key = "PRIVATE KEY" in Path(cert).read_text(errors="replace")
    except OSError:
        has_key = False
    if key is None and not has_key:
        return (f"{cert} has no private key: give the certificate's key, "
                "--key KEY.pem")
    return (f"{cert}" + (f" or {key}" if key else "") + " is not a PEM "
            f"certificate and its key ({text})")


# The names of this machine, which a server answers to without a password.
LOCAL_NAMES = ("localhost", "127.0.0.1", "::1")


def make_server(site: Path, host: str, port: int, backend=None,
                tls=None, password=None, names=()):
    """Return the HTTP server of *site* on *host*:*port*, answering the
    page with the REPL of *backend* if given, over HTTPS with the TLS
    context *tls* if given, to the requests giving the *password* (a str)
    only, if given, else to those sent to this machine's names, *host*
    or one of *names*."""
    base = Handler if backend is None else LocalHandler
    names = {*LOCAL_NAMES, *names, host} - {"0.0.0.0", "::", ""}
    handler = type("Handler", (base,), {
        "backend": backend,
        "password": None if password is None else password.encode(),
        "names": frozenset(name.lower().strip("[]") for name in names),
    })
    server = Server(
        (host, port),
        lambda *args, **kwargs: handler(*args, directory=str(site), **kwargs),
    )
    if tls is not None:
        # The handshake is done by the thread of the request, not by the
        # loop accepting them, which a client that says nothing would stop.
        server.socket = tls.wrap_socket(
            server.socket, server_side=True, do_handshake_on_connect=False
        )
    return server


def write_index(site: Path, page: Path) -> None:
    """Write the main page of *site*, a link to the REPL's *page* that the
    browser follows, unless the site has one (the documentation's)."""
    index = site / "index.html"
    if index.exists():
        return
    target = page.relative_to(site).as_posix()
    index.write_text(
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        "<title>NeuroLang REPL</title>\n"
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'</head><body><a href="{target}">The NeuroLang REPL</a></body>'
        "</html>\n"
    )


def read_file(path: Path) -> str:
    """Return the text of the file *path*; stop, telling why, if it cannot
    be read."""
    try:
        return path.read_text()
    except OSError as error:
        sys.exit(f"{path} cannot be read: {error.strerror or error}")


def read_password(path: Path) -> str:
    """Return the password in the first line of the file *path*, as it is
    but for its line ending."""
    password = read_file(path).partition("\n")[0].rstrip("\r")
    if not password:
        sys.exit(f"The password file {path} has no password on its first line")
    return password


def read_env_file(path: Path) -> dict:
    """Return the variables of the file *path*: lines NAME=VALUE (``export``
    before, quotes around the value and ``#`` comments may be there, after
    a space if on the line of a value)."""
    variables = {}
    for number, line in enumerate(read_file(path).splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, equals, value = line.removeprefix("export ").partition("=")
        if not equals or not name.strip().isidentifier():
            sys.exit(f"{path}:{number}: not a NAME=VALUE line")
        value = value.strip()
        end = value.find(value[:1], 1) if value[:1] in ("'", '"') else -1
        if end > 0:
            value = value[1:end]
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        variables[name.strip()] = value
    return variables


def default_assistant() -> str:
    """Return the provider and the model the assistant answers with by
    default, with the keys of the environment; stop, telling why, if it
    cannot (its key is not there)."""
    from neurolang_apps.utils.llm.config import load_config

    try:
        provider = load_config().provider()
        provider.api_key()
    except Exception as error:
        sys.exit(f"The assistant cannot answer with --env-file: {error}")
    return f"{provider.name} ({provider.model})"


# The service running the server, of this user: a unit of systemd's user
# manager, or a launchd agent on macOS.
SERVICE = "neurolang-repl"
LAUNCHD_LABEL = "org.neurolang.repl"


def service_file() -> Path:
    """Return the file of the service of this user."""
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "LaunchAgents"
                / f"{LAUNCHD_LABEL}.plist")
    return (Path.home() / ".config" / "systemd" / "user"
            / f"{SERVICE}.service")


def interpreter() -> str:
    """Return the Python of this environment: that of its virtual one
    (uv's .venv), whose ``bin/python`` is a link to the Python it was made
    from, which ``sys.executable`` may name instead, without its packages."""
    venv = Path(sys.prefix) / "bin" / "python"
    if sys.prefix != sys.base_prefix and venv.exists():
        return str(venv)
    return sys.executable


def service_command(args) -> list:
    """Return the command the service runs: this one serving as *args*
    say, with no browser, nothing to build again and absolute paths."""
    python = args.python or interpreter()
    if os.sep not in python:  # a name, in the PATH
        python = shutil.which(python) or sys.exit(
            f"--python {python}: no such program in the PATH")
    python = str(Path(python).absolute())
    command = [
        python, "-m", "serve_repl", "--no-browser",
        "--host", args.host, "--port", str(args.port),
        "--site", str(args.site.resolve()),
        "--embeddings", str(args.embeddings.resolve()),
    ]
    for engine in args.engines or ():
        command += ["--engine", engine]
    for name in args.server_names or ():
        command += ["--server-name", name]
    if args.local:
        command += ["--local", "--data-dir", str(args.data_dir.resolve()),
                    "--resolution", str(args.resolution)]
        command += ["--light"] if args.light else []
    for option, value in (("--pr", args.pr), ("--run-id", args.run_id)):
        if value is not None:
            command += [option, str(value), "--repo", args.repo]
    for option, path in (
        ("--cert", args.cert), ("--key", args.key),
        ("--key-password-file", args.key_password_file),
        ("--password-file", args.password_file), ("--env-file", args.env_file),
    ):
        if path is not None:
            command += [option, str(path.resolve())]
    return command


def service_environment(python: str) -> dict:
    """Return the variables of the service's environment: the PATH of the
    one installing it, so that it finds the same programs (gh...), and the
    virtual environment of *python*, if it is one's."""
    environment = {"PATH": os.environ.get("PATH", os.defpath)}
    venv = Path(python).parent.parent
    if (venv / "pyvenv.cfg").is_file():
        environment["VIRTUAL_ENV"] = str(venv)
    return environment


def systemd_quote(word: str, dollar: str = "$$") -> str:
    """Return *word* as systemd reads it back: in double quotes, its
    backslashes and quotes escaped, its specifiers (%) and, in ExecStart,
    variables ($, as *dollar*) not expanded."""
    for character, escaped in (("\\", "\\\\"), ('"', '\\"'), ("%", "%%"),
                               ("$", dollar)):
        word = word.replace(character, escaped)
    return f'"{word}"'


def systemd_unit(command: list, directory: Path) -> str:
    """Return the systemd unit running *command* in *directory*."""
    run = " ".join(map(systemd_quote, command))
    environment = "".join(
        f"Environment={systemd_quote(f'{name}={value}', '$')}\n"
        for name, value in service_environment(command[0]).items()
    )
    directory = str(directory).replace("%", "%%")
    return (
        "[Unit]\n"
        "Description=The NeuroLang REPL (neurolang-repl-serve)\n"
        "After=network-online.target\n\n"
        "[Service]\n"
        f"ExecStart={run}\n"
        f"WorkingDirectory={directory}\n"
        f"{environment}"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def launchd_plist(command: list, directory: Path) -> bytes:
    """Return the launchd agent running *command* in *directory*."""
    log = str(Path.home() / "Library" / "Logs" / f"{SERVICE}.log")
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL, "ProgramArguments": command,
        "WorkingDirectory": str(directory), "RunAtLoad": True,
        "EnvironmentVariables": service_environment(command[0]),
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": log, "StandardErrorPath": log,
    })


def manager() -> list:
    """Return the command of the service manager, systemctl --user, or
    launchctl on macOS; stop, telling why, if this machine lacks it."""
    command = (["launchctl"] if sys.platform == "darwin"
               else ["systemctl", "--user"])
    if shutil.which(command[0]) is None:
        sys.exit(f"--daemon needs {command[0]}, which this machine lacks")
    return command


def service_manager(*args, check=True):
    """Run the service manager with *args*."""
    subprocess.run([*manager(), *args], check=check)


# What the service imports, by mode, and the extras of the workspace that
# have it.
SERVICE_IMPORTS = "serve_repl"
LOCAL_IMPORTS = "neurolang_apps.utils.tui.local_backend"
LOCAL_EXTRAS = "--extra apps --extra tui --extra llm"


def check_service_python(command: list, local: bool) -> None:
    """Stop, telling which extras to install with, if the Python of the
    service *command* cannot import what it serves with."""
    modules = SERVICE_IMPORTS + (f", {LOCAL_IMPORTS}" if local else "")
    result = subprocess.run(
        [command[0], "-c", f"import {modules}"], capture_output=True,
        text=True, env=service_environment(command[0]), cwd="/",
    )
    if result.returncode:
        error = (result.stderr.strip().splitlines() or ["?"])[-1]
        sys.exit(
            f"The Python of the service, {command[0]}, cannot serve: "
            f"{error}. Install it from the environment that has what it "
            f"needs: uv run {LOCAL_EXTRAS if local else '--extra apps'} "
            "neurolang-repl-serve --daemon install ..."
        )


def install_service(command: list) -> None:
    """Install the service running *command*, start it and enable it at
    login; if the service manager fails, remove it and stop, telling so."""
    path = service_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "darwin":
            path.write_bytes(launchd_plist(command, Path.cwd()))
            # The bootout of a reinstall ends after it returns: until then
            # the bootstrap fails (error 5).
            for attempt in range(10):
                try:
                    service_manager("bootstrap", f"gui/{os.getuid()}",
                                    str(path))
                    break
                except subprocess.CalledProcessError:
                    if attempt == 9:
                        raise
                    time.sleep(0.5)
            where = f"its log is {Path.home() / 'Library/Logs' / SERVICE}.log"
        else:
            path.write_text(systemd_unit(command, Path.cwd()))
            service_manager("daemon-reload")
            service_manager("enable", "--now", SERVICE)
            where = (f"journalctl --user -u {SERVICE} shows its log; it "
                     "stops at logout unless: loginctl enable-linger "
                     f"{getpass.getuser()}")
    except subprocess.CalledProcessError as error:
        remove_service()
        sys.exit(f"{shlex.join(error.cmd)} failed (exit status "
                 f"{error.returncode}): the service is not installed")
    print(f"Installed {path}, running {command[0]}: the REPL serves at "
          f"login; {where}", flush=True)


def remove_service() -> None:
    """Stop the service, disable it and remove its file, if installed."""
    path = service_file()
    if not path.exists():
        print("No service of the REPL is installed", flush=True)
        return
    if sys.platform == "darwin":
        service_manager("bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}",
                        check=False)
        path.unlink()
    else:
        service_manager("disable", "--now", SERVICE, check=False)
        path.unlink()
        service_manager("daemon-reload", check=False)
    print(f"Removed {path}", flush=True)


def parse_args(argv=None):
    """Return the options of the command line *argv*."""
    parser = argparse.ArgumentParser(
        prog="neurolang-repl-serve", description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="0.0.0.0 to serve the other machines of the network too (with "
        "--password-file or --server-name)",
    )
    parser.add_argument(
        "--server-name", action="append", dest="server_names", metavar="NAME",
        help="without --password-file, a name of this machine that the "
        "server answers to, besides localhost and --host (repeat for "
        "several): those of the network need it",
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
        "--cert", type=Path, metavar="PEM",
        help="serve HTTPS with this certificate (PEM, its chain after it)",
    )
    parser.add_argument(
        "--key", type=Path, metavar="PEM",
        help="the private key of --cert, if not in its file",
    )
    parser.add_argument(
        "--key-password-file", type=Path, metavar="FILE",
        help="the passphrase of the key of --cert, on the first line of FILE "
        "as it is but for its line ending (else asked in a terminal; a "
        "service needs the file)",
    )
    parser.add_argument(
        "--password-file", type=Path, metavar="FILE",
        help="ask every visitor the password on the first line of FILE, "
        "as it is but for its line ending (HTTP Basic, any user name); "
        "with --cert unless on this machine",
    )
    parser.add_argument(
        "--env-file", type=Path, metavar="FILE",
        help="with --local and --password-file, put the NAME=VALUE lines of "
        "FILE in the server's environment: the key of the AI assistant's "
        "provider (its api_key_env), which those with the password then "
        "use unless their page gives one",
    )
    parser.add_argument(
        "--python", metavar="PATH",
        help="with --daemon, the Python the service runs (default: that of "
        "the environment installing it)",
    )
    parser.add_argument(
        "--daemon", choices=("install", "reinstall", "remove"),
        help="install (building its bundle now), reinstall or remove a "
        "service of this user serving the REPL as the other options say, "
        "at login: systemd's user manager, or launchd on macOS",
    )
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
    if (args.key or args.key_password_file) and not args.cert:
        parser.error("--key and --key-password-file are those of --cert")
    if args.python and not args.daemon:
        parser.error("--python is that of the service: with --daemon")
    if (args.password_file and not args.cert
            and args.host not in ("127.0.0.1", "localhost", "::1")):
        parser.error("--password-file sends the password in clear without "
                     "--cert: give a certificate")
    if args.env_file and not (args.local and args.password_file):
        parser.error("--env-file is the key of those with the password: "
                     "with --local and --password-file")
    return args


def daemon(args) -> None:
    """Install or reinstall the service, as *args* say; the bundle of the
    page in the browser is built now, its errors told here, and the
    service installed before is removed only once all is ready."""
    if args.daemon == "install" and service_file().exists():
        sys.exit(f"A service of the REPL is installed ({service_file()}): "
                 "--daemon reinstall replaces it")
    manager()
    command = service_command(args)
    check_service_python(command, args.local)
    if not (args.local or args.pr is not None or args.run_id is not None):
        prepare_site(args.site, args.rebuild, args.embeddings,
                     clear=args.clear, engines=args.engines)
    if args.daemon == "reinstall":
        remove_service()
    install_service(command)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.daemon == "remove":  # whatever the other options
        return remove_service()
    # A certificate, a password or a key that cannot serve fails now, not
    # in the service.
    tls = tls_context(args.cert, args.key, args.key_password_file,
                      ask=not args.daemon) if args.cert else None
    password = read_password(args.password_file) if args.password_file \
        else None
    if args.env_file:
        os.environ.update(read_env_file(args.env_file))
        print(f"The AI assistant answers with {default_assistant()} for "
              "those with the password", flush=True)
    if args.daemon:
        return daemon(args)
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
    write_index(args.site, page)
    server = make_server(args.site, args.host, args.port, backend, tls,
                         password, args.server_names or ())
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    scheme = "https" if tls else "http"
    url = (f"{scheme}://{host}:{server.server_address[1]}/"
           f"{page.relative_to(args.site).as_posix()}")
    print(f"The REPL is at {url} (Ctrl+C stops it)"
          + (", for those with the password" if password else ""),
          flush=True)
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
