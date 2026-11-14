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
of the IIT atlas, which it starts with), ``neurosynth``, ``neurostore`` and
``neurostore_iit`` (NeuroStore with the same tracts; a select at its top, or
``?engine=neurostore``): the bundle has the data of
all, unless
``--engine`` names the ones to build, and is built again when it lacks one that
is asked for. What the engines download (atlases, templates, studies) is
kept in ``--downloads`` (by default ``~/.cache/neurolang/repl-downloads``),
for the next build. It opens the page in the browser. The text embeddings of
``query_similarity`` are kept in ``--embeddings`` (by default
``~/.cache/neurolang/repl-embeddings``): the bundle has them if they are
there, or if torch is installed to make them (``--extra embeddings``),
which takes a while the first time; otherwise the page goes without.

The REPL with NeuroLang in the Python of this machine::

    uv run --extra apps --extra tui --extra llm neurolang-repl-serve --local
        [--data-dir DIR]
        [--resolution MM] [--light] [--engine NAME ...] [--no-prepare]

builds no bundle: the page's worker sends what it is asked to this server,
which builds the engine from DIR (``neurolang_data`` by default, as
``neurolang-query``), with all of its analyses unless ``--light``, and runs
the REPL (``neurolang_apps.utils.tui.local_backend``). Before serving, it
builds each engine once, in a process of its own, unless ``--no-prepare``:
what it downloads and the tables it finds are kept in DIR, so that a
session starts from them, without the network. The page is the same,
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

Users, each with a password, rather than one password: ``--users FILE``, a
YAML file of their password hashes (``--hash-password`` prints one;
``--add-user NAME --groups admin``, ``--set-password``, ``--set-groups``,
``--set-key NAME VARIABLE``, ``--set-group-key GROUP VARIABLE``,
``--remove-user`` and ``--list-users`` change it from the command line)
and, for each user or group, the variables of their sessions (``env``: the
key of their assistant; a user's own over their group's). With
``--local``, each tab of a user has a session of its own, in a process
forked from a builder's that builds each engine once
(``neurolang_apps.utils.tui.sessions``), at most ``--max-sessions`` at once,
ended after ``--idle-minutes`` unused. Each user changes their password at
``/account``. The users of the group ``admin``
have the admin console at ``/admin``: the users, their groups and their
assistant's key (whose, never the key), the sessions, and the ratings of
``--feedback-db FILE``, a SQLite database of the ratings of the page, signed
by the user's name and groups.

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
most. A POST is answered only if it is the page's: in JSON
(``Content-Type: application/json``, which a form of another site cannot
send), and from this site if the browser gives its ``Origin``.
"""

import argparse
import base64
import contextlib
import getpass
import hashlib
import hmac
import http.server
import importlib.util
import json
import os
import plistlib
import queue
import re
import secrets
import shlex
import shutil
import signal
import socketserver
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from pathlib import Path

import yaml

DOC = Path(__file__).resolve().parents[2] / "doc"
BUNDLE_SCRIPT = DOC / "make_repl_bundle.py"
PAGE = DOC / "_static" / "repl"
LOGO = DOC / "_static" / "logo"
# The defaults of the options, the one place they are written.
DEFAULTS_FILE = Path(__file__).with_name("defaults.yaml")
DEFAULTS = yaml.safe_load(DEFAULTS_FILE.read_text(encoding="utf-8"))
DEFAULT_SITE = Path(DEFAULTS["site"]).expanduser()
# The admin console of the users (--users), served at /admin.
ADMIN_PAGE = Path(__file__).with_name("admin.html")
# The page of a user's account (--users), served at /account.
ACCOUNT_PAGE = Path(__file__).with_name("account.html")
# The page where a user logs in (--users), served at /login, and the cookie
# of a login, kept so many days.
LOGIN_PAGE = Path(__file__).with_name("login.html")
LOGIN_PATHS = ("/login", "/login/guest", "/login/options", "/login/orcid",
               "/login/orcid/callback", "/login/request", "/logout")
# The cookies of a sign-in with ORCID on its way (its state and where to go
# then), and of an iD of no user who may ask for an account; minutes.
ORCID_COOKIE = "neurolang_orcid"
REQUEST_COOKIE = "neurolang_request"
ORCID_MINUTES = 15
# Whether the server serves, for a monitor, without a login.
HEALTH_PATH = "/health"
# The dashboard (--dashboard), Datasette, for the admins:
# neurolang_apps.utils.tui.dashboard.PREFIX.
DASHBOARD_PREFIX = "/admin/logs/"
LOGIN_COOKIE = "neurolang_login"
LOGIN_DAYS = 7
# A guest (--guests): a name of its own, which no user has (theirs have no
# ':'), in the group guest; they bring their own key.
GUEST_PREFIX = "guest:"
GUEST_GROUP = "guest"
BRING_KEY_ENV = "NEUROLANG_REPL_BRING_KEY"
# The variables of a session's process telling its user's assistant (JSON:
# provider, model, base_url, key) and that they may not give their own key.
ASSISTANT_ENV = "NEUROLANG_REPL_LLM"
OWN_KEY_ENV = "NEUROLANG_REPL_OWN_KEY"
# The shortest password a user gives themselves.
SHORTEST_PASSWORD = 8
# Where the text embeddings are kept between builds: making them takes
# torch, the abstracts of the studies from PubMed and a while.
DEFAULT_EMBEDDINGS = Path(DEFAULTS["embeddings"]).expanduser()
# What the engines of the bundle download (atlases, templates, studies), a
# directory each, kept for the next build: a source down (OSF, GitHub) then
# stops no rebuild.
DEFAULT_DOWNLOADS = Path(DEFAULTS["downloads"]).expanduser()
# The repository whose CI uploads the site of each pull request, and the
# artifacts holding it (.github/workflows/docs.yml).
REPOSITORY = "NeuroLang/NeuroLang"
ARTIFACTS = ("repl-pr-", "documentation-html")
# The run whose site a downloaded site is, in that site.
RUN_FILE = ".neurolang-run"


def _sibling(name: str):
    """The module *name* next to this one, loaded by its path, as this one
    may be (the tests load it so, outside of any package)."""
    spec = importlib.util.spec_from_file_location(
        f"serve_repl.{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Signing in with ORCID and the requests of an account (--orcid).
orcid = _sibling("orcid")


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


def build_bundle(
    repl: Path, embeddings, engines=None, downloads=DEFAULT_DOWNLOADS,
) -> None:
    """Build the page's bundle in *repl*, with the text embeddings of the
    data directory *embeddings* if given, and the data of the *engines*
    (by default all those of the page), what they download kept in
    *downloads* (none if None)."""
    command = [sys.executable, str(BUNDLE_SCRIPT), str(repl)]
    if embeddings:
        command.append(str(embeddings))
    if downloads is not None:
        command += ["--downloads", str(downloads)]
    for engine in engines or ():
        command += ["--engine", engine]
    subprocess.check_call(command)


def prepare_site(
    site: Path, rebuild: bool = False, embeddings=None, clear: bool = False,
    engines=None, downloads=DEFAULT_DOWNLOADS,
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
        if downloads == DEFAULT_DOWNLOADS:
            build_bundle(repl, embeddings, engines)
        else:
            build_bundle(repl, embeddings, engines, downloads=downloads)
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


def _prepare_engine(data_dir, name, resolution, light) -> None:
    """Build the engine *name* on *data_dir*, as a session does, telling its
    steps: what it downloads, and the tables it finds, are kept there."""
    from neurolang_apps.utils.tui.local_backend import LocalBackend

    backend = LocalBackend(data_dir, resolution=resolution, light=light)
    backend._build_engine(
        name, lambda step: print(f"  {name}: {step}", flush=True)
    )


def _prepare_in_child(data_dir, resolution, light):
    """Return the function building an engine in a process of its own,
    which tells whether it could: the memory of the engine is not kept by
    the server's process."""
    import multiprocessing

    context = multiprocessing.get_context("spawn")

    def run(name) -> bool:
        child = context.Process(
            target=_prepare_engine, args=(data_dir, name, resolution, light),
        )
        child.start()
        child.join()
        return child.exitcode == 0
    return run


def prepare_engines(
    data_dir: Path, engines, resolution: float = 2.0, light: bool = False,
    run=None,
) -> list:
    """
    Build each of the *engines* once on *data_dir*, before the server
    serves: what they download (atlases, templates, studies) and the
    tables they take long to find are kept there, so that a session starts
    from them, without the network. Tell how each went; return the names
    of those that could not be built, whose sessions download what they
    lack, if they can. *run*(name) builds one and tells whether it could
    (by default in a process of its own).
    """
    run = run or _prepare_in_child(data_dir, resolution, light)
    failed = []
    for name in engines:
        print(f"Preparing the engine {name} in {data_dir}…", flush=True)
        start = time.monotonic()
        if run(name):
            print(f"The engine {name} is ready "
                  f"({time.monotonic() - start:.0f} s)", flush=True)
        else:
            failed.append(name)
            print(f"The engine {name} could not be prepared (see above): "
                  "its sessions download what it lacks, if they can",
                  flush=True)
    return failed


def prepare_local_site(site: Path, engines=None, users=False) -> Path:
    """
    Put the page in *site* with a ``bundle.json`` telling it to have this
    machine's Python answer it (``"backend": "local"``), offering the
    *engines* (by default all of the page's), and that it has *users* (whose
    account page it links), and return the path of the page.
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
        **({"users": True} if users else {}),
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
    #: Or the users, each with the hash of their password (see
    #: :func:`hash_password`); None: none.
    users = None
    #: The user of the request, once authorised ("" without users).
    user = ""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # the downloads of the page are many and of no interest

    def parse_request(self):
        """Read the request, and answer 401 to one without the password,
        or, if there is none, 403 to one sent to a host not of
        ``names``, and 403 to a POST not of this site's page: then it goes
        no further, whatever its method."""
        if not super().parse_request():
            return False
        path = urllib.parse.urlsplit(self.path).path
        if path == HEALTH_PATH:
            return True  # open to all, monitors: it tells nothing of users
        if self.users is not None and path in LOGIN_PATHS:
            return True  # open to all; a POST is checked there
        if self.password is None and self.users is None:
            try:
                host = urllib.parse.urlsplit(
                    "//" + (self.headers.get("Host") or "")).hostname
            except ValueError:  # an IPv6 address without its ]
                host = None
            if host not in self.names:
                self.send_error(403, "Unknown host", (
                    f"This server does not answer to {host}: if it is its "
                    f"name, serve it with --server-name {host}, or with a "
                    "password"
                ))
                return False
        elif not self._authorised():
            if self.users is not None:
                # The login page for a page the browser opens; for the rest
                # a 401 without a challenge, which would have the browser
                # ask.
                if self.command == "GET" and \
                        "text/html" in self.headers.get("Accept", ""):
                    target = urllib.parse.quote(self.path, safe="/?=&")
                    self._reply(303, b"", "text/plain", [
                        ("Location", f"/login?next={target}")])
                else:
                    self._reply_json({"error": "log in first: /login"}, 401)
                return False
            self.send_response(401)
            self.send_header(
                "WWW-Authenticate",
                'Basic realm="NeuroLang REPL", charset="UTF-8"'
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        if self.command == "POST" and not self._same_site():
            self.send_error(403, "A request of this site's page only")
            return False
        return True

    def _same_site(self) -> bool:
        """Whether a POST comes from this site's page: in JSON, which
        another site's form cannot send without asking first, and from
        this site, if the browser says where from. The browser would
        otherwise add the login to a form another site sends here."""
        if self.headers.get_content_type() != "application/json":
            return False
        origin = self.headers.get("Origin")
        return origin is None or \
            urllib.parse.urlsplit(origin).netloc == self.headers.get("Host")

    #: The sessions of the users (--users --local), for the console.
    sessions = None
    #: The feedback database, for the console.
    feedback_db = None
    #: The dashboard (--dashboard), for the console.
    dashboard = None
    #: When the server started (time.monotonic), for /health.
    started = 0.0

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == HEALTH_PATH:
            return self._health()
        if self.users is not None and path in LOGIN_PATHS:
            return self._guarded(self._login, path)
        if path == "/admin" or path.startswith("/admin/"):
            return self._admin(path)
        if path == "/account" or path.startswith("/account/"):
            return self._account(path)
        return super().do_GET()

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if self.users is not None and path in LOGIN_PATHS:
            return self._guarded(self._login, path)
        if path.startswith("/admin/"):
            return self._admin(path)
        if path.startswith("/account/"):
            return self._account(path)
        self.send_error(404)

    def _account(self, path: str):
        """The page of the user's account, ``/account``: who they are, and
        a new password they give themselves (the current one first)."""
        self._guarded(self._account_answer, path)

    def _guarded(self, answer, path: str):
        """Answer the request to *path* with *answer*; a ValueError is told
        (the request's, or the console's own messages), another error by
        its kind only, its traceback in the log."""
        try:
            return answer(path)
        except ValueError as error:
            return self._reply_json(
                {"error": str(error)},
                413 if isinstance(error, TooLarge) else 400)
        except Exception as error:
            traceback.print_exc()
            return self._reply_json({"error": type(error).__name__}, 500)

    def _account_answer(self, path: str):
        if self.users is None:
            return self._reply_json({"error": "no users"}, 404)
        if path in ("/account", "/account/"):
            return self._reply(200, ACCOUNT_PAGE.read_bytes(),
                               "text/html; charset=utf-8")
        if self.user.startswith(GUEST_PREFIX):  # no account: only who
            if path == "/account/api/me" and self.command == "GET":
                return self._reply_json({
                    "name": "guest", "guest": True, "groups": [GUEST_GROUP],
                    "admin": False, "providers": _providers(),
                    "assistant": {"assistant": "none", "own_key": True},
                })
            return self._reply_json(
                {"error": "A guest has no account: bring your key in the "
                 "page"}, 403)
        if path == "/account/api/me" and self.command == "GET":
            entry = self.users.current().get(self.user, {})
            return self._reply_json({
                "name": self.user, "groups": self.users.groups(self.user),
                "orcid_name": entry.get("orcid_name", ""),
                "orcid": entry.get("orcid", ""),
                "orcid_only": "password" not in entry,
                "orcid_signs_in": self.orcid is not None,
                "admin": self.users.is_admin(self.user),
                "assistant": self.users.status(self.user),
                "providers": _providers(),
            })
        if path == "/account/api/llm" and self.command == "POST":
            if not self.users.own_key(self.user):
                return self._reply_json({"error": "Your assistant's key is "
                                         "the one your admin set"}, 403)
            body = self._body()
            self.users.change(self.user, llm=False if body.get("remove")
                              else {k: body.get(k) for k in (
                                  "provider", "model", "base_url", "key")})
            self._refresh_sessions([self.user])
            return self._reply_json({"ok": True})
        if path == "/account/api/unlink-orcid" and self.command == "POST":
            self._body()
            orcid_id = self.users.unlink_orcid(self.user)
            self._refresh_sessions([self.user])
            return self._reply_json({"ok": True, "orcid": orcid_id})
        if path == "/account/api/merge-into" and self.command == "POST":
            # an ORCID user into an account of theirs, proved by its
            # password: logged in as it then
            body = self._body()
            into = str(body.get("name", ""))
            password = str(body.get("password", "")).encode()
            if not orcid.is_orcid(self.user):
                return self._reply_json({"error": "Your account is not one "
                                         "of ORCID"}, 403)
            if not self.users.check(into, password):
                return self._reply_json(
                    {"error": "That name and password are not a user's"}, 403)
            self.users.merge_orcid(self.user, into)
            if self.sessions:
                self.sessions.end(self.user)
            self._refresh_sessions([into])
            return self._reply_json({"ok": True, "name": into}, 200, [
                ("Set-Cookie", self._cookie(into, LOGIN_DAYS * 86400))])
        if path == "/account/api/password" and self.command == "POST":
            body = self._body()
            current = str(body.get("current", "")).encode()
            new = str(body.get("new", ""))
            if "password" not in self.users.current().get(self.user, {}):
                return self._reply_json(
                    {"error": "You sign in with ORCID: you have no "
                     "password to change"}, 403)
            if not self.users.check(self.user, current):
                return self._reply_json(
                    {"error": "The current password is not that one"}, 403)
            if len(new) < SHORTEST_PASSWORD:
                return self._reply_json({"error": "The new password has at "
                                         f"least {SHORTEST_PASSWORD} "
                                         "characters"}, 400)
            self.users.change(self.user, password=new)
            # The login made with the old password is out: a new one.
            return self._reply_json({"ok": True}, 200, [
                ("Set-Cookie", self._cookie(self.user, LOGIN_DAYS * 86400))])
        return self._reply_json({"error": "not found"}, 404)

    def _reply(self, status: int, body: bytes, mime: str, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _reply_json(self, value, status: int = 200, headers=()):
        self._reply(status, json.dumps(value).encode(), "application/json",
                    headers)

    def _health(self):
        """Whether the server serves, for a monitor: 200, or 503 if its
        sessions can no longer start (their builder ended); and for how
        long it has."""
        dead = getattr(self.sessions, "dead", None)
        answer = {"ok": not dead,
                  "uptime_seconds": round(time.monotonic() - self.started)}
        if dead:
            answer["error"] = dead
        self._reply_json(answer, 503 if dead else 200)

    def _admin(self, path: str):
        """The admin console, ``/admin``, and what it asks, for the users
        of the group admin only."""
        if self.users is None or not self.users.is_admin(self.user):
            return self._reply_json({"error": "for the admins only"}, 403)
        self._guarded(self._admin_answer, path)

    def _admin_answer(self, path: str):
        if path == "/admin/logs" or path.startswith(DASHBOARD_PREFIX):
            return self._dashboard(path)
        if self.command == "POST":
            return self._reply_json(self._admin_change(path, self._body()))
        if path in ("/admin", "/admin/"):
            return self._reply(200, ADMIN_PAGE.read_bytes(),
                               "text/html; charset=utf-8")
        if path == "/admin/api/state":
            return self._reply_json(self._admin_state())
        if path == "/admin/api/feedback":
            return self._reply_json(self._admin_feedback())
        activity = getattr(self.sessions, "activity", None)
        if path == "/admin/api/usage":
            return self._reply_json(activity.totals() if activity else [])
        if path == "/admin/activity.jsonl" and activity:
            # The log of a user, ?user=NAME, an event a line.
            query = urllib.parse.urlsplit(self.path).query
            user = urllib.parse.parse_qs(query).get("user", [""])[0]
            entries = activity.entries(user)
            if not entries:
                raise ValueError(f"No activity of {user!r}: ?user=NAME")
            text = "".join(json.dumps(e) + "\n" for e in entries)
            name = urllib.parse.quote(user, safe="")
            return self._reply(200, text.encode(), "application/x-ndjson", [
                ("Content-Disposition",
                 f'attachment; filename="activity-{name}.jsonl"')])
        if path == "/admin/feedback.yaml" and self.feedback_db:
            # The examples of the assistant of an engine, ?engine=NAME.
            from neurolang_apps.utils.llm.feedback_server import (
                FeedbackStore, examples_yaml,
            )

            store = FeedbackStore(self.feedback_db)
            query = urllib.parse.urlsplit(self.path).query
            engine = urllib.parse.parse_qs(query).get("engine", [""])[0]
            if engine not in store.engines():
                raise ValueError("No ratings of the engine "
                                 f"{engine!r}: one of ?engine=NAME")
            text = examples_yaml(store, engine)
            return self._reply(200, text.encode(), "application/yaml", [
                ("Content-Disposition",
                 f'attachment; filename="feedback-{engine}.yaml"')])
        return self._reply_json({"error": "not found"}, 404)

    def _dashboard(self, path: str):
        """Pass a GET of the dashboard to Datasette (--dashboard)."""
        if self.dashboard is None:
            return self._reply_json({"error": "no dashboard: start the "
                                     "server with --dashboard"}, 404)
        if self.command != "GET":
            return self._reply_json({"error": "the dashboard is read only"},
                                    405)
        if path == "/admin/logs":
            return self._reply(301, b"", "text/plain",
                               [("Location", DASHBOARD_PREFIX)])
        status, headers, body = self.dashboard.get(self.path)
        mime = dict(headers).pop("Content-Type", "text/plain")
        self._reply(status, body, mime,
                    [h for h in headers if h[0].title() != "Content-Type"])

    def _admin_state(self) -> dict:
        users = self.users.current()
        groups = self.users.group_envs()
        named = sorted(set(groups) | {g for e in users.values()
                                      for g in e["groups"]})
        return {
            "me": self.user,
            "users": [{"name": name, "groups": entry["groups"],
                       "own_key_set": entry.get("own_key"),
                       "orcid_name": entry.get("orcid_name", ""),
                       "orcid": entry.get("orcid", ""),
                       "orcid_only": "password" not in entry,
                       **self.users.status(name)}
                      for name, entry in sorted(users.items())],
            "orcid": self.orcid is not None,
            "requests": self.account_requests.list()
            if self.account_requests else [],
            "groups": [{"name": group,
                        "members": sorted(n for n, e in users.items()
                                          if group in e["groups"]),
                        **self.users.group_status(group)}
                       for group in named],
            "providers": _providers(),
            "sessions": self.sessions.info() if self.sessions else None,
            # the memory the sessions share, that of the builder's process
            "builder": getattr(self.sessions, "builder_usage", None),
            # the most queries running at once, and their longest run
            "limits": self.sessions and {
                "running": getattr(self.sessions, "running", None),
                "query_seconds": getattr(self.sessions, "query_seconds",
                                         None),
                "sessions": getattr(self.sessions, "most", None)},
            "activity": getattr(self.sessions, "activity", None) is not None,
            "dashboard": self.dashboard is not None,
            "feedback": bool(self.feedback_db),
            "feedback_engines": self._feedback_engines(),
        }

    def _feedback_engines(self) -> list:
        if not self.feedback_db:
            return []
        from neurolang_apps.utils.llm.feedback_server import FeedbackStore

        return FeedbackStore(self.feedback_db).engines()

    def _admin_feedback(self) -> list:
        if not self.feedback_db:
            return []
        from neurolang_apps.utils.llm.feedback_server import FeedbackStore

        keep = ("created", "user", "groups", "engine", "question", "query",
                "valid", "error", "status")
        return [{**{key: answer.get(key) for key in keep},
                 "ratings": [{key: rating.get(key) for key in
                              ("created", "user", "verdict", "comment")}
                             for rating in answer["feedback"]]}
                for answer in FeedbackStore(self.feedback_db).answers(
                    limit=200)]

    def _refresh_sessions(self, names) -> None:
        """Give the sessions of the users *names* their user as they are
        now."""
        if self.sessions is not None and hasattr(self.sessions, "refresh"):
            self.sessions.refresh(names)

    def _admin_change(self, path: str, body: dict) -> dict:
        answer = self._admin_changed(path, body)
        name = str(body.get("name", ""))
        if path in ("/admin/api/user", "/admin/api/separate-orcid"):
            self._refresh_sessions([name])
        elif path == "/admin/api/merge-orcid":
            self._refresh_sessions([str(body.get("into", ""))])
        elif path == "/admin/api/group":  # its members
            self._refresh_sessions([n for n, e in self.users.current().items()
                                    if name in e["groups"]])
        return answer

    def _admin_changed(self, path: str, body: dict) -> dict:
        name = str(body.get("name", ""))
        if path == "/admin/api/user":
            self.users.change(
                name, password=body.get("password") or None,
                groups=body.get("groups"), env=body.get("env"),
                create=bool(body.get("create")), llm=body.get("llm"),
                own_key=body.get("own_key"),
            )
            return {"ok": True}
        if path == "/admin/api/group":
            self.users.change_group(name, body.get("env"), body.get("llm"),
                                    body.get("own_key"))
            return {"ok": True}
        if path == "/admin/api/merge-orcid":
            into = str(body.get("into", ""))
            self.users.merge_orcid(name, into)
            if self.sessions:  # the ORCID user is no more
                self.sessions.end(name)
            return {"ok": True}
        if path == "/admin/api/separate-orcid":
            return {"ok": True, "orcid": self.users.separate_orcid(name)}
        if path in ("/admin/api/approve-request",
                    "/admin/api/decline-request"):
            if self.account_requests is None:
                raise ValueError("No requests: the server has no --orcid")
            asked = self.account_requests.current().get(name)
            if asked is None:
                raise ValueError(f"No request of {name}")
            if path.endswith("approve-request"):
                self.users.change(name, groups=body.get("groups") or [],
                                  create=True,
                                  orcid_name=asked.get("name", ""))
            self.account_requests.remove(name)
            return {"ok": True}
        if path == "/admin/api/remove-user":
            self.users.remove(name)
            if self.sessions:
                self.sessions.end(name, reason="the user was removed")
            return {"ok": True}
        if path == "/admin/api/end-session" and self.sessions:
            tab = body.get("tab")
            return {"ended": self.sessions.end(name, tab and str(tab))}
        raise ValueError("no such change")

    #: The key that signs the login cookies (--users), and whether they
    #: go over HTTPS only.
    login_secret = None
    secure = False
    #: When each user (or guest) logged out last, by name: a login cookie
    #: made before is out (make_server gives each server its own).
    logged_out = {}
    #: Whether a visitor may come in as a guest (--guests).
    guests = False

    #: The client of the server at ORCID (--orcid), and the requests of an
    #: account of those who sign in with it and are no user.
    orcid = None
    account_requests = None

    def _login(self, path: str):
        """``/login``: its page, and a user's name and password, which give
        a login cookie; ``/logout``, which takes it back; signing in with
        ORCID (``/login/orcid``) and asking for an account."""
        if path == "/login/guest":
            return self._guest()
        if path == "/login/options" and self.command == "GET":
            return self._reply_json({"guests": self.guests,
                                     "orcid": self.orcid is not None})
        if path.startswith(("/login/orcid", "/login/request")):
            if self.orcid is None:
                return self._reply_json({"error": "no ORCID here"}, 404)
            if path == "/login/orcid":
                return self._orcid_start()
            if path == "/login/orcid/callback":
                return self._orcid_callback()
            return self._account_request()
        if path == "/logout":
            if self._authorised():
                now = time.time()
                for name, when in list(self.logged_out.items()):
                    if when < now - LOGIN_DAYS * 86400:  # expired anyway
                        del self.logged_out[name]
                self.logged_out[self.user] = now
            return self._reply(303, b"", "text/plain", [
                ("Location", "/login"),
                ("Set-Cookie", self._cookie("", 0))])
        if self.command == "GET":
            return self._reply(200, LOGIN_PAGE.read_bytes(),
                               "text/html; charset=utf-8")
        if not self._same_site():
            return self._reply_json(
                {"error": "A request of this site's page only"}, 403)
        body = self._request_body()
        if body is None:
            return
        name = str(body.get("name", ""))
        password = str(body.get("password", "")).encode()
        if not self.users.check(name, password):
            return self._reply_json(
                {"error": "That name and password are not a user's"}, 403)
        return self._reply_json({"ok": True}, 200, [
            ("Set-Cookie", self._cookie(name, LOGIN_DAYS * 86400))])

    def _guest(self):
        """``/login/guest``: whether guests come in, and a guest's login,
        a name of its own, which brings its own key."""
        if self.command == "GET":
            return self._reply_json({"guests": self.guests})
        if not self.guests:
            return self._reply_json({"error": "no guests here"}, 404)
        if not self._same_site():
            return self._reply_json(
                {"error": "A request of this site's page only"}, 403)
        if self._request_body() is None:
            return
        name = GUEST_PREFIX + secrets.token_hex(6)
        return self._reply_json({"ok": True}, 200, [
            ("Set-Cookie", self._cookie(name, LOGIN_DAYS * 86400))])

    def _signed(self, cookie: str, value: dict, seconds: int) -> str:
        """The cookie *cookie* of *value*, signed, for *seconds* (0: none)."""
        text = ""
        if seconds:
            payload = base64.urlsafe_b64encode(json.dumps(
                {**value, "e": int(time.time()) + seconds}).encode()).decode()
            text = payload + "." + hmac.new(
                self.login_secret, payload.encode(), "sha256").hexdigest()
        return (f"{cookie}={text}; Path=/login; Max-Age={seconds}; "
                "HttpOnly; SameSite=Lax" + ("; Secure" if self.secure else ""))

    def _read_signed(self, cookie: str):
        """The value of the signed cookie *cookie* if it is good and not
        expired, else None."""
        for header in self.headers.get_all("Cookie") or ():
            for part in header.split(";"):
                key, _, text = part.strip().partition("=")
                if key != cookie:
                    continue
                payload, _, signature = text.partition(".")
                good = hmac.new(self.login_secret, payload.encode(),
                                "sha256").hexdigest()
                if not hmac.compare_digest(signature.encode(), good.encode()):
                    continue
                try:
                    value = json.loads(base64.urlsafe_b64decode(payload))
                except ValueError:
                    continue
                if isinstance(value, dict) and value.get("e", 0) > time.time():
                    return value
        return None

    def _redirect_uri(self) -> str:
        """Where ORCID sends the visitor back: the client's, else this
        server's ``/login/orcid/callback`` at the address of the request."""
        if self.orcid.redirect_uri:
            return self.orcid.redirect_uri
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.headers.get('Host')}/login/orcid/callback"

    def _orcid_start(self):
        """``/login/orcid?next=PAGE``: off to ORCID's page, with a state
        that only this browser has, to come back to PAGE."""
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        target = query.get("next", ["/"])[0]
        if not target.startswith("/") or target.startswith("//"):
            target = "/"  # a page of this site only
        started = {"s": secrets.token_urlsafe(24), "n": target}
        if query.get("link"):  # a user's own iD, for them to sign in with
            if not self._authorised() or self.user.startswith(GUEST_PREFIX):
                return self._reply_json({"error": "log in first: /login"},
                                        401)
            started.update(l=self.user, n="/account")
        return self._reply(303, b"", "text/plain", [
            ("Location", self.orcid.authorize_url(self._redirect_uri(),
                                                  started["s"])),
            ("Set-Cookie", self._signed(ORCID_COOKIE, started,
                                        ORCID_MINUTES * 60))])

    def _orcid_callback(self):
        """``/login/orcid/callback?code=&state=``: ORCID tells who signed
        in. A user logs in; an iD of no user may ask for an account."""
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        started = self._read_signed(ORCID_COOKIE)
        forget = ("Set-Cookie", self._signed(ORCID_COOKIE, {}, 0))
        if query.get("error"):  # the visitor said no at ORCID
            return self._reply(303, b"", "text/plain", [forget, (
                "Location", "/login?orcid=cancelled")])
        if started is None or not hmac.compare_digest(
                str(query.get("state", [""])[0]), str(started.get("s"))):
            return self._reply(303, b"", "text/plain", [forget, (
                "Location", "/login?orcid=expired")])
        try:
            orcid_id, name = self.orcid.exchange(
                str(query.get("code", [""])[0]), self._redirect_uri())
        except orcid.OrcidError:
            return self._reply(303, b"", "text/plain", [forget, (
                "Location", "/login?orcid=failed")])
        if started.get("l"):
            return self._orcid_linked(started["l"], orcid_id, name, forget)
        user = self.users.by_orcid(orcid_id)
        if user is not None:
            self.users.seen_orcid(user, name)
            return self._reply(303, b"", "text/plain", [forget, (
                "Set-Cookie", self._cookie(user, LOGIN_DAYS * 86400)), (
                "Location", started.get("n") or "/")])
        return self._reply(303, b"", "text/plain", [forget, (
            "Set-Cookie", self._signed(REQUEST_COOKIE, {
                "o": orcid_id, "m": name}, ORCID_MINUTES * 60)), (
            "Location", "/login?orcid=request")])

    def _orcid_linked(self, user: str, orcid_id: str, name: str, forget):
        """The iD *orcid_id* proved by *user*, still logged in in this
        browser, who asked to sign in with it too: linked to them, or
        the ORCID user of it merged into them. Back to their account,
        ``?orcid=`` telling how it went."""
        if not self._authorised() or self.user != user:
            outcome = "expired"
        else:
            try:
                outcome = self.users.link_orcid(user, orcid_id, name)
                if outcome == "merged" and self.sessions:
                    self.sessions.end(orcid_id)  # that user is no more
                self._refresh_sessions([user])
            except ValueError as error:
                outcome = "taken" if "signs in" in str(error) else "refused"
        return self._reply(303, b"", "text/plain", [forget, (
            "Location", f"/account?orcid={outcome}")])

    def _account_request(self):
        """``/login/request``: who signed in with ORCID and is no user, and
        whether they asked for an account (GET); their request, with a
        message for the admins (POST)."""
        who = self._read_signed(REQUEST_COOKIE)
        if who is None:
            return self._reply_json({"error": "Sign in with ORCID first"},
                                    403)
        orcid_id = str(who.get("o"))
        if self.command == "GET":
            return self._reply_json({
                "orcid": orcid_id, "name": who.get("m", ""),
                "requested": orcid_id in self.account_requests.current()})
        if not self._same_site():
            return self._reply_json(
                {"error": "A request of this site's page only"}, 403)
        body = self._request_body()
        if body is None:
            return
        self.account_requests.add(orcid_id, who.get("m", ""),
                                  str(body.get("message", "")))
        return self._reply_json({"ok": True})

    def _request_body(self):
        """Return the JSON object of the request's body, or None once told
        why not (see :meth:`_body`)."""
        try:
            return self._body()
        except ValueError as error:
            self._reply_json({"error": str(error)},
                             413 if isinstance(error, TooLarge) else 400)
            return None

    def _cookie(self, name: str, seconds: int) -> str:
        """The login cookie of *name* for *seconds* (0: none)."""
        value = sign_login(self.login_secret, name, self.users, seconds) \
            if name else ""
        return (f"{LOGIN_COOKIE}={value}; Path=/; Max-Age={seconds}; "
                "HttpOnly; SameSite=Lax" + ("; Secure" if self.secure else ""))

    def _authorised(self) -> bool:
        if self.users is not None:
            # Each login cookie, read by hand: another cookie of the host
            # that the standard library cannot parse does not hide it.
            for header in self.headers.get_all("Cookie") or ():
                for cookie in header.split(";"):
                    key, _, value = cookie.strip().partition("=")
                    name = None if key != LOGIN_COOKIE else check_login(
                        self.login_secret, value, self.users, self.guests,
                        self.logged_out)
                    if name is not None:
                        self.user = name
                        return True
        scheme, _, encoded = (
            self.headers.get("Authorization") or ""
        ).partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            credentials = base64.b64decode(encoded.strip(), validate=True)
        except ValueError:
            return False
        name, _, given = credentials.partition(b":")
        if self.users is None:
            return hmac.compare_digest(given, self.password)
        try:
            self.user = name.decode()
        except UnicodeDecodeError:
            return False
        return self.users.check(self.user, given)

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
    #: The sessions the guests have at most at once.
    max_guest_sessions = 2
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

    def _backend(self):
        """Return the backend of the request: with sessions (``--users``),
        that of the user and of the tab (``?session=ID``) asking."""
        if not hasattr(self.backend, "session"):
            return self.backend
        query = urllib.parse.urlsplit(self.path).query
        tab = urllib.parse.parse_qs(query).get("session", ["default"])[0][:64]
        if self.user.startswith(GUEST_PREFIX):
            guests = {(i["user"], i["tab"]) for i in self.backend.info()
                      if i["user"].startswith(GUEST_PREFIX)}
            if (self.user, tab) not in guests and \
                    len(guests) >= self.max_guest_sessions:
                raise RuntimeError("The sessions of the guests are all at "
                                   "work: come back later")
        return self.backend.session(self.user, tab)

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path != "/api/events":
            return super().do_GET()
        try:
            backend = self._backend()
        except RuntimeError as error:  # too many sessions
            return self._json({"error": str(error)}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        listener = backend.listen()
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
            backend.forget(listener)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith(("/admin/", "/account/")) or path in LOGIN_PATHS:
            return super().do_POST()
        if not path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        action = path[len("/api/"):]
        try:
            body = self._body()
        except ValueError as error:
            return self._json({"error": str(error)},
                              413 if isinstance(error, TooLarge) else 400)
        try:
            self._answer(action, body)
        except Exception as error:  # told to the page, and in the log
            if type(error).__name__ == "SessionEnded":
                # The page brings the session back (sessions.py).
                return self._json({"error": str(error), "ended": True}, 409)
            traceback.print_exc()
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def _answer(self, action: str, body: dict):
        backend = self._backend()
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
                ask: bool = True, key_password=None) -> ssl.SSLContext:
    """
    Return the TLS context of a server with the certificate *cert*, a PEM
    file (its chain after it), and its private key, in *key* or else in
    *cert*. A key with a passphrase is opened with *key_password* (that
    of the secrets), or the first line of *password_file*, else the
    passphrase asked in a terminal, if *ask*.
    Stop, telling why, if they cannot serve.
    """
    asked = []  # whether the key asked its passphrase

    def passphrase():
        asked.append(True)
        if key_password:
            return key_password
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
                tls=None, password=None, names=(), users=None,
                feedback_db=None, guests: bool = False,
                max_guest_sessions: int = 2, secure_cookie: bool = False,
                orcid_client=None, dashboard=None):
    """Return the HTTP server of *site* on *host*:*port*, answering the
    page with the REPL of *backend* if given, over HTTPS with the TLS
    context *tls* if given, to the requests giving the *password* (a str)
    only, if given, else to those sent to this machine's names, *host*
    or one of *names*. The login cookie goes over HTTPS only with *tls*,
    or *secure_cookie* (behind a proxy of HTTPS). With *orcid_client* (an
    :class:`~serve_repl.orcid.OrcidClient`), the users sign in with ORCID
    too, and an iD of no user may ask for an account, in a file next to the
    users file (:func:`~serve_repl.orcid.requests_path`)."""
    base = Handler if backend is None else LocalHandler
    names = {*LOCAL_NAMES, *names, host} - {"0.0.0.0", "::", ""}
    handler = type("Handler", (base,), {
        "backend": backend,
        "password": None if password is None else password.encode(),
        "names": frozenset(name.lower().strip("[]") for name in names),
        "users": users,
        "login_secret": os.urandom(32) if users is not None else None,
        "secure": tls is not None or secure_cookie, "logged_out": {},
        "guests": guests, "max_guest_sessions": max_guest_sessions,
        "sessions": backend if hasattr(backend, "info") else None,
        "feedback_db": feedback_db,
        "orcid": orcid_client if users is not None else None,
        "account_requests": orcid.AccountRequests(
            orcid.requests_path(users.path))
        if users is not None and orcid_client is not None else None,
        "dashboard": dashboard, "started": time.monotonic(),
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


# The cost of the scrypt hash of a password (RFC 7914's interactive
# parameters).
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    """Return the hash of *password* that a users file keeps."""
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    encode = base64.b64encode
    return (f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}$"
            f"{encode(salt).decode()}${encode(digest).decode()}")


def _verify(stored: str, password: bytes) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        digest = base64.b64decode(digest)
        given = hashlib.scrypt(
            password, salt=base64.b64decode(salt), n=int(n), r=int(r),
            p=int(p), dklen=len(digest),
        )
    except ValueError:
        return False
    return kind == "scrypt" and hmac.compare_digest(given, digest)


# The credentials checked already, by a hash of their own: Basic sends them
# with each request, which a hash of scrypt each would slow down.
_checked = set()
# A hash an unknown user's password is checked against, to take as long as
# a known user's: the time does not tell who is a user.
_unknown = []


def check_password(users: dict, user: str, password: bytes) -> bool:
    """Return whether *password* is that of *user* among *users* (a user
    who signs in with ORCID only has none)."""
    entry = users.get(user)
    if entry is None or "password" not in entry:
        if not _unknown:
            _unknown.append(hash_password(os.urandom(16).hex()))
        _verify(_unknown[0], password)
        return False
    seen = hashlib.sha256(
        b"\0".join([user.encode(), password, entry["password"].encode()])
    ).digest()
    if seen in _checked:
        return True
    if not _verify(entry["password"], password):
        return False
    _checked.add(seen)
    return True


def _env(path, owner, env) -> dict:
    env = env or {}
    if not isinstance(env, dict):
        sys.exit(f"{path}: the env of {owner} is not NAME: VALUE lines")
    return {str(k): str(v) for k, v in env.items()}


def read_users_file(path: Path, new: bool = False) -> tuple:
    """
    Return the users and the groups of the YAML file *path*. The users, by
    name: the ``password`` hash of each (``neurolang-repl-serve
    --add-user`` or ``--hash-password``), their ``groups`` and, if they have
    some, the variables of their sessions, ``env`` (the key of their AI
    assistant); the groups, by name: their ``env``, which their users have
    unless theirs say otherwise::

        users:
          ann:
            password: scrypt$16384$8$1$...
            groups: [admin, lab]
            env:
              ALBERT_API_KEY: ...
          bob:
            password: scrypt$16384$8$1$...
            groups: [lab]
        groups:
          lab:
            env:
              ALBERT_API_KEY: ...

    A file of the users alone, without ``users:``, is read as well; an
    empty one, or one without users, is that of no users if *new* (its first
    is added).
    """
    import yaml

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        sys.exit(f"The users file {path} cannot be read: {error}")
    if not isinstance(data, dict):
        sys.exit(f"The users file {path} has no users")
    sections = isinstance(data.get("users"), dict) \
        and set(data) <= {"users", "groups"} \
        and all(isinstance(e, dict) for e in data["users"].values())
    users = data["users"] if sections else data
    groups = (data.get("groups") or {}) if sections else {}
    if not (users or new):
        sys.exit(f"The users file {path} has no users")
    if not isinstance(groups, dict):
        sys.exit(f"{path}: groups is not a mapping of the groups")
    for name, entry in users.items():
        # A user named by an ORCID iD signs in with ORCID: a password is
        # theirs if they like.
        signs_in = orcid.is_orcid(str(name)) and isinstance(entry, dict) \
            and "password" not in entry
        if not isinstance(entry, dict) or not signs_in and \
                not str(entry.get("password", "")).startswith("scrypt$"):
            sys.exit(f"{path}: the user {name} has no password hash "
                     "(neurolang-repl-serve --add-user makes one), nor "
                     "is an ORCID iD")
        if not isinstance(entry.get("orcid_name", ""), str):
            sys.exit(f"{path}: the orcid_name of {name} is a text")
        if "orcid" in entry and not orcid.is_orcid(str(entry["orcid"])):
            sys.exit(f"{path}: the orcid of {name} is not an ORCID iD")
        entry["env"] = _env(path, name, entry.get("env"))
        _read_assistant(path, f"the user {name}", entry)
        member = entry.get("groups") or []
        if isinstance(member, str):
            member = [member]
        if not isinstance(member, list) or \
                not all(isinstance(group, str) for group in member):
            sys.exit(f"{path}: the groups of the user {name} are a list of "
                     "names")
        entry["groups"] = sorted({str(group) for group in member})
    # An iD signs in one user: the one it names, or the one it is merged
    # into (orcid).
    seen = {}
    for name, entry in users.items():
        for orcid_id in {str(name), str(entry.get("orcid", ""))}:
            if orcid.is_orcid(orcid_id):
                if orcid_id in seen:
                    sys.exit(f"{path}: the ORCID iD {orcid_id} signs in "
                             f"both {seen[orcid_id]} and {name}")
                seen[orcid_id] = str(name)
    kept = {}
    for name, entry in groups.items():
        if not isinstance(entry or {}, dict):
            sys.exit(f"{path}: the group {name} is not a mapping (env, llm, "
                     "own_key)")
        entry = dict(entry or {})
        group = {"env": _env(path, f"the group {name}", entry.get("env"))}
        _read_assistant(path, f"the group {name}", entry)
        group.update({k: entry[k] for k in ("llm", "own_key") if k in entry})
        kept[str(name)] = group
    return {str(name): entry for name, entry in users.items()}, kept


def _read_assistant(path, owner: str, entry: dict) -> None:
    """Check the assistant of *entry*: ``llm``, its ``provider``, ``model``,
    ``base_url`` and ``key``, and ``own_key``, whether its users may give
    their own key."""
    llm = entry.get("llm")
    if llm is not None:
        if not isinstance(llm, dict) or not llm.get("provider"):
            sys.exit(f"{path}: the llm of {owner} has no provider")
        entry["llm"] = {k: str(llm[k]) for k in
                        ("provider", "model", "base_url", "key") if llm.get(k)}
    if "own_key" in entry and not isinstance(entry["own_key"], bool):
        sys.exit(f"{path}: own_key of {owner} is true or false")


def read_users(path: Path) -> dict:
    """Return the users of the users file *path* (see
    :func:`read_users_file`)."""
    return read_users_file(path)[0]


class Users:
    """
    The users of the users file *path* as it is now: read again when it
    changes (an edit by hand), and written by the admin console, at once
    (a new file in its place), readable by its owner only. Those of the
    group ``admin`` have the console. A change reads the file again and
    writes it holding a lock of the file (``users.yaml.lock``), so that the
    changes of the server and of the command line all stay.
    """

    ADMIN = "admin"

    def __init__(self, path: Path, new: bool = False):
        """*new*: the file may not be there yet, or be empty (its first
        user is added from the command line)."""
        self.path = Path(path)
        self.new = new
        self._lock = threading.RLock()
        self._users, self._groups = {}, {}
        self._seen = None
        self._read()

    def _stamp(self, stat=None):
        """What changes when the file does, even within a tick of its
        file system's clock."""
        try:
            stat = stat or self.path.stat()
        except OSError:
            return None
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def _read(self) -> None:
        """Read the file as it is now (stop, telling why, if it cannot)."""
        stamp = self._stamp()  # before: a write meanwhile is read later
        if stamp is None and self.new:
            self._users, self._groups = {}, {}
        else:
            self._users, self._groups = read_users_file(self.path, self.new)
        self._seen = stamp

    @contextlib.contextmanager
    def _changing(self):
        """Hold the lock of the file, with the users read again."""
        import fcntl

        with self._lock, open(f"{self.path}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._read()
            except SystemExit as error:  # an edit by hand to fix first
                raise ValueError(str(error)) from None
            yield

    def current(self) -> dict:
        """Return the users, read again if the file changed."""
        with self._lock:
            stamp = self._stamp()
            if stamp is not None and stamp != self._seen:
                self._seen = stamp
                try:
                    self._users, self._groups = read_users_file(
                        self.path, self.new)
                except SystemExit as error:  # kept as they were
                    print(f"The users file is not read again: {error}",
                          file=sys.stderr, flush=True)
            return self._users

    def check(self, name: str, password: bytes) -> bool:
        return check_password(self.current(), name, password)

    def groups(self, name: str) -> list:
        return list(self.current().get(name, {}).get("groups", []))

    def is_admin(self, name: str) -> bool:
        return self.ADMIN in self.groups(name)

    def group_envs(self) -> dict:
        """Return the variables of the groups, by group."""
        self.current()
        return {name: dict(g["env"]) for name, g in self._groups.items()}

    def env(self, name: str) -> dict:
        """Return the variables of the sessions of *name*: those of their
        groups (of the first, by name, that has one), and their own over
        them."""
        entry = self.current().get(name, {})
        env = {}
        groups = self.group_envs()
        for group in entry.get("groups", []):
            for variable, value in groups.get(group, {}).items():
                env.setdefault(variable, value)
        return {**env, **entry.get("env", {})}

    def assistant(self, name: str):
        """Return the assistant *name* has (``llm``: provider, model, key),
        with whose it is, ``source``: theirs (``own``) or that of the first
        of their groups, by name, that has one (``group:NAME``); or None."""
        entry = self.current().get(name, {})
        if entry.get("llm"):
            return {**entry["llm"], "source": "own"}
        for group in entry.get("groups", []):
            llm = self._groups.get(group, {}).get("llm")
            if llm:
                return {**llm, "source": f"group:{group}"}
        return None

    def own_key(self, name: str) -> bool:
        """Whether *name* may give their own key: as they say, else not if
        one of their groups says not, else they may."""
        entry = self.current().get(name, {})
        if "own_key" in entry:
            return entry["own_key"]
        return all(self._groups.get(g, {}).get("own_key", True)
                   for g in entry.get("groups", []))

    def identity(self, name: str) -> tuple:
        """Return the variables of the sessions of *name* and their groups,
        as :class:`~neurolang_apps.utils.tui.sessions.Sessions` asks: their
        assistant and whether they may give their own key go there too."""
        entry = self.current().get(name, {})
        env = self.env(name)
        assistant = self.assistant(name)
        if assistant:
            env[ASSISTANT_ENV] = json.dumps(
                {k: v for k, v in assistant.items() if k != "source"})
        if not self.own_key(name):
            env[OWN_KEY_ENV] = "0"
        return env, ",".join(entry.get("groups", []))

    def status(self, name: str) -> dict:
        """Return, for the console and the account page, whose assistant
        *name* has (see :func:`key_status`) and whether they may give
        their own key; never a key."""
        entry = self.current()[name]
        allowed = {"own_key": self.own_key(name),
                   "variables": sorted(entry["env"])}
        assistant = self.assistant(name)
        if not assistant:
            return {**key_status(entry, self.group_envs()), **allowed}
        source = assistant["source"]
        return {**allowed,
                "assistant": "own" if source == "own" else "group",
                **({} if source == "own" else {"group": source[6:]}),
                "provider": f"{assistant['provider']} "
                            f"({assistant.get('model') or 'its model'})",
                "variable": ""}

    def group_status(self, group: str) -> dict:
        """Return a group's assistant and settings, for the console."""
        entry = self._groups.get(group, {})
        llm = entry.get("llm")
        return {"variables": sorted(entry.get("env", {})),
                "llm": f"{llm['provider']} ({llm.get('model') or 'its model'})"
                       if llm else "",
                "own_key": entry.get("own_key", True)}

    def change_group(self, group: str, env=None, llm=None,
                     own_key=None) -> None:
        """Change the variables *env* of *group* (a None value removes
        one), its assistant *llm* and whether its users may give their own
        key, *own_key* (see :meth:`change`); a group left with nothing is
        forgotten."""
        with self._changing():
            if not group or not group.strip():
                raise ValueError("A group's name is not empty")
            groups = {n: {**g, "env": dict(g["env"])}
                      for n, g in self._groups.items()}
            entry = groups.setdefault(group, {"env": {}})
            _change_env(entry["env"], env)
            _change_assistant(entry, llm, own_key)
            if not entry["env"] and set(entry) == {"env"}:
                del groups[group]
            self._write(self._users, groups)

    def change(self, name: str, password=None, groups=None, env=None,
               create: bool = False, llm=None, own_key=None,
               orcid_name=None) -> None:
        """Add the user *name* (*create*) or change theirs: the *password*,
        the *groups*, the variables *env* (a None value removes one), their
        assistant *llm* (see :func:`_change_assistant`), whether they may
        give their own key, *own_key* (``"inherit"``: as their groups say),
        and the name ORCID gives a user named by their iD, *orcid_name*.
        A new user has a password, or an ORCID iD as their name.
        """
        if password and len(password) < SHORTEST_PASSWORD:
            raise ValueError(f"A password has at least {SHORTEST_PASSWORD} "
                             "characters")
        hashed = password and hash_password(password)  # slow: not locked
        with self._changing():
            users = {n: dict(e, env=dict(e["env"]))
                     for n, e in self._users.items()}
            if create == (name in users):
                raise ValueError(f"The user {name} "
                                 + ("exists" if create else "does not exist"))
            if not name or not name.strip() or ":" in name:
                raise ValueError("A user's name is not empty and has no ':'")
            entry = users.setdefault(name, {"env": {}, "groups": []})
            if create and not password and not orcid.is_orcid(name):
                raise ValueError("A new user needs a password, or an ORCID "
                                 "iD as their name (they sign in with "
                                 "ORCID)")
            if orcid_name is not None:
                entry["orcid_name"] = str(orcid_name)
            if hashed:
                entry["password"] = hashed
            if groups is not None:
                entry["groups"] = sorted({str(g).strip() for g in groups
                                          if str(g).strip()})
            _change_env(entry["env"], env)
            _change_assistant(entry, llm, own_key)
            self._keep_an_admin(users)
            self._write(users)

    def by_orcid(self, orcid_id: str):
        """Return the user who signs in with the ORCID iD *orcid_id*: the
        one it names, or the one it was merged into (``orcid``); None."""
        users = self.current()
        if orcid_id in users:
            return orcid_id
        return next((name for name, entry in users.items()
                     if entry.get("orcid") == orcid_id), None)

    def seen_orcid(self, user: str, name: str) -> None:
        """Keep the name ORCID gives the *user* who signed in with it, if
        it changed."""
        entry = self.current().get(user, {})
        if name and entry.get("orcid_name") != name:
            self.change(user, orcid_name=name)

    def merge_orcid(self, orcid_id: str, into: str) -> None:
        """
        Merge the user named by the ORCID iD *orcid_id* into the user
        *into*, who then signs in with it too (their ``orcid``): their
        groups together; *into*'s own assistant, variables and own key
        kept, the ORCID user's taken where *into* has none. The ORCID user
        is no more.
        """
        with self._changing():
            users = {n: dict(e, env=dict(e["env"]))
                     for n, e in self._users.items()}
            source, target = users.get(orcid_id), users.get(into)
            if source is None or not orcid.is_orcid(orcid_id):
                raise ValueError(f"No user named by the ORCID iD {orcid_id}")
            if target is None or orcid.is_orcid(into):
                raise ValueError(f"{into} is not a user with a name of their "
                                 "own")
            if target.get("orcid"):
                raise ValueError(f"{into} signs in with the ORCID iD "
                                 f"{target['orcid']} already: separate it "
                                 "first")
            target["orcid"] = orcid_id
            if source.get("orcid_name"):
                target["orcid_name"] = source["orcid_name"]
            target["groups"] = sorted(set(target["groups"])
                                      | set(source["groups"]))
            for key in ("llm", "own_key"):
                if key in source and key not in target:
                    target[key] = source[key]
            for variable, value in source["env"].items():
                target["env"].setdefault(variable, value)
            del users[orcid_id]
            self._write(users)

    def link_orcid(self, user: str, orcid_id: str, orcid_name: str = "") \
            -> str:
        """
        Let *user*, of a name of their own, sign in with the ORCID iD
        *orcid_id* too, which they proved theirs by signing in with it:
        ``"linked"``; if the iD names an ORCID user, merge them into
        *user* (:meth:`merge_orcid`): ``"merged"``. ValueError if the iD
        signs in another user, or *user* has one already.
        """
        owner = self.by_orcid(orcid_id)
        if owner == orcid_id:
            self.merge_orcid(orcid_id, user)
            return "merged"
        if owner is not None:
            raise ValueError(f"The ORCID iD {orcid_id} signs in "
                             f"{'you' if owner == user else owner} already")
        with self._changing():
            users = {n: dict(e, env=dict(e["env"]))
                     for n, e in self._users.items()}
            entry = users.get(user)
            if entry is None or orcid.is_orcid(user):
                raise ValueError(f"{user} is not a user with a name of "
                                 "their own")
            if entry.get("orcid"):
                raise ValueError(f"You sign in with the ORCID iD "
                                 f"{entry['orcid']} already")
            entry["orcid"] = orcid_id
            if orcid_name:
                entry["orcid_name"] = orcid_name
            self._write(users)
        return "linked"

    def unlink_orcid(self, user: str) -> str:
        """Let *user* sign in with their password only, no more with their
        ORCID iD; return it. ValueError if they have none, or no
        password."""
        with self._changing():
            users = {n: dict(e, env=dict(e["env"]))
                     for n, e in self._users.items()}
            entry = users.get(user) or {}
            if not entry.get("orcid"):
                raise ValueError("You have no ORCID iD to unlink")
            if "password" not in entry:
                raise ValueError("Without a password you would not log in")
            orcid_id = entry.pop("orcid")
            entry.pop("orcid_name", None)
            self._write(users)
            return orcid_id

    def separate_orcid(self, name: str) -> str:
        """
        Separate the ORCID iD of the user *name* from them: a user of its
        own again, named by it, in the groups of *name* but ``admin``
        (which an admin gives again, if it is theirs); return the iD.
        """
        with self._changing():
            users = {n: dict(e, env=dict(e["env"]))
                     for n, e in self._users.items()}
            entry = users.get(name)
            if entry is None or not entry.get("orcid"):
                raise ValueError(f"{name} has no ORCID iD to separate")
            orcid_id = entry.pop("orcid")
            separated = {"env": {}, "groups": [
                g for g in entry["groups"] if g != self.ADMIN]}
            if entry.get("orcid_name"):
                separated["orcid_name"] = entry.pop("orcid_name")
            users[orcid_id] = separated
            self._write(users)
            return orcid_id

    def remove(self, name: str) -> None:
        with self._changing():
            users = dict(self._users)
            if users.pop(name, None) is None:
                raise ValueError(f"The user {name} does not exist")
            if not users:
                raise ValueError("The users file keeps a user")
            self._keep_an_admin(users)
            self._write(users)

    def _keep_an_admin(self, users: dict) -> None:
        if any(self.ADMIN in e["groups"] for e in self._users.values()) \
                and not any(self.ADMIN in e["groups"] for e in users.values()):
            raise ValueError("The users keep an admin")

    def _write(self, users: dict, groups=None) -> None:
        import tempfile

        import yaml

        groups = self._groups if groups is None else groups
        kept = {"users": {name: {key: value for key, value in entry.items()
                                 if value or key in ("password", "own_key")}
                          for name, entry in users.items()}}
        if groups:
            kept["groups"] = groups
        handle, temporary = tempfile.mkstemp(dir=self.path.parent,
                                             prefix=".users-")
        try:
            with os.fdopen(handle, "w") as file:  # mkstemp's mode: 600
                yaml.safe_dump(kept, file, sort_keys=True,
                               allow_unicode=True)
            try:  # a file the server reads, or nothing
                written = read_users_file(Path(temporary))
            except SystemExit as error:
                raise ValueError(str(error)) from None
            stamp = self._stamp(os.stat(temporary))  # a rename keeps it
            os.replace(temporary, self.path)
        except BaseException:
            os.unlink(temporary)
            raise
        self._users, self._groups = written
        self._seen = stamp


def _change_assistant(entry: dict, llm, own_key) -> None:
    """Change the assistant of a user's or group's *entry*: *llm*, a
    provider of the page's form with its ``model``, ``base_url`` (that of
    ``custom``) and ``key`` (the key it had kept if none is given, for the
    same provider and address only), or False to remove it, None to leave
    it; *own_key* True, False, ``"inherit"`` or None."""
    if llm is False:
        entry.pop("llm", None)
    elif llm:
        provider = str(llm.get("provider", ""))
        try:
            from neurolang_apps.utils.llm.browser import PROVIDERS
        except ImportError:  # without the extra llm: not checked
            PROVIDERS = None
        if not provider or PROVIDERS is not None and provider not in PROVIDERS:
            raise ValueError(f"No provider {provider!r}: one of "
                             f"{', '.join(PROVIDERS or [])}")
        base_url = llm.get("base_url") or ""
        if PROVIDERS and not PROVIDERS[provider]["base_url"] and not base_url:
            raise ValueError(f"The assistant of {provider} needs its address, "
                             "base_url")
        old = entry.get("llm") or {}
        same = (old.get("provider"), old.get("base_url") or "") == \
            (provider, base_url)
        key = llm.get("key") or (old.get("key") if same else "")
        if not key:
            raise ValueError(f"The assistant of {provider} needs its key")
        entry["llm"] = {k: str(v) for k, v in {
            "provider": provider, "model": llm.get("model"),
            "base_url": base_url, "key": key}.items() if v}
    if own_key == "inherit":
        entry.pop("own_key", None)
    elif own_key is not None:
        entry["own_key"] = bool(own_key)


def guest_identity(strip=()) -> tuple:
    """Return the variables and the group of a guest's sessions: no key of
    the server (the variables *strip* taken out of their process), theirs
    from the page."""
    return {**{name: None for name in strip}, BRING_KEY_ENV: "1"}, \
        GUEST_GROUP


def server_key_names(env_file_names=()) -> list:
    """Return the variables holding keys of the server: those of
    ``--env-file`` and those the providers of its configuration read."""
    names = set(env_file_names)
    try:
        from neurolang_apps.utils.llm.config import load_config

        names |= {p.api_key_env for p in load_config().providers.values()
                  if p.api_key_env}
    except Exception:
        pass
    return sorted(names)


def _providers() -> list:
    """Return the providers of the page's form, or none without the extra
    llm."""
    try:
        from neurolang_apps.utils.llm.browser import providers_for_form
    except ImportError:
        return []
    return providers_for_form()


def _change_env(env: dict, changes) -> None:
    """Change the variables *env* as *changes* say (None removes one)."""
    for variable, value in (changes or {}).items():
        if not str(variable).isidentifier():
            raise ValueError(f"{variable} is not a variable's name")
        if value is None:
            env.pop(variable, None)
        else:
            env[variable] = str(value)


def key_status(entry: dict, groups=None) -> dict:
    """Return, for the admin console, the variables a user (or a group)
    has (their names) and whose key their assistant answers with: ``own``,
    that of one of their *groups* (``group``, with its name in ``group``),
    the server's (``server``), ``none`` (the user gives it in the page) or
    ``unneeded``; never a key."""
    from neurolang_apps.utils.llm.config import load_config

    env = entry.get("env", {})
    status = {"variables": sorted(env)}
    from_group = {}
    for group in entry.get("groups", []):
        for variable, value in (groups or {}).get(group, {}).items():
            from_group.setdefault(variable, (group, value))
    config = env.get("NEUROLANG_LLM_CONFIG") or \
        from_group.get("NEUROLANG_LLM_CONFIG", ("", None))[1]
    try:
        provider = load_config(config).provider()
    except Exception as error:
        return {**status, "assistant": "unknown",
                "detail": type(error).__name__}
    variable = provider.api_key_env
    status.update(provider=f"{provider.name} ({provider.model})",
                  variable=variable or "")
    if not variable:
        return {**status, "assistant": "unneeded"}
    if env.get(variable):
        return {**status, "assistant": "own"}
    if variable in from_group:
        return {**status, "assistant": "group",
                "group": from_group[variable][0]}
    if os.environ.get(variable):
        return {**status, "assistant": "server"}
    return {**status, "assistant": "none"}


def users_assistants(users: "Users") -> dict:
    """Return, by user of *users*, the provider and model their assistant
    answers with by default (their ``llm``, their groups', else the
    server's configuration with their variables and their groups'), or
    None if it has no key (they give theirs in the page); warn if the
    variables of a user give no key."""
    assistants = {}
    for name in users.current():
        assistant = users.assistant(name)
        if assistant:
            assistants[name] = (f"{assistant['provider']} "
                                f"({assistant.get('model') or 'its model'})")
            continue
        env = users.env(name)
        saved = dict(os.environ)
        os.environ.update(env)
        try:
            assistants[name] = default_assistant(quiet=True)
        except SystemExit as error:
            if env:
                print(f"The variables of {name} give their assistant no "
                      f"key: {error}", file=sys.stderr, flush=True)
            assistants[name] = None
        finally:
            os.environ.clear()
            os.environ.update(saved)
    return assistants


def _fingerprint(entry: dict) -> str:
    """What changes when a user's password does: a login made before is
    then out (that of ORCID, without a password, stays)."""
    return hashlib.sha256(
        entry.get("password", "orcid").encode()).hexdigest()[:16]


def sign_login(secret: bytes, name: str, users, seconds: int) -> str:
    """Return the value of the login cookie of *name* (a user, or a guest),
    good for *seconds*, signed with *secret*."""
    login = {"u": name, "e": int(time.time()) + seconds, "i": time.time()}
    if name.startswith(GUEST_PREFIX):
        login["g"] = 1
    else:
        login["p"] = _fingerprint(users.current()[name])
    payload = base64.urlsafe_b64encode(json.dumps(login).encode()).decode()
    signature = hmac.new(secret, payload.encode(), "sha256").hexdigest()
    return f"{payload}.{signature}"


def check_login(secret: bytes, value: str, users, guests: bool = False,
                logged_out=None):
    """Return the user of the login cookie *value* if it is good: signed
    with *secret*, not expired, made after its user last logged out (as
    *logged_out* says, by name), of a user who is still there with the same
    password, or of a guest if *guests* are let in; else None."""
    payload, _, signature = value.partition(".")
    good = hmac.new(secret, payload.encode(), "sha256").hexdigest()
    if not hmac.compare_digest(signature.encode(), good.encode()):
        return None
    try:
        login = json.loads(base64.urlsafe_b64decode(payload.encode()))
    except ValueError:
        return None
    if not isinstance(login, dict) or login.get("e", 0) < time.time():
        return None
    name = str(login.get("u", ""))
    if login.get("i", 0) <= (logged_out or {}).get(name, -1):
        return None
    if login.get("g"):
        return name if guests and name.startswith(GUEST_PREFIX) else None
    entry = users.current().get(name)
    if entry is None or login.get("p") != _fingerprint(entry):
        return None
    return name


def read_password(path: Path) -> str:
    """Return the password in the first line of the file *path*, as it is
    but for its line ending."""
    password = read_file(path).partition("\n")[0].rstrip("\r")
    if not password:
        sys.exit(f"The password file {path} has no password on its first line")
    return password


# The options a configuration file cannot give: what to do now, not how
# to serve.
NOT_CONFIGURED = {"help", "config", "daemon", "hash_password", "list_users",
                  "add_user", "set_password", "set_groups", "set_key",
                  "set_group_key", "set_llm", "set_group_llm", "own_key",
                  "remove_user", "list_requests", "approve_request",
                  "decline_request", "merge_orcid", "separate_orcid",
                  "groups"}


def read_config(parser, path: Path) -> dict:
    """
    Return the options of the YAML file *path*, as the defaults of the
    *parser*: each by its name, that of the option (``data-dir``) or of
    its value (``data_dir``), a list for one given several times
    (``engines: [neurosynth]``), true for a flag; a relative path is
    relative to the file. Stop, telling why, if it is not one.
    """
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        sys.exit(f"The configuration {path} cannot be read: {error}")
    if not isinstance(data, dict):
        sys.exit(f"The configuration {path} is not a mapping of options")
    actions = {}
    for action in parser._actions:
        if action.dest in NOT_CONFIGURED:
            continue
        actions[action.dest] = action
        for option in action.option_strings:
            actions[option.lstrip("-").replace("-", "_")] = action
    base = Path(path).resolve().parent
    options = {}
    for key, value in data.items():
        action = actions.get(str(key).replace("-", "_"))
        if action is None:
            sys.exit(f"{path}: no option {key!r}; the options are "
                     f"{', '.join(sorted({a.dest for a in actions.values()}))}")
        many = isinstance(action, argparse._AppendAction)
        values = value if isinstance(value, list) else [value]
        if many is not isinstance(value, list) and value is not None:
            sys.exit(f"{path}: {key} is " + ("a list" if many else "one value"))
        if action.nargs == 0 and not isinstance(value, bool):
            sys.exit(f"{path}: {key} is true or false")
        if action.type is not None and action.nargs != 0:
            try:
                values = [None if v is None else action.type(v)
                          for v in values]
            except (TypeError, ValueError):
                sys.exit(f"{path}: {key} is not a "
                         f"{getattr(action.type, '__name__', 'value')}")
        if action.type is Path:
            values = [v if v is None or v.is_absolute() else base / v
                      for v in values]
        if action.choices is not None:
            bad = [v for v in values if v not in action.choices]
            if bad:
                sys.exit(f"{path}: {key} is one of "
                         f"{', '.join(map(str, action.choices))}, not "
                         f"{', '.join(map(str, bad))}")
        options[action.dest] = values if many else values[0]
    return options


# What a file of secrets holds.
SECRETS = ("orcid", "env", "password", "key_password")


def read_secrets(path: Path) -> dict:
    """
    Return the secrets of the YAML file *path*, readable by its owner
    only: ``orcid`` (``client_id``, ``client_secret``...), ``env`` (the
    variables of the server), ``password`` and ``key_password``. Stop,
    telling why, if it is not one, or others may read it.
    """
    import yaml

    path = Path(path)
    try:
        mode = path.stat().st_mode
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        sys.exit(f"The secrets {path} cannot be read: {error}")
    if mode & 0o077:
        sys.exit(f"The secrets {path} may be read by others: chmod 600 "
                 f"{path}")
    if not isinstance(data, dict) or set(data) - set(SECRETS):
        sys.exit(f"{path}: the secrets are those of {', '.join(SECRETS)}")
    if not isinstance(data.get("env", {}), dict):
        sys.exit(f"{path}: env is a mapping NAME: VALUE")
    if not isinstance(data.get("orcid", {}), dict):
        sys.exit(f"{path}: orcid is a mapping (client_id, client_secret)")
    secrets_ = {k: v for k, v in data.items() if v is not None}
    if "env" in secrets_:
        secrets_["env"] = {str(k): str(v) for k, v in secrets_["env"].items()}
    for key in ("password", "key_password"):
        if key in secrets_:
            secrets_[key] = str(secrets_[key])
    return secrets_


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


def default_assistant(quiet: bool = False) -> str:
    """Return the provider and the model the assistant answers with by
    default, with the keys of the environment; stop, telling why, if it
    cannot (its key is not there)."""
    from neurolang_apps.utils.llm.config import load_config

    try:
        provider = load_config().provider()
        provider.api_key()
    except Exception as error:
        sys.exit(f"{error}" if quiet else
                 f"The assistant cannot answer with --env-file: {error}")
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
    if args.config:  # the file says it all, as it will when it changes
        return [python, "-m", "serve_repl", "--no-browser", "--config",
                str(args.config.resolve())]
    command = [
        python, "-m", "serve_repl", "--no-browser",
        "--host", args.host, "--port", str(args.port),
        "--site", str(args.site.resolve()),
        "--embeddings", str(args.embeddings.resolve()),
        "--downloads", str(args.downloads.resolve()),
    ]
    for engine in args.engines or ():
        command += ["--engine", engine]
    for name in args.server_names or ():
        command += ["--server-name", name]
    if args.local:
        command += ["--local", "--data-dir", str(args.data_dir.resolve()),
                    "--resolution", str(args.resolution)]
        command += ["--light"] if args.light else []
        command += [] if args.prepare else ["--no-prepare"]
    if args.users:
        command += ["--max-sessions", str(args.max_sessions),
                    "--idle-minutes", str(args.idle_minutes),
                    "--query-minutes", str(args.query_minutes)]
        if args.max_running:
            command += ["--max-running", str(args.max_running)]
    if args.secure_cookie:
        command += ["--secure-cookie"]
    if args.dashboard:
        command += ["--dashboard"]
    if args.guests:
        command += ["--guests", "--max-guest-sessions",
                    str(args.max_guest_sessions)]
    for option, value in (("--pr", args.pr), ("--run-id", args.run_id)):
        if value is not None:
            command += [option, str(value), "--repo", args.repo]
    for option, path in (
        ("--cert", args.cert), ("--key", args.key),
        ("--key-password-file", args.key_password_file),
        ("--password-file", args.password_file), ("--env-file", args.env_file),
        ("--users", args.users), ("--feedback-db", args.feedback_db),
        ("--orcid", args.orcid), ("--secrets", args.secrets),
        ("--activity-db", args.activity_db),
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
DASHBOARD_IMPORTS = "datasette, datasette_vega"


def check_service_python(command: list, local: bool,
                         dashboard: bool = False) -> None:
    """Stop, telling which extras to install with, if the Python of the
    service *command* cannot import what it serves with (its *dashboard*'s
    too)."""
    modules = SERVICE_IMPORTS + (f", {LOCAL_IMPORTS}" if local else "") + \
        (f", {DASHBOARD_IMPORTS}" if dashboard else "")
    extras = (LOCAL_EXTRAS if local else "--extra apps") + \
        (" --extra dashboard" if dashboard else "")
    result = subprocess.run(
        [command[0], "-c", f"import {modules}"], capture_output=True,
        text=True, env=service_environment(command[0]), cwd="/",
    )
    if result.returncode:
        error = (result.stderr.strip().splitlines() or ["?"])[-1]
        sys.exit(
            f"The Python of the service, {command[0]}, cannot serve: "
            f"{error}. Install it from the environment that has what it "
            f"needs: uv run {extras} "
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
    parser.add_argument(
        "--config", type=Path, metavar="FILE",
        help="the YAML FILE of the options, each by its name (port, "
        "data_dir or data-dir, engines: [...]...), its paths relative to "
        "it; the options given here replace its own",
    )
    parser.add_argument(
        "--secrets", type=Path, metavar="FILE",
        help="the YAML FILE of the secrets, readable by its owner only: "
        "orcid (the client of --orcid), env (the variables of --env-file), "
        "password (of --password-file), key_password (of "
        "--key-password-file)",
    )
    parser.add_argument("--port", type=int, default=DEFAULTS["port"])
    parser.add_argument(
        "--host", default=DEFAULTS["host"],
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
        "--downloads", type=Path, default=DEFAULT_DOWNLOADS,
        help="directory keeping what the engines of the bundle download, "
        "for the next build (default: %(default)s)",
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
        "--users", type=Path, metavar="FILE",
        help="ask every visitor the name and the password of a user of the "
        "YAML FILE (their hashes, --hash-password); with --local, each tab "
        "of a user has a session of its own, in a process of its own, "
        "with the variables (the assistant's key) FILE gives the user",
    )
    parser.add_argument(
        "--guests", action="store_true",
        help="with --users and --local, let a visitor in as a guest, with "
        "sessions of their own and no key of the server's: they bring "
        "their own",
    )
    parser.add_argument(
        "--orcid", type=Path, metavar="FILE",
        help="with --users, let the users named by an ORCID iD sign in with "
        "ORCID, the client of the YAML FILE (client_id, client_secret; "
        "ORCID's developer tools); an iD of no user may ask for an "
        "account, which an admin approves",
    )
    parser.add_argument(
        "--secure-cookie", action="store_true",
        help="with --users behind a reverse proxy of HTTPS (no --cert), "
        "send the login cookie over HTTPS only",
    )
    parser.add_argument(
        "--max-guest-sessions", type=int,
        default=DEFAULTS["max_guest_sessions"], metavar="N",
        help="with --guests, the sessions the guests have at most at once "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=DEFAULTS["max_sessions"],
        metavar="N",
        help="with --users and --local, the sessions at most at once "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--idle-minutes", type=float,
        default=float(DEFAULTS["idle_minutes"]),
        metavar="M",
        help="with --users and --local, end a session unused for M minutes "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max-running", type=int, metavar="N",
        help="with --users and --local, run at most N queries at once, of "
        "all the sessions, the others waiting their turn (default: the "
        "number of CPUs of this machine)",
    )
    parser.add_argument(
        "--query-minutes", type=float,
        default=float(DEFAULTS["query_minutes"]),
        metavar="M",
        help="with --users and --local, stop a query running for more than "
        "M minutes, ending its session's process (0: never; default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--feedback-db", type=Path, metavar="FILE",
        help="with --local, store the ratings of the page in the SQLite "
        "database FILE (made if need be), with the name and the groups of "
        "the user who rates (--users); the admin console shows them and "
        "exports them as YAML",
    )
    parser.add_argument(
        "--activity-db", type=Path, metavar="FILE",
        help="with --local and --users, log what each user runs, its "
        "result as text, how long it took, the CPU time and the memory of "
        "their session, in the SQLite database FILE (made if need be); the "
        "admin console shows the totals of each user and gives their log; "
        "the page tells the users their inputs are logged",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="with --activity-db, serve a dashboard of the activity log and "
        "the ratings to the admins, at /admin/logs/: Datasette, with its "
        "charts and queries of the uptime (install with --extra dashboard)",
    )
    parser.add_argument(
        "--hash-password", action="store_true",
        help="ask a password and print its hash, for a users file",
    )
    # The users of --users FILE, changed from the command line as the
    # admin console does; a server running reads the file again.
    manage = parser.add_mutually_exclusive_group()
    manage.add_argument("--list-users", action="store_true",
                        help="list the users of --users, their groups and "
                        "their assistant's key (whose, never the key)")
    manage.add_argument("--add-user", metavar="NAME",
                        help="add the user NAME to --users (made if need "
                        "be), asking their password; with --groups")
    manage.add_argument("--set-password", metavar="NAME",
                        help="ask a new password of the user NAME")
    manage.add_argument("--set-groups", metavar="NAME",
                        help="give the user NAME the --groups")
    manage.add_argument("--set-key", nargs=2, metavar=("NAME", "VARIABLE"),
                        help="ask the key of the user NAME's assistant, "
                        "in their variable VARIABLE (e.g. ALBERT_API_KEY); "
                        "an empty one removes it")
    manage.add_argument("--set-group-key", nargs=2,
                        metavar=("GROUP", "VARIABLE"),
                        help="ask the key of the assistant of the users of "
                        "GROUP, in their variable VARIABLE, which a user's "
                        "own replaces; an empty one removes it")
    manage.add_argument("--set-llm", nargs="+",
                        metavar=("NAME PROVIDER", "MODEL"),
                        help="give the user NAME the assistant of PROVIDER "
                        "(anthropic, openai, openrouter...), with its MODEL "
                        "if given, asking its key; an empty key removes it")
    manage.add_argument("--set-group-llm", nargs="+",
                        metavar=("GROUP PROVIDER", "MODEL"),
                        help="give the users of GROUP the assistant of "
                        "PROVIDER, as --set-llm")
    manage.add_argument("--own-key", nargs=2,
                        metavar=("NAME|group:GROUP", "yes|no|inherit"),
                        help="let a user, or the users of a group, give "
                        "their own key in the page, or not (a user's "
                        "inherit: as their groups say)")
    manage.add_argument("--remove-user", metavar="NAME",
                        help="remove the user NAME")
    manage.add_argument("--merge-orcid", nargs=2, metavar=("ORCID", "USER"),
                        help="merge the user named by the ORCID iD into the "
                        "user USER, who then signs in with it too")
    manage.add_argument("--separate-orcid", metavar="USER",
                        help="separate the ORCID iD of USER: a user of its "
                        "own again, in USER's groups but admin")
    manage.add_argument("--list-requests", action="store_true",
                        help="list the requests of an account of those who "
                        "signed in with ORCID (--orcid)")
    manage.add_argument("--approve-request", metavar="ORCID",
                        help="add the user of the request of ORCID, with "
                        "--groups")
    manage.add_argument("--decline-request", metavar="ORCID",
                        help="decline the request of ORCID")
    parser.add_argument("--groups", default="", metavar="G1,G2",
                        help="the groups of --add-user, --set-groups or "
                        "--approve-request (admin: the admin console)")
    parser.add_argument(
        "--env-file", type=Path, metavar="FILE",
        help="with --local and --password-file or --users, put the "
        "NAME=VALUE lines of FILE in the server's environment: the key of "
        "the AI assistant's provider (its api_key_env), which those who "
        "log in use unless their page, or their user, gives one",
    )
    parser.add_argument(
        "--python", metavar="PATH",
        help="with --daemon, the Python the service runs (default: that of "
        "the environment installing it)",
    )
    parser.add_argument(
        "--daemon", choices=("install", "reinstall", "remove", "status"),
        help="install (building its bundle now), reinstall or remove a "
        "service of this user serving the REPL as the other options say, "
        "at login: systemd's user manager, or launchd on macOS; or tell "
        "whether it runs, since when and how often it restarted",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="run NeuroLang in the Python of this machine rather than in "
        "the browser: no bundle is built",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(DEFAULTS["data_dir"]),
        help="with --local, the data directory of the engines (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--resolution", type=float, default=float(DEFAULTS["resolution"]),
        metavar="MM",
        help="with --local, the resolution of the engines' MNI grid "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="with --local, leave out the analyses a browser cannot hold, "
        "as the page does",
    )
    parser.add_argument(
        "--no-prepare", action="store_false", dest="prepare",
        default=DEFAULTS["prepare"],
        help="with --local, do not build each engine before serving (which "
        "keeps what it downloads in --data-dir): its first session does",
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
    config, _ = parser.parse_known_args(argv)
    if config.config:
        parser.set_defaults(**read_config(parser, config.config))
    args = parser.parse_args(argv)
    args.secret = read_secrets(args.secrets) if args.secrets else {}
    for option, key in (("password_file", "password"), ("env_file", "env"),
                        ("orcid", "orcid"),
                        ("key_password_file", "key_password")):
        if getattr(args, option) and key in args.secret:
            parser.error(f"--{option.replace('_', '-')} or the {key} of "
                         f"{args.secrets}, not both")
    password = args.password_file or "password" in args.secret
    if (args.pr or args.run_id) and (args.rebuild or args.clear):
        parser.error("--pr and --run-id build nothing: no --rebuild nor "
                     "--clear")
    if args.local and (args.pr or args.run_id or args.rebuild
                       or args.clear):
        parser.error("--local builds no bundle: no --pr, --run-id, "
                     "--rebuild nor --clear")
    if (args.key or args.key_password_file
            or "key_password" in args.secret) and not args.cert:
        parser.error("--key and --key-password-file are those of --cert")
    if args.python and not args.daemon:
        parser.error("--python is that of the service: with --daemon")
    if password and args.users:
        parser.error("--password-file or --users, not both")
    if args.max_sessions < 1 or not args.idle_minutes > 0:
        parser.error("--max-sessions is at least 1, --idle-minutes above 0")
    if (args.max_running is not None and args.max_running < 1) or \
            not args.query_minutes >= 0:
        parser.error("--max-running is at least 1, --query-minutes 0 or "
                     "more")
    if ((password or args.users) and not args.cert
            and args.host not in ("127.0.0.1", "localhost", "::1")):
        parser.error("a password goes in clear without --cert: give a "
                     "certificate")
    if args.secure_cookie and not args.users:
        parser.error("--secure-cookie is that of the login of --users")
    if (args.orcid or "orcid" in args.secret) and not args.users:
        parser.error("--orcid signs in the users of --users")
    if args.guests and not (args.users and args.local):
        parser.error("--guests needs --users and --local")
    if args.feedback_db and not args.local:
        parser.error("--feedback-db stores the ratings of --local")
    if args.activity_db and not (args.local and args.users):
        parser.error("--activity-db logs the sessions of --local --users")
    if args.dashboard and not args.activity_db:
        parser.error("--dashboard shows the log of --activity-db")
    if (args.env_file or "env" in args.secret) and not (
            args.local and (password or args.users)):
        parser.error("--env-file is the key of those who log in: with "
                     "--local and --password-file or --users")
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
    check_service_python(command, args.local, args.dashboard)
    if not (args.local or args.pr is not None or args.run_id is not None):
        prepare_site(args.site, args.rebuild, args.embeddings,
                     clear=args.clear, engines=args.engines)
    if args.daemon == "reinstall":
        remove_service()
    install_service(command)


def ask_password(prompt: str = "Password: ") -> str:
    """Ask a password twice; stop if it is empty or they differ."""
    password = getpass.getpass(prompt)
    if not password or password != getpass.getpass("Again: "):
        sys.exit("The passwords are empty or differ")
    return password


def ask_password_hash() -> None:
    """Ask a password twice and print its hash, for a users file."""
    print(hash_password(ask_password()))


def manage_users(args) -> None:
    """Change the users of ``--users`` as the command line *args* say."""
    if not args.users:
        sys.exit("Give the users file: --users FILE")
    users = Users(args.users, new=bool(args.add_user))
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    try:
        if args.list_users:
            groups = users.group_envs()
            for name, entry in sorted(users.current().items()):
                status = users.status(name)
                print(f"{name}\tgroups: {', '.join(entry['groups']) or '-'}"
                      f"\tassistant: {status['assistant']}"
                      + (f" of {status['group']}" if "group" in status
                         else "")
                      + (f" ({status['variable']})"
                         if status.get("variable") else "")
                      + f"\t{status.get('provider', '')}"
                      + ("" if status["own_key"] else "\tno own key"))
            for group in sorted(groups):
                info = users.group_status(group)
                print(f"group {group}\tassistant: {info['llm'] or '-'}"
                      f"\tvariables: {', '.join(info['variables']) or '-'}"
                      + ("" if info["own_key"] else "\tno own key"))
            return
        requests = orcid.AccountRequests(orcid.requests_path(args.users))
        if args.list_requests:
            for asked in requests.list():
                when = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(asked["when"]))
                print(f"{asked['orcid']}\t{asked['name'] or '-'}\t{when}"
                      f"\t{asked['message']}")
            return
        if args.merge_orcid:
            orcid_id, into = args.merge_orcid
            users.merge_orcid(orcid_id, into)
            done = f"{into} signs in with the ORCID iD {orcid_id} now"
        elif args.separate_orcid:
            orcid_id = users.separate_orcid(args.separate_orcid)
            done = (f"Separated {orcid_id} from {args.separate_orcid}: a "
                    "user of its own")
        elif args.approve_request or args.decline_request:
            orcid_id = args.approve_request or args.decline_request
            asked = requests.current().get(orcid_id)
            if asked is None:
                sys.exit(f"No request of {orcid_id}")
            if args.approve_request:
                users.change(orcid_id, groups=groups, create=True,
                             orcid_name=asked.get("name", ""))
            requests.remove(orcid_id)
            done = (f"Added {orcid_id}, in: {', '.join(groups) or '-'}"
                    if args.approve_request
                    else f"Declined the request of {orcid_id}")
        elif args.add_user and orcid.is_orcid(args.add_user):
            # they sign in with ORCID: no password
            users.change(args.add_user, groups=groups, create=True)
            done = f"Added {args.add_user}, who signs in with ORCID"
        elif args.add_user:
            users.change(args.add_user, password=ask_password(
                f"Password of {args.add_user}: "), groups=groups, create=True)
            done = f"Added {args.add_user}"
        elif args.set_password:
            users.change(args.set_password, password=ask_password(
                f"New password of {args.set_password}: "))
            done = f"Changed the password of {args.set_password}"
        elif args.set_groups:
            users.change(args.set_groups, groups=groups)
            done = f"{args.set_groups} is in: {', '.join(groups) or '-'}"
        elif args.set_llm or args.set_group_llm:
            words = args.set_llm or args.set_group_llm
            if len(words) not in (2, 3):
                sys.exit("Give the name, the provider and, if you like, "
                         "the model")
            target, provider, model = (words + [""])[:3]
            key = getpass.getpass(f"The key of {provider} for {target} "
                                  "(empty: none): ")
            llm = {"provider": provider, "model": model, "key": key,
                   "base_url": input("The address of its API (base_url): ")
                   if key and provider == "custom" else ""} \
                if key else False
            if args.set_llm:
                users.change(target, llm=llm)
            else:
                users.change_group(target, llm=llm)
            done = (f"{'Set' if key else 'Removed'} the assistant of "
                    f"{'the group ' if args.set_group_llm else ''}{target}")
        elif args.own_key:
            target, choice = args.own_key
            if choice not in ("yes", "no", "inherit"):
                sys.exit("--own-key: yes, no or inherit")
            own = {"yes": True, "no": False}.get(choice, "inherit")
            if target.startswith("group:"):
                if own == "inherit":
                    sys.exit("A group says yes or no")
                users.change_group(target[6:], own_key=own)
            else:
                users.change(target, own_key=own)
            done = f"{target} may give their own key: {choice}"
        elif args.set_group_key:
            group, variable = args.set_group_key
            key = getpass.getpass(f"{variable} of the group {group} "
                                  "(empty: none): ")
            users.change_group(group, {variable: key or None})
            done = (f"{'Set' if key else 'Removed'} {variable} of the "
                    f"group {group}")
        elif args.set_key:
            name, variable = args.set_key
            key = getpass.getpass(f"{variable} of {name} (empty: none): ")
            users.change(name, env={variable: key or None})
            done = f"{'Set' if key else 'Removed'} {variable} of {name}"
        else:
            users.remove(args.remove_user)
            done = f"Removed {args.remove_user}"
    except ValueError as error:
        sys.exit(str(error))
    print(f"{done} ({users.path}; a server running reads it again)")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.hash_password:
        return ask_password_hash()
    if args.daemon == "remove":  # whatever the other options
        return remove_service()
    if args.daemon == "status":
        return service_status()
    if (args.list_users or args.add_user or args.set_password
            or args.set_groups or args.set_key or args.set_group_key
            or args.set_llm or args.set_group_llm or args.own_key
            or args.remove_user or args.list_requests
            or args.approve_request or args.decline_request
            or args.merge_orcid or args.separate_orcid):
        return manage_users(args)
    # A certificate, a password or a key that cannot serve fails now, not
    # in the service.
    secret = args.secret
    tls = tls_context(args.cert, args.key, args.key_password_file,
                      ask=not args.daemon,
                      key_password=secret.get("key_password")) \
        if args.cert else None
    password = read_password(args.password_file) if args.password_file \
        else secret.get("password")
    env_file_names = []
    variables = read_env_file(args.env_file) if args.env_file \
        else secret.get("env", {})
    if variables:
        env_file_names = list(variables)
        os.environ.update(variables)
        print(f"The AI assistant answers with {default_assistant()} for "
              "those who log in", flush=True)
    users = Users(args.users) if args.users else None
    orcid_client = None
    if args.orcid or "orcid" in secret:
        try:
            orcid_client = orcid.OrcidClient.from_file(args.orcid) \
                if args.orcid else orcid.OrcidClient.from_mapping(
                    secret["orcid"], f"the orcid of {args.secrets}")
        except ValueError as error:
            sys.exit(str(error))
        print(f"The users sign in with ORCID ({orcid_client.base}) too; "
              "the requests of an account are in "
              f"{orcid.requests_path(args.users)}", flush=True)
    if users and args.local:
        for name, assistant in users_assistants(users).items():
            print(f"{name}: the AI assistant answers with "
                  f"{assistant or 'the key their page gives'}", flush=True)
    if args.daemon:
        return daemon(args)
    backend = None
    if args.local:
        from neurolang_apps.utils.tui.local_backend import LocalBackend

        args.site = args.site / "local"
        page = prepare_local_site(args.site, args.engines, bool(users))
        if args.prepare:
            prepare_engines(
                args.data_dir.resolve(),
                list(args.engines or bundle_engines()),
                args.resolution, args.light,
            )
        options = dict(resolution=args.resolution, light=args.light,
                       feedback=feedback_target())
        if users:
            from neurolang_apps.utils.tui.sessions import Sessions

            strip = server_key_names(env_file_names)

            def identity(name):
                if name.startswith(GUEST_PREFIX):
                    return guest_identity(strip)
                env, groups = users.identity(name)
                # The server's keys, under the user's and their groups'.
                return {**{n: os.environ[n] for n in strip
                           if n in os.environ}, **env}, groups

            backend = Sessions(
                args.data_dir, identity=identity, withhold=strip,
                feedback_db=args.feedback_db and args.feedback_db.resolve(),
                activity_db=args.activity_db and args.activity_db.resolve(),
                most=args.max_sessions, idle=args.idle_minutes * 60,
                running=args.max_running,
                query_seconds=args.query_minutes * 60,
                **options,
            )
            if args.activity_db:
                print("What the users run is logged in "
                      f"{args.activity_db.resolve()}", flush=True)
        else:
            if args.feedback_db:
                options["feedback"] = {
                    "path": str(args.feedback_db.resolve()), "user": "",
                    "groups": ""}
            backend = LocalBackend(args.data_dir, **options)
        if args.feedback_db:
            from neurolang_apps.utils.llm.feedback_server import (
                FeedbackStore,
            )

            FeedbackStore(args.feedback_db)  # made now, its errors here
            print(f"The ratings are stored in {args.feedback_db.resolve()}",
                  flush=True)
        print(f"NeuroLang runs in {sys.executable}, on the data of "
              f"{args.data_dir.resolve()}", flush=True)
    elif args.pr is not None or args.run_id is not None:
        page = fetch_site(args)
    else:
        page = prepare_site(
            args.site, args.rebuild, args.embeddings, clear=args.clear,
            engines=args.engines, downloads=args.downloads,
        )
    write_index(args.site, page)
    board = None
    if args.dashboard:
        board = start_dashboard(args)
    server = make_server(args.site, args.host, args.port, backend, tls,
                         password, args.server_names or (), users,
                         args.feedback_db, args.guests,
                         args.max_guest_sessions, args.secure_cookie,
                         orcid_client, board)
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    scheme = "https" if tls else "http"
    url = (f"{scheme}://{host}:{server.server_address[1]}/"
           f"{page.relative_to(args.site).as_posix()}")
    print(f"The REPL is at {url} (Ctrl+C stops it)"
          + (", for those with the password" if password else "")
          + (f", for its {len(users.current())} users; the admins' "
             "console is at /admin" if users else ""),
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    # The service manager stops the server with SIGTERM: as Ctrl+C, it
    # ends the sessions and logs its stop.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if hasattr(backend, "close"):
            backend.close()
        if board is not None:
            board.close()


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def start_dashboard(args):
    """Start the dashboard of the activity log of *args* (and of their
    ratings), or stop, telling what to install."""
    if importlib.util.find_spec("datasette") is None or \
            importlib.util.find_spec("datasette_vega") is None:
        sys.exit("--dashboard needs Datasette: uv run ... --extra dashboard "
                 "neurolang-repl-serve ...")
    from neurolang_apps.utils.tui.dashboard import Dashboard

    paths = [args.activity_db] + ([args.feedback_db] if args.feedback_db
                                  else [])
    board = Dashboard(paths)
    print("The dashboard of the activity is at /admin/logs/", flush=True)
    return board


def service_status() -> None:
    """Tell whether the service runs, since when, how often it restarted
    and how its last run ended, as its service manager says."""
    if not service_file().exists():
        print("No service of the REPL is installed", flush=True)
        return
    if sys.platform == "darwin":
        result = subprocess.run(
            [*manager(), "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
            capture_output=True, text=True)
        lines = [line.strip() for line in result.stdout.splitlines()
                 if line.strip().startswith(
                     ("state =", "pid =", "runs =", "last exit code ="))]
        log = f"~/Library/Logs/{SERVICE}.log"
    else:
        result = subprocess.run(
            [*manager(), "show", SERVICE, "--property", "ActiveState,"
             "SubState,ActiveEnterTimestamp,NRestarts,MainPID,"
             "ExecMainStatus,Result"],
            capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        log = f"journalctl --user -u {SERVICE} -n 80"
    for line in lines or [result.stderr.strip() or "no answer"]:
        print(line, flush=True)
    print(f"Its log: {log}; whether it serves: GET {HEALTH_PATH}",
          flush=True)


if __name__ == "__main__":
    main()
