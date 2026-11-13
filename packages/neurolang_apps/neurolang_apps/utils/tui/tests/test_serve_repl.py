"""Tests for the script serving the REPL in the browser on this machine."""

import base64
import importlib.util
import json
import os
import re
import plistlib
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[6]
SCRIPT = ROOT / "tools" / "serve_repl" / "__init__.py"


@pytest.fixture
def serve():
    spec = importlib.util.spec_from_file_location("serve_repl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_site_has_the_page_and_the_logo(serve, monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(serve, "build_bundle", lambda *a: built.append(a))
    page = serve.prepare_site(tmp_path)
    assert page == tmp_path / "_static" / "repl" / "index.html"
    assert (tmp_path / "_static" / "repl" / "worker.js").is_file()
    assert any((tmp_path / "_static" / "logo").rglob("*.svg"))
    assert len(built) == 1


def test_the_bundle_is_built_once_unless_asked_again(
    serve, monkeypatch, tmp_path
):
    built = []

    def build(repl, embeddings, engines=None):
        built.append(embeddings)
        (repl / "bundle.json").write_text(
            json.dumps({"engines": dict.fromkeys(
                    ["neurosynth", "neurosynth_iit", "neurostore"], {})})
        )

    monkeypatch.setattr(serve, "build_bundle", build)
    serve.prepare_site(tmp_path)
    serve.prepare_site(tmp_path)
    assert built == [None]
    serve.prepare_site(tmp_path, rebuild=True, embeddings="emb")
    assert built == [None, "emb"]


def test_clear_removes_the_bundle_built_before_and_builds_it_again(
    serve, monkeypatch, tmp_path
):
    built = []

    def build(repl, embeddings, engines=None):
        built.append(sorted(path.name for path in repl.iterdir()))
        (repl / "bundle.json").write_text("{}")

    monkeypatch.setattr(serve, "build_bundle", build)
    serve.prepare_site(tmp_path)
    stale = tmp_path / "_static" / "repl" / "nl_core-0.1.dev1-py3-none-any.whl"
    stale.write_bytes(b"wheel")
    serve.prepare_site(tmp_path, clear=True)
    assert built == [[], []]
    assert not stale.exists()
    assert (tmp_path / "_static" / "repl" / "index.html").is_file()
    assert serve.parse_args(["--clear"]).clear


def test_the_bundle_is_built_by_the_bundle_script(
    serve, monkeypatch, tmp_path
):
    commands = []
    monkeypatch.setattr(serve.subprocess, "check_call", commands.append)
    serve.build_bundle(tmp_path, None)
    serve.build_bundle(tmp_path, "emb")
    assert commands[0][1:] == [str(serve.BUNDLE_SCRIPT), str(tmp_path)]
    assert commands[1][-1] == "emb"


def test_the_bundle_is_built_with_the_engines_asked_for(
    serve, monkeypatch, tmp_path
):
    commands = []
    monkeypatch.setattr(serve.subprocess, "check_call", commands.append)
    serve.build_bundle(tmp_path, None, ["neurostore"])
    assert commands[0][-2:] == ["--engine", "neurostore"]
    serve.build_bundle(tmp_path, "emb", ["neurosynth", "neurostore"])
    assert commands[1][-4:] == [
        "--engine", "neurosynth", "--engine", "neurostore"
    ]


def test_a_bundle_lacking_an_engine_asked_for_is_built_again(
    serve, monkeypatch, tmp_path
):
    built = []

    def build(repl, embeddings, engines=None):
        built.append(engines)
        names = engines or [
                "neurosynth", "neurosynth_iit", "neurostore"]
        (repl / "bundle.json").write_text(
            json.dumps({"engines": {name: {} for name in names}})
        )

    monkeypatch.setattr(serve, "build_bundle", build)
    serve.prepare_site(tmp_path, engines=["neurosynth"])
    serve.prepare_site(tmp_path, engines=["neurosynth"])
    assert built == [["neurosynth"]]
    serve.prepare_site(tmp_path)  # by default, all the engines of the page
    assert built == [["neurosynth"], None]
    serve.prepare_site(tmp_path)
    assert len(built) == 2


def test_a_bundle_from_before_the_choice_of_engine_has_neurosynth(
    serve, tmp_path
):
    assert serve.built_engines(tmp_path) == set()
    (tmp_path / "bundle.json").write_text('{"wheels": [], "data": "a.zip"}')
    assert serve.built_engines(tmp_path) == {"neurosynth"}


def test_the_engines_of_the_page_are_those_of_the_bundle_script(serve):
    assert serve.bundle_engines()[0] == "neurosynth"
    assert "neurostore" in serve.bundle_engines()
    assert serve.parse_args(["--engine", "neurostore"]).engines == [
        "neurostore"
    ]


def test_files_are_served_uncached(serve, tmp_path):
    (tmp_path / "bundle.json").write_text(json.dumps({"wheels": []}))
    server = serve.make_server(tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/bundle.json"
        ) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == {"wheels": []}
    finally:
        server.shutdown()
        server.server_close()


def test_scripts_are_served_as_javascript(serve):
    handler = serve.Handler
    assert handler.extensions_map[".js"] == "text/javascript"
    assert handler.extensions_map[".mjs"] == "text/javascript"


def test_the_embeddings_are_kept_in_a_cache_by_default(serve, tmp_path):
    """Made once (with torch) and then reused, as CI keeps them."""
    args = serve.parse_args([])
    assert args.embeddings == serve.DEFAULT_EMBEDDINGS
    assert serve.parse_args(["--embeddings", str(tmp_path)]).embeddings == \
        tmp_path


@pytest.fixture
def gh(serve, monkeypatch):
    """A GitHub CLI answering from its dict ``answers``, by the first words
    of the command, and recording the commands; ``run download`` writes a
    site with the page and its bundle."""
    commands = []

    def run(command, **kwargs):
        commands.append(command[1:])
        if command[1:3] == ["run", "download"]:
            repl = Path(command[command.index("-D") + 1]) / "_static" / "repl"
            repl.mkdir(parents=True)
            (repl / "index.html").write_text("<html>")
            (repl / "bundle.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, "", "")
        for words, answer in run.answers.items():
            if " ".join(command[1:]).startswith(words):
                return subprocess.CompletedProcess(command, 0, answer, "")
        return subprocess.CompletedProcess(command, 1, "", "HTTP 404")

    run.answers, run.commands = {}, commands
    monkeypatch.setattr(serve.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(serve.subprocess, "run", run)
    return run


def test_pr_and_run_id_are_options_that_build_nothing(serve):
    args = serve.parse_args(["--pr", "1120"])
    assert (args.pr, args.run_id, args.repo) == (1120, None, serve.REPOSITORY)
    assert serve.parse_args(["--run-id", "42", "--repo", "a/b"]).repo == "a/b"
    for argv in (["--pr", "1", "--run-id", "2"], ["--pr", "1", "--rebuild"],
                 ["--run-id", "2", "--clear"]):
        with pytest.raises(SystemExit):
            serve.parse_args(argv)


def test_the_site_of_a_pull_request_is_that_of_its_latest_run(
    serve, gh, tmp_path, monkeypatch
):
    gh.answers["api repos/NeuroLang/NeuroLang/actions/artifacts"
               "?name=repl-pr-7"] = "42 abc123\n"
    monkeypatch.setattr(serve, "build_bundle", pytest.fail)
    args = serve.parse_args(["--pr", "7", "--site", str(tmp_path)])
    page = serve.fetch_site(args)
    assert page == tmp_path / "pr-7" / "_static" / "repl" / "index.html"
    assert page.is_file()
    assert args.site == tmp_path / "pr-7"
    assert gh.commands[-1] == [
        "run", "download", "42", "-R", "NeuroLang/NeuroLang",
        "-n", "repl-pr-7", "-D", str(tmp_path / "pr-7"),
    ]
    # Served again, the site of the same run is not downloaded again.
    serve.fetch_site(serve.parse_args(["--pr", "7", "--site", str(tmp_path)]))
    assert [c[:2] for c in gh.commands].count(["run", "download"]) == 1


def test_a_newer_run_replaces_the_site_downloaded_before(
    serve, gh, tmp_path
):
    stale = tmp_path / "pr-7" / "stale.whl"
    stale.parent.mkdir()
    stale.write_bytes(b"old")
    (tmp_path / "pr-7" / serve.RUN_FILE).write_text("41")
    gh.answers["api repos/NeuroLang/NeuroLang/actions/artifacts"] = "42 abc"
    serve.fetch_site(serve.parse_args(["--pr", "7", "--site", str(tmp_path)]))
    assert not stale.exists()
    assert (tmp_path / "pr-7" / serve.RUN_FILE).read_text() == "42"


def test_the_site_of_a_run_is_its_artifact(serve, gh, tmp_path):
    gh.answers["api repos/NeuroLang/NeuroLang/actions/runs/42/artifacts"] = \
        "coverage\nrepl-pr-9\n"
    page = serve.fetch_site(
        serve.parse_args(["--run-id", "42", "--site", str(tmp_path)])
    )
    assert page == tmp_path / "run-42" / "_static" / "repl" / "index.html"
    assert "repl-pr-9" in gh.commands[-1]


def test_a_pull_request_without_a_site_is_told(serve, gh, tmp_path):
    gh.answers["api repos/NeuroLang/NeuroLang/actions/artifacts"] = ""
    with pytest.raises(SystemExit, match="No site of pull request 7"):
        serve.fetch_site(
            serve.parse_args(["--pr", "7", "--site", str(tmp_path)])
        )


def test_a_failing_gh_is_told(serve, gh, tmp_path):
    with pytest.raises(SystemExit, match="HTTP 404"):
        serve.fetch_site(
            serve.parse_args(["--run-id", "1", "--site", str(tmp_path)])
        )


def test_without_gh_the_way_to_get_it_is_told(serve, monkeypatch, tmp_path):
    monkeypatch.setattr(serve.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="cli.github.com"):
        serve.fetch_site(
            serve.parse_args(["--pr", "7", "--site", str(tmp_path)])
        )


def test_main_serves_the_downloaded_site(serve, gh, tmp_path, monkeypatch):
    gh.answers["api repos/NeuroLang/NeuroLang/actions/artifacts"] = "42 abc"
    served = []

    class Server:
        server_address = ("127.0.0.1", 8766)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(
        serve, "make_server", lambda *a: served.append(a) or Server()
    )
    serve.main(["--pr", "7", "--site", str(tmp_path), "--no-browser"])
    assert served[0][0] == tmp_path / "pr-7"


class _FakeBackend:
    """A backend of the page that records what it is asked."""

    def __init__(self):
        import queue

        self.listener = queue.Queue()
        self.asked = []

    def listen(self):
        return self.listener

    def forget(self, listener):
        self.asked.append(("forget",))

    def stop(self):
        return False

    def start(self, engine):
        self.listener.put(json.dumps({"type": "status", "message": "built"}))
        return {"banner": f"<b>{engine}</b>"}

    def submit(self, text):
        if text == "boom":
            raise ValueError("it broke")
        if text == "gone":
            from neurolang_apps.utils.tui.sessions import SessionEnded

            raise SessionEnded("This session ended")
        return {"html": text, "failed": False, "tables": []}

    def set_llm(self, **options):
        self.asked.append(("llm", options))

    def rate(self, verdict, comment=""):
        return {"ok": True, "message": verdict}

    def download(self, block, kind, value=""):
        from neurolang_apps.utils.tui.web import Download

        return Download("résultat 1.csv", "text/csv", b"a,b\n")


def _post(port, action, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/{action}",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request)


def test_the_local_site_has_the_page_answered_by_this_machine(
    serve, tmp_path
):
    page = serve.prepare_local_site(tmp_path, ["neurosynth"])
    bundle = json.loads((page.parent / "bundle.json").read_text())
    assert bundle == {
        "backend": "local", "default": "neurosynth",
        "engines": {"neurosynth": {"label": "Neurosynth"}},
    }
    assert (page.parent / "local_worker.js").is_file()
    page = serve.prepare_local_site(tmp_path, ["neurosynth", "neurosynth_iit"])
    bundle = json.loads((page.parent / "bundle.json").read_text())
    assert bundle["default"] == getattr(
        serve.bundle_script(), "DEFAULT_ENGINE", "neurosynth"
    )
    assert (page.parent / "highlight.json").is_file()
    assert serve.parse_args(["--local"]).local
    with pytest.raises(SystemExit):
        serve.parse_args(["--local", "--rebuild"])


def test_the_local_server_answers_the_worker(serve, tmp_path):
    backend = _FakeBackend()
    server = serve.make_server(tmp_path, "127.0.0.1", 0, backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        events = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/events"
        )
        with _post(port, "start", {"engine": "neurosynth"}) as response:
            assert json.load(response) == {"banner": "<b>neurosynth</b>"}
        assert events.readline() == (
            b'data: {"type": "status", "message": "built"}\n'
        )
        with _post(port, "submit", {"text": "ans(x) :- r(x)"}) as response:
            assert json.load(response)["html"] == "ans(x) :- r(x)"
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(port, "submit", {"text": "boom"})
        with error.value:  # closed, not left to the collector to warn
            assert error.value.code == 500
            assert json.load(error.value) == {"error": "ValueError: it broke"}
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(port, "submit", {"text": "gone"})
        with error.value:  # the page brings its session back
            assert error.value.code == 409
            assert json.load(error.value) == {"error": "This session ended",
                                              "ended": True}
        with _post(port, "download", {"block": 0, "kind": "csv"}) as response:
            assert response.read() == b"a,b\n"
            assert response.headers["Content-Type"] == "text/csv"
            assert urllib.parse.unquote(
                response.headers["X-File-Name"]
            ) == "résultat 1.csv"
        with _post(port, "llm", {"provider": "anthropic", "key": "k"}):
            pass
        assert ("llm", {"provider": "anthropic", "key": "k"}) in backend.asked
        events.close()
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def certificate(tmp_path):
    """A self-signed certificate of localhost and its key, made by openssl."""
    if shutil.which("openssl") is None:
        pytest.skip("needs openssl")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "1",
         "-subj", "/CN=localhost"],
        check=True, capture_output=True,
    )
    return cert, key


def test_https_serves_with_the_certificate_even_with_a_silent_client(
    serve, tmp_path, certificate
):
    cert, key = certificate
    (tmp_path / "bundle.json").write_text(json.dumps({"wheels": []}))
    server = serve.make_server(
        tmp_path, "127.0.0.1", 0, tls=serve.tls_context(cert, key)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    # A client that connects and says nothing does not stop the others.
    silent = socket.create_connection(("127.0.0.1", port))
    try:
        context = ssl.create_default_context(cafile=str(cert))
        context.check_hostname = False
        with urllib.request.urlopen(
            f"https://127.0.0.1:{port}/bundle.json", context=context,
            timeout=10,
        ) as response:
            assert json.load(response) == {"wheels": []}
    finally:
        silent.close()
        server.shutdown()
        server.server_close()


def test_a_key_in_the_certificate_file_serves(serve, tmp_path, certificate):
    cert, key = certificate
    both = tmp_path / "both.pem"
    both.write_text(cert.read_text() + key.read_text())
    assert serve.tls_context(both)


def _locked(key, path, passphrase="secret"):
    """The key *key* with a passphrase, in *path*."""
    subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-out", str(path), "-aes256",
         "-passout", f"pass:{passphrase}"],
        check=True, capture_output=True,
    )
    return path


def test_a_key_with_a_passphrase_serves_with_it(
    serve, tmp_path, certificate, monkeypatch
):
    cert, key = certificate
    locked = _locked(key, tmp_path / "locked.pem")
    secret = tmp_path / "passphrase"
    secret.write_text("secret\n")
    assert serve.tls_context(cert, locked, secret)
    # that of the secrets (key_password)
    assert serve.tls_context(cert, locked, key_password="secret")
    secret.write_text("wrong\n")
    with pytest.raises(SystemExit, match="passphrase .* is not that one"):
        serve.tls_context(cert, locked, secret)
    # Without a file, a terminal is asked; a service has none.
    monkeypatch.setattr(serve.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="key-password-file"):
        serve.tls_context(cert, locked)


def test_a_certificate_that_cannot_serve_says_why(
    serve, tmp_path, certificate
):
    cert, key = certificate
    with pytest.raises(SystemExit, match="has no private key.*--key"):
        serve.tls_context(cert)
    other = tmp_path / "other.pem"
    subprocess.run(["openssl", "genrsa", "-out", str(other), "2048"],
                   check=True, capture_output=True)
    with pytest.raises(SystemExit, match="is not that of the certificate"):
        serve.tls_context(cert, other)
    with pytest.raises(SystemExit, match="No such file"):
        serve.tls_context(tmp_path / "none.pem", key)


def test_key_needs_cert(serve):
    with pytest.raises(SystemExit):
        serve.parse_args(["--key", "key.pem"])


def _daemon_args(serve, tmp_path, *more):
    return serve.parse_args([
        "--site", str(tmp_path / "site"), "--port", "8443",
        "--cert", "cert.pem", "--engine", "neurosynth", *more,
    ])


def test_the_service_serves_as_asked_with_absolute_paths(
    serve, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--local", "--data-dir", "data",
    ))
    assert command[1:4] == ["-m", "serve_repl", "--no-browser"]
    assert "--daemon" not in command
    after = dict(zip(command, command[1:]))
    assert after["--port"] == "8443"
    assert after["--cert"] == str(tmp_path / "cert.pem")
    assert after["--data-dir"] == str(tmp_path / "data")
    assert after["--engine"] == "neurosynth"
    assert "--local" in command


def test_the_service_runs_the_python_of_the_virtual_environment(
    serve, tmp_path, monkeypatch
):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    # The Python the virtual environment links to, without its packages.
    monkeypatch.setattr(serve.sys, "executable", "/usr/bin/python3.12")
    monkeypatch.setattr(serve.sys, "prefix", str(venv))
    monkeypatch.setattr(serve.sys, "base_prefix", "/usr")
    command = serve.service_command(_daemon_args(serve, tmp_path))
    assert command[0] == str(venv / "bin" / "python")


def test_the_service_runs_the_python_asked_for(serve, tmp_path):
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--daemon", "install",
        "--python", "/opt/py/bin/python",
    ))
    assert command[0] == "/opt/py/bin/python"
    assert "--python" not in command


def test_python_is_that_of_the_service(serve):
    with pytest.raises(SystemExit):
        serve.parse_args(["--python", "/opt/py/bin/python"])


def test_the_service_has_the_path_and_the_virtual_environment(
    serve, tmp_path, monkeypatch
):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").touch()
    monkeypatch.setenv("PATH", "/opt/gh/bin:/usr/bin")
    python = str(venv / "bin" / "python")
    unit = serve.systemd_unit([python, "-m", "serve_repl"], tmp_path)
    assert 'Environment="PATH=/opt/gh/bin:/usr/bin"\n' in unit
    assert f'Environment="VIRTUAL_ENV={venv}"\n' in unit
    agent = plistlib.loads(serve.launchd_plist([python], tmp_path))
    assert agent["EnvironmentVariables"] == {
        "PATH": "/opt/gh/bin:/usr/bin", "VIRTUAL_ENV": str(venv),
    }


def test_the_service_builds_nothing_again(serve, tmp_path):
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--rebuild", "--clear", "--daemon", "install",
    ))
    assert not {"--rebuild", "--clear", "--daemon"} & set(command)


def test_the_systemd_unit_runs_the_command(serve, tmp_path):
    unit = serve.systemd_unit(["/py", "-m", "serve_repl", "--site",
                               "/a b/100%"], tmp_path)
    assert 'ExecStart="/py" "-m" "serve_repl" "--site" "/a b/100%%"\n' in unit
    assert f"WorkingDirectory={tmp_path}\n" in unit
    assert "WantedBy=default.target" in unit
    # Restarted after a failure, not as fast as systemd stops restarting.
    assert "Restart=on-failure\nRestartSec=5\n" in unit


def test_the_systemd_unit_runs_any_path_as_it_is(serve, tmp_path, monkeypatch):
    """systemd expands $ and % and unescapes \\ in ExecStart, reads quotes
    only around a whole word, and expands % in WorkingDirectory."""
    monkeypatch.setenv("PATH", "/it's/$bin")
    unit = serve.systemd_unit(
        ["/it's/py", "-m", "serve_repl", "--site", 'a "$HOME" \\n %i'],
        tmp_path / "50%",
    )
    assert ('ExecStart="/it\'s/py" "-m" "serve_repl" "--site" '
            '"a \\"$$HOME\\" \\\\n %%i"\n') in unit
    assert f"WorkingDirectory={tmp_path}/50%%\n" in unit
    assert 'Environment="PATH=/it\'s/$bin"\n' in unit


def test_the_launchd_agent_runs_the_command(serve, tmp_path):
    agent = plistlib.loads(serve.launchd_plist(["/py", "-m", "x"], tmp_path))
    assert agent["Label"] == serve.LAUNCHD_LABEL
    assert agent["ProgramArguments"] == ["/py", "-m", "x"]
    assert agent["RunAtLoad"] and agent["WorkingDirectory"] == str(tmp_path)


@pytest.fixture(params=["linux", "darwin"])
def manager(serve, tmp_path, monkeypatch, request):
    """The service manager of a platform, in a HOME of the test, recording
    the verbs it runs (``managed``), the checks of the service's Python
    (``checked``, failing if ``python``) and the builds (``built``); the
    verbs of ``fails`` fail as many times as they say."""
    state = SimpleNamespace(
        platform=request.param, managed=[], checked=[], built=[], fails={},
        python=0, slept=[],
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(serve.sys, "platform", request.param)
    monkeypatch.setattr(serve.shutil, "which", lambda name: name)
    monkeypatch.setattr(serve.time, "sleep", state.slept.append)

    def run(command, check=True, **kwargs):
        if command[1] == "-c":  # the check of the service's Python
            state.checked.append(command[2])
            return subprocess.CompletedProcess(
                command, state.python, "", "ModuleNotFoundError: no rich")
        verb = command[2] if command[1] == "--user" else command[1]
        state.managed.append(verb)
        if state.fails.get(verb) and check:
            state.fails[verb] -= 1
            raise subprocess.CalledProcessError(5, command)

    monkeypatch.setattr(serve.subprocess, "run", run)
    monkeypatch.setattr(serve, "prepare_site",
                        lambda *args, **kwargs: state.built.append(args))
    monkeypatch.setattr(serve, "tls_context", lambda *args, **kw: object())
    return state


def test_the_service_is_installed_reinstalled_and_removed(
    serve, tmp_path, manager
):
    path = serve.service_file()
    assert str(path).startswith(str(tmp_path))

    serve.main(["--daemon", "install", "--site", str(tmp_path / "site"),
                "--cert", "cert.pem"])
    assert path.is_file() and len(manager.built) == 1
    assert "--cert" in path.read_bytes().decode()
    with pytest.raises(SystemExit, match="reinstall"):
        serve.main(["--daemon", "install"])
    serve.main(["--daemon", "reinstall", "--site", str(tmp_path / "site")])
    assert path.is_file() and len(manager.built) == 2
    assert "--cert" not in path.read_bytes().decode()
    serve.main(["--daemon", "remove"])
    assert not path.exists()
    serve.main(["--daemon", "remove"])  # nothing installed: nothing done
    assert manager.checked == ["import serve_repl"] * 2
    if manager.platform == "linux":
        assert manager.managed == [
            "daemon-reload", "enable", "disable", "daemon-reload",
            "daemon-reload", "enable", "disable", "daemon-reload"]
    else:
        assert manager.managed == [
            "bootstrap", "bootout", "bootstrap", "bootout"]


def test_a_reinstall_that_cannot_serve_keeps_the_service_installed(
    serve, tmp_path, manager, monkeypatch
):
    serve.main(["--daemon", "install", "--site", str(tmp_path / "site")])
    path, managed = serve.service_file(), list(manager.managed)
    installed = path.read_bytes()
    manager.python = 1  # the Python of the service cannot serve
    with pytest.raises(SystemExit, match="cannot serve"):
        serve.main(["--daemon", "reinstall", "--local"])
    manager.python = 0

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["make_repl_bundle.py"])

    monkeypatch.setattr(serve, "prepare_site", fail)  # the build fails
    with pytest.raises(subprocess.CalledProcessError):
        serve.main(["--daemon", "reinstall", "--site", str(tmp_path / "s")])
    assert path.read_bytes() == installed
    assert manager.managed == managed  # not stopped


def test_a_failing_service_manager_leaves_no_service(serve, manager):
    manager.fails = {"bootstrap": 99, "enable": 99}
    with pytest.raises(SystemExit, match="failed .*not installed"):
        serve.main(["--daemon", "install", "--local"])
    assert not serve.service_file().exists()
    if manager.platform == "linux":
        assert manager.managed == [
            "daemon-reload", "enable", "disable", "daemon-reload"]
    else:  # tried for 5 seconds
        assert manager.managed == ["bootstrap"] * 10 + ["bootout"]
        assert sum(manager.slept) == pytest.approx(4.5)
    manager.fails = {}
    serve.main(["--daemon", "install", "--local"])  # it can be installed


def test_a_bootstrap_waits_for_the_bootout_of_a_reinstall(serve, manager):
    manager.fails = {"bootstrap": 2}
    serve.main(["--daemon", "install", "--local"])
    assert serve.service_file().is_file()
    if manager.platform == "darwin":
        assert manager.managed == ["bootstrap"] * 3
        assert manager.slept == [0.5, 0.5]


def test_a_machine_without_service_manager_is_told_before_anything(
    serve, manager, monkeypatch
):
    monkeypatch.setattr(serve.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="this machine lacks"):
        serve.main(["--daemon", "install", "--local"])
    assert not serve.service_file().exists()
    assert manager.checked == manager.managed == []


def test_remove_needs_no_other_option(serve, manager, monkeypatch, tmp_path):
    serve.main(["--daemon", "install", "--local"])
    monkeypatch.setattr(serve, "tls_context", pytest.fail)
    serve.main(["--daemon", "remove", "--cert", "none.pem",
                "--password-file", str(tmp_path / "none")])
    assert not serve.service_file().exists()


def test_a_service_python_lacking_the_extras_is_told(
    serve, tmp_path, monkeypatch
):
    monkeypatch.setattr(serve.subprocess, "run", lambda command, **kw: (
        subprocess.CompletedProcess(
            command, 1, "", "ModuleNotFoundError: No module named 'rich'")))
    with pytest.raises(SystemExit, match="--extra tui") as stop:
        serve.check_service_python(["/venv/bin/python", "-m", "x"], True)
    assert "No module named 'rich'" in str(stop.value)


def _basic(password, user="anyone"):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _status(port, path, headers=None, method="GET", body=None):
    headers = dict(headers or {})
    if body is not None:  # as the page posts
        headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if path == "/api/events":  # a stream: its first event will do
                response.readline()
            else:
                response.read()
            return response.status
    except urllib.error.HTTPError as error:
        with error:
            return error.code


@pytest.mark.parametrize("password", ["s3cret", "mot de passe é"])
def test_every_request_needs_the_password(serve, tmp_path, password):
    (tmp_path / "bundle.json").write_text("{}")
    backend = _FakeBackend()
    backend.listener.put(json.dumps({"type": "status", "message": "hi"}))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, backend,
                               password=password)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    requests = [
        ("/bundle.json", "GET", None), ("/bundle.json", "HEAD", None),
        ("/api/start", "POST", {"engine": "neurosynth"}),
        ("/api/events", "GET", None),
    ]
    try:
        for path, method, body in requests:
            for wrong in ({}, _basic("wrong"), {"Authorization": "Basic !!"},
                          {"Authorization": "Bearer x"}):
                assert _status(port, path, wrong, method, body) == 401, path
            assert _status(port, path, _basic(password), method, body) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_the_main_page_leads_to_the_repl_unless_the_site_has_one(
    serve, tmp_path
):
    page = serve.prepare_local_site(tmp_path, ["neurosynth"])
    serve.write_index(tmp_path, page)
    assert 'url=_static/repl/index.html"' in (
        tmp_path / "index.html").read_text()
    (tmp_path / "index.html").write_text("the documentation")
    serve.write_index(tmp_path, page)
    assert (tmp_path / "index.html").read_text() == "the documentation"


def test_a_password_in_clear_and_a_key_for_all_are_refused(serve):
    with pytest.raises(SystemExit):  # over the network without TLS
        serve.parse_args(["--host", "0.0.0.0", "--password-file", "p"])
    assert serve.parse_args(["--password-file", "p"]).password_file
    with pytest.raises(SystemExit):  # a key without a password
        serve.parse_args(["--local", "--env-file", "e"])
    with pytest.raises(SystemExit):  # a key in the browser's page
        serve.parse_args(["--password-file", "p", "--env-file", "e"])


def test_the_env_file_gives_the_assistant_its_key(
    serve, tmp_path, monkeypatch
):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    env = tmp_path / "env"
    env.write_text(f"# the key\nexport TEST_LLM_KEY='k-1'\n"
                   f"NEUROLANG_LLM_CONFIG={config}\n")
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    monkeypatch.delenv("NEUROLANG_LLM_CONFIG", raising=False)
    assert serve.read_env_file(env) == {
        "TEST_LLM_KEY": "k-1", "NEUROLANG_LLM_CONFIG": str(config),
    }
    monkeypatch.setattr(serve.os, "environ", {
        **serve.os.environ, **serve.read_env_file(env)})
    assert serve.default_assistant() == "test (m)"
    monkeypatch.setattr(serve.os, "environ", {
        "NEUROLANG_LLM_CONFIG": str(config)})
    with pytest.raises(SystemExit, match="TEST_LLM_KEY"):
        serve.default_assistant()


def test_the_secrets_stay_in_their_files(serve, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "password").write_text("s3cret\n")
    (tmp_path / "env").write_text("TEST_LLM_KEY=k-1\n")
    command = serve.service_command(serve.parse_args([
        "--local", "--password-file", "password", "--env-file", "env",
        "--daemon", "install",
    ]))
    after = dict(zip(command, command[1:]))
    assert after["--password-file"] == str(tmp_path / "password")
    assert after["--env-file"] == str(tmp_path / "env")
    unit = serve.systemd_unit(command, tmp_path)
    assert "s3cret" not in unit and "k-1" not in unit


@contextmanager
def _serving(server):
    """The port of *server*, serving in a thread until the end."""
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _raw(port, request: bytes) -> tuple:
    """Send *request* as it is; return the status and the body answered."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        answer = b""
        while chunk := sock.recv(65536):
            answer += chunk
    head, _, body = answer.partition(b"\r\n\r\n")
    return int(head.split()[1]), body


def test_a_silent_connection_is_closed(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(serve.Handler, "timeout", 0.5)
    with _serving(serve.make_server(tmp_path, "127.0.0.1", 0)) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            assert s.recv(1) == b""  # closed by the server


def test_a_silent_or_plain_http_client_of_https_is_closed_and_logged(
    serve, tmp_path, certificate, monkeypatch, capsys
):
    monkeypatch.setattr(serve.Handler, "timeout", 0.5)
    tls = serve.tls_context(*certificate)
    with _serving(serve.make_server(tmp_path, "127.0.0.1", 0, tls=tls)) \
            as port:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            assert s.recv(1) == b""  # its handshake timed out
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            try:
                s.recv(1024)
            except ConnectionError:  # reset by the server
                pass
        time.sleep(0.2)
    err = capsys.readouterr().err
    assert "SSLError" in err and "Traceback" not in err


def test_the_stream_of_events_outlives_the_timeout(
    serve, tmp_path, monkeypatch
):
    monkeypatch.setattr(serve.Handler, "timeout", 0.3)
    monkeypatch.setattr(serve.LocalHandler, "KEEPALIVE_SECONDS", 0.1)
    server = serve.make_server(tmp_path, "127.0.0.1", 0, _FakeBackend())
    with _serving(server) as port:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/events", timeout=10
        ) as events:
            start = time.monotonic()
            while time.monotonic() - start < 1.0:
                assert events.readline() in (b": alive\n", b"\n")


@pytest.mark.parametrize("headers, body, status, error", [
    ("Content-Length: abc", b"", 400, "is not a length"),
    ("Content-Length: -1", b"", 400, "is not a length"),
    ("Content-Length: 2000000", b"", 413, "over 1048576 bytes"),
    ("Content-Length: 10", b"{}", 400, "shorter than its Content-Length"),
    ("Content-Length: 3", b"[1]", 400, "not a JSON object"),
    ("Content-Length: 4", b"{bad", 400, "Expecting property name"),
    ("Transfer-Encoding: chunked", b"0\r\n\r\n", 400, "Content-Length"),
    ("Content-Length: 1000000", b"[" * 500000 + b"]" * 500000, 400,
     "nests too deep"),
])
def test_a_body_that_is_not_a_json_object_is_refused(
    serve, tmp_path, monkeypatch, capsys, headers, body, status, error
):
    monkeypatch.setattr(serve.Handler, "timeout", 0.5)
    server = serve.make_server(tmp_path, "127.0.0.1", 0, _FakeBackend())
    with _serving(server) as port:
        answered, text = _raw(port, (
            "POST /api/submit HTTP/1.1\r\nHost: localhost\r\n"
            "Content-Type: application/json\r\n"
            f"Connection: close\r\n{headers}\r\n\r\n"
        ).encode() + body)
        assert answered == status
        assert error in json.loads(text)["error"]
        # A request without a body has an empty one.
        assert _raw(port, b"POST /api/stop HTTP/1.0\r\nHost: localhost\r\n"
                    b"Content-Type: application/json\r\n\r\n"
                    ) == (200, b'{"stopped": false}')
    assert "Traceback" not in capsys.readouterr().err


def test_without_password_only_the_names_of_the_server_are_answered(
    serve, tmp_path
):
    """A page of another site, whose name it made lead to 127.0.0.1 (DNS
    rebinding), sends its own name as the host."""
    (tmp_path / "bundle.json").write_text("{}")
    server = serve.make_server(tmp_path, "127.0.0.1", 0, _FakeBackend(),
                               names=["Repl.Lab.org"])
    with _serving(server) as port:
        for host in ("localhost", f"localhost:{port}", f"127.0.0.1:{port}",
                     f"[::1]:{port}", "repl.lab.org"):
            assert _raw(port, f"GET /bundle.json HTTP/1.0\r\nHost: {host}"
                        "\r\n\r\n".encode())[0] == 200, host
        for host in ("evil.example", "evil.example:80", "[::1", ""):
            status, body = _raw(port, (
                "POST /api/submit HTTP/1.0\r\nContent-Length: 2\r\n"
                f"Host: {host}\r\n\r\n{{}}").encode())
            assert status == 403, host
        assert b"--server-name evil.example" in _raw(
            port, b"GET / HTTP/1.0\r\nHost: evil.example\r\n\r\n")[1]
        assert _raw(port, b"GET /bundle.json HTTP/1.0\r\n\r\n")[0] == 403
    # With a password, the host is not looked at.
    server = serve.make_server(tmp_path, "127.0.0.1", 0, password="pw")
    with _serving(server) as port:
        token = base64.b64encode(b"a:pw").decode()
        assert _raw(port, (
            "GET /bundle.json HTTP/1.0\r\nHost: evil.example\r\n"
            f"Authorization: Basic {token}\r\n\r\n").encode())[0] == 200


def test_the_service_answers_the_server_names(serve, tmp_path):
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--host", "0.0.0.0", "--server-name", "a.org",
        "--server-name", "b.org",
    ))
    assert " ".join(command).count("--server-name") == 2
    assert command[command.index("--server-name") + 1] == "a.org"


def test_a_passphrase_with_spaces_is_kept_as_it_is(
    serve, tmp_path, certificate
):
    cert, key = certificate
    locked = _locked(key, tmp_path / "locked.pem", " sec ret ")
    secret = tmp_path / "passphrase"
    secret.write_text(" sec ret \r\nnext line\n")
    assert serve.read_password(secret) == " sec ret "
    assert serve.tls_context(cert, locked, secret)


def test_a_missing_file_is_told(serve, tmp_path, certificate):
    missing = tmp_path / "none"
    with pytest.raises(SystemExit, match="none cannot be read: No such file"):
        serve.read_password(missing)
    with pytest.raises(SystemExit, match="none cannot be read"):
        serve.read_env_file(missing)
    cert, key = certificate
    with pytest.raises(SystemExit, match="none cannot be read"):
        serve.tls_context(cert, _locked(key, tmp_path / "locked.pem"),
                          missing)


def test_a_comment_after_a_value_is_left_out(serve, tmp_path):
    env = tmp_path / "env"
    env.write_text("A=1 # the first\nB='x # y' # z\nC=a#b\nD=\"q\"\n")
    assert serve.read_env_file(env) == {
        "A": "1", "B": "x # y", "C": "a#b", "D": "q",
    }


def test_the_python_of_the_service_may_be_a_name_in_the_path(
    serve, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serve.shutil, "which", lambda name: f"/opt/{name}")
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--daemon", "install", "--python", "python3"))
    assert command[0] == "/opt/python3"
    command = serve.service_command(_daemon_args(
        serve, tmp_path, "--daemon", "install", "--python", "./bin/py"))
    assert command[0] == str(tmp_path / "bin" / "py")
    monkeypatch.setattr(serve.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="python3: no such program"):
        serve.service_command(_daemon_args(
            serve, tmp_path, "--daemon", "install", "--python", "python3"))


@pytest.mark.parametrize("password", [None, "pw"])
def test_a_post_of_another_site_is_refused(serve, tmp_path, password):
    """A form of another site can POST text/plain whose body is JSON; a
    script of another site sends its Origin."""
    backend = _FakeBackend()
    server = serve.make_server(tmp_path, "127.0.0.1", 0, backend,
                               password=password)
    login = _basic(password) if password else {}
    form = b'{"text":"evil","x":"="}'
    with _serving(server) as port:
        page = f"http://127.0.0.1:{port}"
        for headers in (
            {"Content-Type": "text/plain"},
            {"Content-Type": "application/x-www-form-urlencoded"},
            {"Content-Type": "multipart/form-data; boundary=x"},
            {"Origin": "https://evil.example"},
            {"Origin": "null"},
        ):
            request = urllib.request.Request(
                f"{page}/api/submit", data=form, method="POST",
                headers={"Content-Type": "application/json", **login,
                         **headers},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=10)
            error.value.close()
            assert error.value.code == 403, headers
        assert backend.asked == []
        # The page's own requests: JSON, from its origin or none said.
        for origin in ({}, {"Origin": page}):
            request = urllib.request.Request(
                f"{page}/api/submit", data=b'{"text":"ok"}', method="POST",
                headers={"Content-Type": "application/json; charset=utf-8",
                         **login, **origin},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert json.load(response)["html"] == "ok"


def test_a_password_hash_checks_its_password_only(serve):
    stored = serve.hash_password("s3cret é")
    assert stored.startswith("scrypt$") and "s3cret" not in stored
    users = {"ann": {"password": stored}}
    assert serve.check_password(users, "ann", "s3cret é".encode())
    assert serve.check_password(users, "ann", "s3cret é".encode())  # cached
    assert not serve.check_password(users, "ann", b"wrong")
    assert not serve.check_password(users, "bob", "s3cret é".encode())
    assert not serve._verify("scrypt$bad", b"x")


def test_the_users_file_is_read_and_checked(serve, tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(
        f"ann:\n  password: {serve.hash_password('a')}\n"
        "  env:\n    TEST_LLM_KEY: k-1\n"
        f"bob:\n  password: {serve.hash_password('b')}\n"
    )
    users = serve.read_users(path)
    assert users["ann"]["env"] == {"TEST_LLM_KEY": "k-1"}
    assert users["bob"]["env"] == {}
    path.write_text("ann:\n  password: a\n")  # not a hash
    with pytest.raises(SystemExit, match="add-user"):
        serve.read_users(path)


def _users(serve, tmp_path, **users):
    """The users file of *users*, name=(password, groups), as the server
    reads it."""
    path = tmp_path / "users.yaml"
    path.write_text("".join(
        f"{name}:\n  password: {serve.hash_password(password)}\n"
        f"  groups: [{', '.join(groups)}]\n"
        for name, (password, groups) in users.items()
    ))
    return serve.Users(path)


class _FakeSessions:
    """Sessions that record the user and the tab of each request."""

    def __init__(self):
        self.asked = []

    def session(self, user, tab):
        self.asked.append((user, tab))
        return _FakeBackend()


def test_each_user_and_tab_has_its_session(serve, tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    sessions = _FakeSessions()
    users = _users(serve, tmp_path, ann=("annannann", []),
                   bob=("bobbobbob", []))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, sessions,
                               users=users)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        start = ("/api/start?session=t1", "POST", {"engine": "neurosynth"})
        assert _status(port, *start[:1], _basic("ann", "ann"), *start[1:]) \
            == 401
        assert _status(port, start[0], _basic("annannann", "ann"),
                       *start[1:]) == 200
        assert _status(port, "/api/submit?session=t2", _basic("bobbobbob",
                       "bob"), "POST", {"text": "x"}) == 200
        assert _status(port, "/api/submit", _basic("bobbobbob", "bob"),
                       "POST", {"text": "x"}) == 200
        assert _status(port, "/bundle.json", _basic("annannann", "bob")) \
            == 401
    finally:
        server.shutdown()
        server.server_close()
    assert sessions.asked == [("ann", "t1"), ("bob", "t2"),
                              ("bob", "default")]


def test_users_options(serve, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        serve.parse_args(["--users", "u", "--password-file", "p"])
    with pytest.raises(SystemExit):  # in clear over the network
        serve.parse_args(["--users", "u", "--host", "0.0.0.0"])
    assert serve.parse_args(["--local", "--users", "u", "--env-file", "e"])
    monkeypatch.chdir(tmp_path)
    command = serve.service_command(serve.parse_args([
        "--local", "--users", "users.yaml", "--max-sessions", "3",
        "--idle-minutes", "30", "--max-running", "2", "--daemon", "install",
    ]))
    after = dict(zip(command, command[1:]))
    assert (after["--max-running"], after["--query-minutes"]) == ("2",
                                                                  "60.0")
    assert after["--users"] == str(tmp_path / "users.yaml")
    assert (after["--max-sessions"], after["--idle-minutes"]) == ("3", "30.0")


def test_hash_password_prints_a_hash(serve, monkeypatch, capsys):
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "s3cret")
    serve.main(["--hash-password"])
    stored = capsys.readouterr().out.strip()
    assert serve._verify(stored, b"s3cret")


def test_an_unknown_user_is_checked_as_long_as_a_user(serve, monkeypatch):
    users = {"ann": {"password": serve.hash_password("a")}}
    verified = []
    verify = serve._verify
    monkeypatch.setattr(serve, "_verify", lambda stored, password: (
        verified.append(stored), verify(stored, password))[1])
    assert not serve.check_password(users, "nobody", b"a")
    assert not serve.check_password(users, "nobody", b"a")
    assert len(verified) == 2 and verified[0].startswith("scrypt$")
    assert verified[0] != users["ann"]["password"]


def test_the_sessions_options_are_checked(serve):
    for options in (["--max-sessions", "0"], ["--idle-minutes", "0"],
                    ["--idle-minutes", "-1"], ["--max-running", "0"],
                    ["--query-minutes", "-1"]):
        with pytest.raises(SystemExit):
            serve.parse_args(["--local", "--users", "u", *options])


def test_a_user_whose_variables_give_no_key_is_told(
    serve, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    monkeypatch.setenv("NEUROLANG_LLM_CONFIG", str(config))
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    users = _users(serve, tmp_path, ann=("a", ["admin"]), bob=("b", []),
                   carl=("c", []))
    users.change("ann", env={"TEST_LLM_KEY": "k"})
    users.change("bob", env={"MY_VAR": "x"})
    assistants = serve.users_assistants(users)
    assert assistants == {"ann": "test (m)", "bob": None, "carl": None}
    assert "The variables of bob" in capsys.readouterr().err


@pytest.fixture
def console(serve, tmp_path):
    """A server whose users are the admin ann and bob, with the console."""
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]),
                   bob=("bob-pw", ["lab"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], users
    server.shutdown()
    server.server_close()


def _json_of(port, path, user, password, body=None):
    headers = _basic(password, user)
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers=headers,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def test_the_console_is_the_admins(console):
    port, _ = console
    for path in ("/admin", "/admin/api/state"):
        assert _status(port, path) == 401
        assert _status(port, path, _basic("bob-pw", "bob")) == 403
        assert _status(port, path, _basic("ann-pw", "ann")) == 200
    assert _status(port, "/admin/api/user", _basic("bob-pw", "bob"), "POST",
                   {"name": "eve", "password": "x", "create": True}) == 403
    state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
    assert state["me"] == "ann"
    assert [(u["name"], u["groups"]) for u in state["users"]] == [
        ("ann", ["admin"]), ("bob", ["lab"])]
    assert "password" not in json.dumps(state)


def test_another_sites_post_is_refused(console):
    port, _ = console
    form = {**_basic("ann-pw", "ann"), "Content-Type": "text/plain"}
    body = {"name": "eve", "password": "eve-password", "create": True}
    assert _status(port, "/admin/api/user", form, "POST", body) == 403
    elsewhere = {**_basic("ann-pw", "ann"), "Origin": "https://evil.test"}
    assert _status(port, "/admin/api/user", elsewhere, "POST", body) == 403
    here = {**_basic("ann-pw", "ann"), "Origin": f"http://127.0.0.1:{port}"}
    assert _status(port, "/admin/api/user", here, "POST", body) == 200


def test_the_console_adds_changes_and_removes_users(console):
    port, users = console
    _json_of(port, "/admin/api/user", "ann", "ann-pw",
             {"name": "eve", "password": "eve-password", "groups": ["lab"],
              "create": True})
    assert _status(port, "/bundle.json", _basic("eve-password", "eve")) == 200
    assert oct(users.path.stat().st_mode & 0o777) == "0o600"
    _json_of(port, "/admin/api/user", "ann", "ann-pw",
             {"name": "eve", "env": {"TEST_LLM_KEY": "k-9"}})
    assert users.identity("eve") == ({"TEST_LLM_KEY": "k-9"}, "lab")
    state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
    eve = next(u for u in state["users"] if u["name"] == "eve")
    assert eve["variables"] == ["TEST_LLM_KEY"]
    assert "k-9" not in json.dumps(state)
    _json_of(port, "/admin/api/remove-user", "ann", "ann-pw",
             {"name": "eve"})
    assert _status(port, "/bundle.json", _basic("eve-password", "eve")) == 401


def test_the_users_keep_an_admin(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("a", ["admin"]), bob=("b", []))
    with pytest.raises(ValueError, match="admin"):
        users.remove("ann")
    with pytest.raises(ValueError, match="admin"):
        users.change("ann", groups=["lab"])
    users.change("bob", groups=["admin"])
    users.remove("ann")
    assert users.is_admin("bob") and not users.check("ann", b"a")


def test_an_edit_by_hand_is_read_again(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("a", ["admin"]))
    assert not users.check("bob", b"b")
    text = users.path.read_text() + (
        f"bob:\n  password: {serve.hash_password('b')}\n")
    users.path.write_text(text)
    os.utime(users.path, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert users.check("bob", b"b")


def test_the_key_of_a_user_is_their_own_the_servers_or_none(
    serve, tmp_path, monkeypatch
):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    monkeypatch.setenv("NEUROLANG_LLM_CONFIG", str(config))
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    own = serve.key_status({"env": {"TEST_LLM_KEY": "k"}})
    assert own["assistant"] == "own" and own["variable"] == "TEST_LLM_KEY"
    assert "k" not in own.values()
    assert serve.key_status({"env": {}})["assistant"] == "none"
    monkeypatch.setenv("TEST_LLM_KEY", "server's")
    assert serve.key_status({"env": {}})["assistant"] == "server"


def test_the_users_are_managed_from_the_command_line(
    serve, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "users.yaml"
    answers = []
    monkeypatch.setattr(serve.getpass, "getpass",
                        lambda prompt: answers.pop(0))
    # The first admin makes the file.
    answers[:] = ["ann-password", "ann-password"]
    serve.main(["--users", str(path), "--add-user", "ann",
                "--groups", "admin,lab"])
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    answers[:] = ["bob-password", "bob-password"]
    serve.main(["--users", str(path), "--add-user", "bob"])
    answers[:] = ["k-1"]
    serve.main(["--users", str(path), "--set-key", "bob", "TEST_LLM_KEY"])
    serve.main(["--users", str(path), "--set-groups", "bob",
                "--groups", "lab"])
    answers[:] = ["new-password", "new-password"]
    serve.main(["--users", str(path), "--set-password", "bob"])
    users = serve.Users(path)
    assert users.check("ann", b"ann-password") and users.is_admin("ann")
    assert users.check("bob", b"new-password")
    assert not users.check("bob", b"bob-password")
    assert users.identity("bob") == ({"TEST_LLM_KEY": "k-1"}, "lab")
    capsys.readouterr()
    serve.main(["--users", str(path), "--list-users"])
    listed = capsys.readouterr().out
    assert "ann\tgroups: admin, lab" in listed and "bob\tgroups: lab" in listed
    assert "k-1" not in listed and "scrypt" not in listed
    answers[:] = [""]  # an empty key removes it
    serve.main(["--users", str(path), "--set-key", "bob", "TEST_LLM_KEY"])
    assert serve.Users(path).identity("bob") == ({}, "lab")
    with pytest.raises(SystemExit, match="admin"):
        serve.main(["--users", str(path), "--remove-user", "ann"])
    serve.main(["--users", str(path), "--remove-user", "bob"])
    assert list(serve.Users(path).current()) == ["ann"]
    answers[:] = ["a", "b"]
    with pytest.raises(SystemExit, match="differ"):
        serve.main(["--users", str(path), "--add-user", "eve"])


def test_a_user_changes_their_own_password(console):
    port, users = console
    bob = _basic("bob-pw", "bob")
    assert _status(port, "/account", bob) == 200
    me = _json_of(port, "/account/api/me", "bob", "bob-pw")
    assert (me["name"], me["groups"], me["admin"]) == ("bob", ["lab"], False)
    assert me["assistant"]["own_key"]
    change = "/account/api/password"
    assert _status(port, change, bob, "POST",
                   {"current": "wrong", "new": "long-enough"}) == 403
    assert _status(port, change, bob, "POST",
                   {"current": "bob-pw", "new": "short"}) == 400
    cross = {**bob, "Content-Type": "text/plain"}
    assert _status(port, change, cross, "POST",
                   {"current": "bob-pw", "new": "long-enough"}) == 403
    assert _status(port, change, bob, "POST",
                   {"current": "bob-pw", "new": "long-enough"}) == 200
    assert _status(port, "/bundle.json", bob) == 401  # the old one is out
    assert _status(port, "/bundle.json",
                   _basic("long-enough", "bob")) == 200
    # Their own password only: groups, keys and others are the admins'.
    assert users.groups("bob") == ["lab"] and users.check("ann", b"ann-pw")
    assert _status(port, "/admin/api/user", _basic("long-enough", "bob"),
                   "POST", {"name": "bob", "groups": ["admin"]}) == 403


def test_the_local_site_links_the_account_page_of_its_users(serve, tmp_path):
    page = serve.prepare_local_site(tmp_path, ["neurosynth"], users=True)
    assert json.loads((page.parent / "bundle.json").read_text())["users"]
    page = serve.prepare_local_site(tmp_path, ["neurosynth"])
    assert "users" not in json.loads((page.parent / "bundle.json").read_text())


def test_without_users_there_is_no_account_page(serve, tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    server = serve.make_server(tmp_path, "127.0.0.1", 0, password="pw")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        assert _status(port, "/account", _basic("pw")) == 404
        assert _status(port, "/bundle.json", _basic("pw")) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_a_group_gives_its_key_to_its_users(serve, tmp_path, monkeypatch):
    users = _users(serve, tmp_path, ann=("a", ["admin", "lab"]),
                   bob=("b", ["lab", "zoo"]), eve=("e", []))
    users.change_group("lab", {"TEST_LLM_KEY": "lab-key"})
    users.change_group("zoo", {"TEST_LLM_KEY": "zoo-key", "OTHER": "o"})
    users.change("ann", env={"TEST_LLM_KEY": "ann-key"})
    # Her own over her group's; the first group by name over the others.
    assert users.identity("ann") == ({"TEST_LLM_KEY": "ann-key"},
                                     "admin,lab")
    assert users.env("bob") == {"TEST_LLM_KEY": "lab-key", "OTHER": "o"}
    assert users.env("eve") == {}
    # The file keeps them, the users in a section of their own.
    text = users.path.read_text()
    assert text.startswith("groups:") and "\nusers:" in text
    assert serve.Users(users.path).env("bob")["TEST_LLM_KEY"] == "lab-key"
    users.change_group("zoo", {"TEST_LLM_KEY": None, "OTHER": None})
    assert "zoo" not in users.group_envs()
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    monkeypatch.setenv("NEUROLANG_LLM_CONFIG", str(config))
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    status = serve.key_status(users.current()["bob"], users.group_envs())
    assert (status["assistant"], status["group"]) == ("group", "lab")
    assert "lab-key" not in json.dumps(status)


def test_a_file_of_the_users_alone_is_read(serve, tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(f"ann:\n  password: {serve.hash_password('a')}\n"
                    "  groups: [admin]\n")
    users = serve.Users(path)
    assert users.check("ann", b"a") and users.group_envs() == {}
    users.change_group("lab", {"TEST_LLM_KEY": "k"})
    assert serve.Users(path).group_envs() == {"lab": {"TEST_LLM_KEY": "k"}}


def test_the_console_and_the_command_line_set_a_groups_key(
    serve, console, monkeypatch
):
    port, users = console
    _json_of(port, "/admin/api/group", "ann", "ann-pw",
             {"name": "lab", "env": {"TEST_LLM_KEY": "lab-key"}})
    assert users.env("bob") == {"TEST_LLM_KEY": "lab-key"}
    state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
    lab = next(g for g in state["groups"] if g["name"] == "lab")
    assert lab["members"] == ["bob"] and lab["variables"] == ["TEST_LLM_KEY"]
    assert "lab-key" not in json.dumps(state)
    assert _status(port, "/admin/api/group", _basic("bob-pw", "bob"), "POST",
                   {"name": "lab", "env": {"TEST_LLM_KEY": "x"}}) == 403
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "")
    serve.main(["--users", str(users.path), "--set-group-key", "lab",
                "TEST_LLM_KEY"])
    assert users.env("bob") == {}


def test_the_assistant_of_a_user_their_groups_and_their_own_key(
    serve, tmp_path
):
    users = _users(serve, tmp_path, ann=("a", ["admin", "lab"]),
                   bob=("b", ["lab"]), eve=("e", ["guest"]))
    users.change_group("lab", llm={"provider": "openrouter",
                                   "model": "m-lab", "key": "lab-key"},
                       own_key=False)
    users.change_group("guest", own_key=True)
    # The group's assistant; the group forbids its users their own key.
    assert users.assistant("bob") == {"provider": "openrouter",
                                      "model": "m-lab", "key": "lab-key",
                                      "source": "group:lab"}
    assert not users.own_key("bob") and users.own_key("eve")
    env, _ = users.identity("bob")
    assert json.loads(env[serve.ASSISTANT_ENV])["key"] == "lab-key"
    assert env[serve.OWN_KEY_ENV] == "0"
    # Her own, and her own setting, over her group's.
    users.change("ann", llm={"provider": "anthropic", "key": "ann-key"},
                 own_key=True)
    assert users.assistant("ann")["source"] == "own" and users.own_key("ann")
    assert serve.OWN_KEY_ENV not in users.identity("ann")[0]
    # The key is kept when the provider is given again without it.
    users.change_group("lab", llm={"provider": "openrouter", "model": "m2"})
    assert users.assistant("bob")["key"] == "lab-key"
    with pytest.raises(ValueError, match="needs its key"):
        users.change("eve", llm={"provider": "openai"})
    with pytest.raises(ValueError, match="No provider"):
        users.change("eve", llm={"provider": "nobody", "key": "k"})
    status = users.status("bob")
    assert (status["assistant"], status["group"]) == ("group", "lab")
    assert "lab-key" not in json.dumps(status)
    assert "lab-key" not in json.dumps(users.group_status("lab"))
    users.change("ann", own_key="inherit")
    assert not users.own_key("ann")  # as her group lab says


def test_a_user_saves_their_assistant_unless_forbidden(console):
    port, users = console
    bob = _basic("bob-pw", "bob")
    llm = {"provider": "anthropic", "model": "m", "key": "bob-key"}
    assert _status(port, "/account/api/llm", bob, "POST", llm) == 200
    assert users.assistant("bob")["key"] == "bob-key"
    me = _json_of(port, "/account/api/me", "bob", "bob-pw")
    assert me["assistant"]["assistant"] == "own"
    assert "bob-key" not in json.dumps(me)
    assert _status(port, "/account/api/llm", bob, "POST",
                   {"remove": True}) == 200
    assert users.assistant("bob") is None
    _json_of(port, "/admin/api/group", "ann", "ann-pw",
             {"name": "lab", "own_key": False,
              "llm": {"provider": "openai", "key": "lab-key"}})
    assert _status(port, "/account/api/llm", bob, "POST", llm) == 403
    state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
    lab = next(g for g in state["groups"] if g["name"] == "lab")
    assert lab["llm"].startswith("openai") and not lab["own_key"]
    assert "lab-key" not in json.dumps(state)


def test_the_command_line_sets_assistants_and_own_keys(
    serve, tmp_path, monkeypatch, capsys
):
    users = _users(serve, tmp_path, ann=("a", ["admin"]), bob=("b", ["lab"]))
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "k-1")
    path = str(users.path)
    serve.main(["--users", path, "--set-group-llm", "lab", "openrouter",
                "m"])
    serve.main(["--users", path, "--set-llm", "ann", "anthropic"])
    serve.main(["--users", path, "--own-key", "group:lab", "no"])
    users = serve.Users(users.path)
    assert users.assistant("bob")["model"] == "m"
    assert users.assistant("ann") == {"provider": "anthropic", "key": "k-1",
                                      "source": "own"}
    assert not users.own_key("bob")
    capsys.readouterr()
    serve.main(["--users", path, "--list-users"])
    listed = capsys.readouterr().out
    assert "no own key" in listed and "k-1" not in listed
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "")
    serve.main(["--users", path, "--set-llm", "ann", "anthropic"])
    assert serve.Users(users.path).assistant("ann") is None
    # custom asks its address.
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "k-2")
    monkeypatch.setattr("builtins.input", lambda prompt: "https://x.test/v1")
    serve.main(["--users", path, "--set-llm", "ann", "custom", "m"])
    assert serve.Users(users.path).assistant("ann")["base_url"] == \
        "https://x.test/v1"


def test_the_server_and_the_sessions_name_the_same_variables(serve):
    from neurolang_apps.utils.tui import local_backend

    assert serve.ASSISTANT_ENV == local_backend.ASSISTANT_ENV
    assert serve.OWN_KEY_ENV == local_backend.OWN_KEY_ENV


def _http(port, method, path, headers=None, body=None):
    """The status and the headers of a request, its redirects not
    followed."""
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = None if body is None else json.dumps(body)
    headers = dict(headers or {})
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    response.read()
    connection.close()
    return response.status, response.headers


def _login(port, name, password):
    status, headers = _http(port, "POST", "/login", {},
                            {"name": name, "password": password})
    cookie = headers.get("Set-Cookie", "")
    return status, cookie.split(";")[0]


def test_users_log_in_on_a_page_and_out(console):
    port, users = console
    # A page the browser opens leads to the login page; the rest is
    # refused without a challenge, which would have the browser ask.
    status, headers = _http(port, "GET", "/admin?x=1",
                            {"Accept": "text/html"})
    assert (status, headers["Location"]) == (303, "/login?next=/admin?x=1")
    status, headers = _http(port, "GET", "/bundle.json")
    assert status == 401 and "WWW-Authenticate" not in headers
    assert _http(port, "GET", "/login")[0] == 200
    assert _login(port, "bob", "wrong")[0] == 403
    assert _http(port, "POST", "/login", {"Content-Type": "text/plain"})[0] \
        == 403
    status, cookie = _login(port, "bob", "bob-pw")
    assert status == 200 and cookie.startswith("neurolang_login=")
    assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] == 200
    me = _http(port, "GET", "/account/api/me", {"Cookie": cookie})
    assert me[0] == 200
    status, headers = _http(port, "GET", "/logout", {"Cookie": cookie})
    assert status == 303 and "Max-Age=0" in headers["Set-Cookie"]
    # A cookie altered, expired, or of a user removed is no login.
    name, _, signature = cookie.partition("=")[2].partition(".")
    forged = "neurolang_login=" + name[:-2] + "xx." + signature
    assert _http(port, "GET", "/bundle.json", {"Cookie": forged})[0] == 401


def test_a_login_ends_with_its_password_or_its_user(serve, console):
    port, users = console
    _, cookie = _login(port, "bob", "bob-pw")
    status, headers = _http(port, "POST", "/account/api/password",
                            {"Cookie": cookie},
                            {"current": "bob-pw", "new": "long-enough"})
    assert status == 200
    renewed = headers["Set-Cookie"].split(";")[0]
    assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] == 401
    assert _http(port, "GET", "/bundle.json", {"Cookie": renewed})[0] == 200
    users.remove("bob")
    assert _http(port, "GET", "/bundle.json", {"Cookie": renewed})[0] == 401


def test_a_login_cookie_expires_and_is_signed(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("a", ["admin"]))
    secret = b"s" * 32
    value = serve.sign_login(secret, "ann", users, 60)
    assert serve.check_login(secret, value, users) == "ann"
    assert serve.check_login(b"t" * 32, value, users) is None
    assert serve.check_login(
        secret, serve.sign_login(secret, "ann", users, -1), users) is None
    assert serve.check_login(secret, "garbage", users) is None


def test_the_login_cookie_is_secure_over_https(
    serve, tmp_path, certificate
):
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users,
                               tls=serve.tls_context(*certificate))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import http.client

        context = ssl.create_default_context(cafile=str(certificate[0]))
        context.check_hostname = False
        connection = http.client.HTTPSConnection(
            "127.0.0.1", server.server_address[1], context=context,
            timeout=10)
        connection.request(
            "POST", "/login", headers={"Content-Type": "application/json"},
            body=json.dumps({"name": "ann", "password": "ann-pw"}))
        response = connection.getresponse()
        response.read()
        connection.close()
        cookie = response.headers["Set-Cookie"]
        assert response.status == 200
        assert "Secure" in cookie and "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def guests(serve, tmp_path):
    """A server of users that lets guests in, with fake sessions."""
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    sessions = _FakeSessions()
    sessions.info = lambda: [{"user": u, "tab": t}
                             for u, t in dict.fromkeys(sessions.asked)]
    server = serve.make_server(tmp_path, "127.0.0.1", 0, sessions,
                               users=users, guests=True,
                               max_guest_sessions=1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], sessions
    server.shutdown()
    server.server_close()


def _guest(port):
    status, headers = _http(port, "POST", "/login/guest", {}, {})
    return status, headers.get("Set-Cookie", "").split(";")[0]


def test_a_guest_comes_in_without_an_account(serve, guests):
    port, sessions = guests
    assert _http(port, "GET", "/login/guest")[0] == 200
    status, cookie = _guest(port)
    assert status == 200 and cookie
    assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] == 200
    assert _http(port, "POST", "/api/start?session=t1", {"Cookie": cookie},
                 {"engine": "neurosynth"})[0] == 200
    [(user, tab)] = sessions.asked
    assert user.startswith(serve.GUEST_PREFIX) and tab == "t1"
    # Who they are, and nothing of an account or of the console.
    me = _http(port, "GET", "/account/api/me", {"Cookie": cookie})
    assert me[0] == 200
    for path, body in (("/account/api/password",
                        {"current": "x", "new": "long-enough"}),
                       ("/account/api/llm",
                        {"provider": "anthropic", "key": "k"})):
        assert _http(port, "POST", path, {"Cookie": cookie}, body)[0] == 403
    assert _http(port, "GET", "/admin/api/state", {"Cookie": cookie})[0] \
        == 403
    # Each guest is someone else, and the guests' sessions have a cap.
    _, other = _guest(port)
    assert other != cookie
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/start?session=t1", method="POST",
        headers={"Cookie": other, "Content-Type": "application/json"},
        data=b'{"engine": "neurosynth"}')
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=10)
    with refused.value:
        assert refused.value.code == 500
        assert "guests are all at work" in json.load(refused.value)["error"]
    assert len(sessions.asked) == 1  # the second guest got no session


def test_without_guests_there_are_none(serve, console):
    port, users = console
    assert _guest(port)[0] == 404
    # A guest's cookie of another server, or of this one before, is out.
    value = serve.sign_login(b"k" * 32, "guest:abc", users, 60)
    assert serve.check_login(b"k" * 32, value, users) is None
    assert serve.check_login(b"k" * 32, value, users, guests=True) \
        == "guest:abc"


def test_a_guests_sessions_have_no_key_of_the_server(serve):
    env, group = serve.guest_identity(["ALBERT_API_KEY"])
    assert env == {"ALBERT_API_KEY": None, serve.BRING_KEY_ENV: "1"}
    assert group == "guest"
    from neurolang_apps.utils.tui import local_backend

    assert serve.BRING_KEY_ENV == local_backend.BRING_KEY_ENV


def test_writers_of_the_users_file_in_two_processes_lose_nothing(
    serve, tmp_path
):
    users = _users(serve, tmp_path, ann=("ann-password", ["admin"]))
    script = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('serve_repl', "
        "sys.argv[1])\n"
        "serve = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(serve)\n"
        "users = serve.Users(sys.argv[2])\n"
        "for i in range(6):\n"
        "    users.change(sys.argv[3] + str(i), password='long-enough',\n"
        "                 create=True)\n"
    )
    writers = [subprocess.Popen([sys.executable, "-c", script, str(SCRIPT),
                                 str(users.path), who]) for who in "pq"]
    assert [writer.wait(120) for writer in writers] == [0, 0]
    assert len(serve.Users(users.path).current()) == 13
    assert len(users.current()) == 13  # the server reads them again


def test_a_change_of_the_users_file_within_a_tick_is_read(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("ann-password", ["admin"]))
    before = users.path.stat()
    users.path.write_text(users.path.read_text() + (
        f"bob:\n  password: {serve.hash_password('bob-password')}\n"))
    os.utime(users.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert users.check("bob", b"bob-password")


def test_the_users_file_keeps_a_user_and_starts_empty(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("ann-password", []))
    with pytest.raises(ValueError, match="keeps a user"):
        users.remove("ann")
    assert users.check("ann", b"ann-password")
    for text in ("", "users: {}\n"):  # a file made empty by hand
        users.path.write_text(text)
        new = serve.Users(users.path, new=True)
        new.change("bob", password="bob-password", groups=["admin"],
                   create=True)
        assert list(serve.Users(users.path).current()) == ["bob"]


@pytest.mark.parametrize("edit", [
    "  groups: 5\n", "  groups: [lab, [x]]\n", "  groups: {lab: 1}\n",
])
def test_a_malformed_users_file_is_told_and_kept(serve, tmp_path, edit):
    users = _users(serve, tmp_path, ann=("ann-password", ["admin"]))
    users.path.write_text(
        f"ann:\n  password: {serve.hash_password('ann-password')}\n{edit}")
    with pytest.raises(SystemExit, match="groups"):
        serve.read_users(users.path)
    assert users.check("ann", b"ann-password")  # the users as they were
    with pytest.raises(ValueError, match="groups"):  # no write over it
        users.change("ann", env={"X": "1"})
    assert edit in users.path.read_text()


def test_a_group_given_as_a_list_is_told(serve, tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(f"users:\n  ann:\n    password: "
                    f"{serve.hash_password('a')}\ngroups:\n  lab: [a]\n")
    with pytest.raises(SystemExit, match="the group lab"):
        serve.read_users(path)


def test_passwords_have_eight_characters_everywhere(
    serve, console, monkeypatch
):
    port, users = console
    with pytest.raises(ValueError, match="8 characters"):
        users.change("bob", password="short")
    assert _status(port, "/admin/api/user", _basic("ann-pw", "ann"), "POST",
                   {"name": "eve", "password": "short", "create": True}) \
        == 400
    monkeypatch.setattr(serve.getpass, "getpass", lambda prompt: "short")
    with pytest.raises(SystemExit, match="8 characters"):
        serve.main(["--users", str(users.path), "--set-password", "bob"])
    assert users.check("bob", b"bob-pw")


def test_a_key_kept_is_for_the_same_provider_and_address(serve, tmp_path):
    users = _users(serve, tmp_path, ann=("a", ["admin"]))
    users.change("ann", llm={"provider": "openai", "key": "admin-set"})
    with pytest.raises(ValueError, match="needs its key"):
        users.change("ann", llm={"provider": "openai",
                                 "base_url": "https://evil.test/v1"})
    users.change("ann", llm={"provider": "openai", "model": "m"})
    assert users.assistant("ann")["key"] == "admin-set"
    with pytest.raises(ValueError, match="base_url"):
        users.change("ann", llm={"provider": "custom", "key": "k"})
    users.change("ann", llm={"provider": "custom", "key": "k",
                             "base_url": "https://mine.test/v1"})
    with pytest.raises(ValueError, match="needs its key"):
        users.change("ann", llm={"provider": "custom",
                                 "base_url": "https://other.test/v1"})


def test_the_summary_of_the_assistants_is_that_of_the_users(
    serve, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    monkeypatch.setenv("NEUROLANG_LLM_CONFIG", str(config))
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    users = _users(serve, tmp_path, ann=("a", ["admin"]), bob=("b", ["lab"]),
                   eve=("e", ["zoo"]), carl=("c", []))
    users.change("ann", llm={"provider": "anthropic", "model": "m-a",
                             "key": "k"})
    users.change_group("lab", llm={"provider": "openai", "key": "k"})
    users.change_group("zoo", {"TEST_LLM_KEY": "zoo-key"})
    assistants = serve.users_assistants(users)
    assert assistants["ann"] == "anthropic (m-a)"
    assert assistants["bob"].startswith("openai")
    assert assistants["eve"] == "test (m)" and assistants["carl"] is None
    assert "TEST_LLM_KEY" not in os.environ


def test_a_logout_ends_the_login_of_a_copied_cookie(serve, guests):
    port, _ = guests
    for status, cookie in (_login(port, "ann", "ann-pw"), _guest(port)):
        assert status == 200
        assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] \
            == 200
        assert _http(port, "GET", "/logout", {"Cookie": cookie})[0] == 303
        assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] \
            == 401
    # A login after it, in the same second, is good.
    status, cookie = _login(port, "ann", "ann-pw")
    assert _http(port, "GET", "/bundle.json", {"Cookie": cookie})[0] == 200


def test_the_login_cookie_is_secure_behind_a_proxy(serve, tmp_path):
    args = serve.parse_args(["--local", "--users", "u", "--secure-cookie",
                             "--daemon", "install"])
    assert "--secure-cookie" in serve.service_command(args)
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users,
                               secure_cookie=True)
    with _serving(server) as port:
        status, headers = _http(port, "POST", "/login", {},
                                {"name": "ann", "password": "ann-pw"})
        assert status == 200 and "; Secure" in headers["Set-Cookie"]


@pytest.mark.parametrize("before", [
    "x=a b", 'x={"a":1}', "neurolang_login=é", "neurolang_login=bad.sig",
    "x", "",
])
def test_the_login_cookie_is_found_among_others(console, before):
    port, _ = console
    _, cookie = _login(port, "bob", "bob-pw")
    header = f"{before}; {cookie}; y=\"z\""
    assert _http(port, "GET", "/bundle.json", {"Cookie": header})[0] == 200
    assert _http(port, "GET", "/bundle.json",
                 {"Cookie": before or "x=1"})[0] == 401


@pytest.mark.parametrize("path", [
    "/login", "/login/guest", "/account/api/llm", "/account/api/password",
    "/admin/api/user",
])
def test_the_bodies_of_the_login_account_and_console_are_checked(
    serve, guests, path, capsys
):
    port, _ = guests
    _, cookie = _login(port, "ann", "ann-pw")
    for body, status in ((b"[]", 400), (b"{bad", 400), (b"", 413)):
        headers = ("Content-Length: 2000000" if status == 413
                   else f"Content-Length: {len(body)}")
        answered, text = _raw(port, (
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\nCookie: {cookie}\r\n"
            f"Connection: close\r\n{headers}\r\n\r\n"
        ).encode() + body)
        assert answered == status, (body, text)
        assert json.loads(text)["error"]
    assert "Traceback" not in capsys.readouterr().err


def test_a_failing_change_of_the_account_is_told_and_logged(
    console, monkeypatch, capsys
):
    port, users = console

    def fail(*args, **kwargs):
        raise RuntimeError("the secret detail")

    monkeypatch.setattr(users, "change", fail)
    _, cookie = _login(port, "bob", "bob-pw")
    for path, body in (("/account/api/llm", {"provider": "anthropic",
                                             "key": "k"}),
                       ("/account/api/password",
                        {"current": "bob-pw", "new": "long-enough"})):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", method="POST",
            headers={"Cookie": cookie, "Content-Type": "application/json"},
            data=json.dumps(body).encode())
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=10)
        with refused.value:
            assert refused.value.code == 500
            assert json.load(refused.value) == {"error": "RuntimeError"}
    assert "the secret detail" in capsys.readouterr().err


def test_a_change_refreshes_the_sessions_of_its_users_only(serve, tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]),
                   bob=("bob-pw", ["lab"]), eve=("eve-pw", []))
    sessions = _FakeSessions()
    sessions.info = lambda: []
    refreshed = []
    sessions.refresh = refreshed.append
    server = serve.make_server(tmp_path, "127.0.0.1", 0, sessions,
                               users=users)
    with _serving(server) as port:
        llm = {"provider": "openai", "key": "k"}
        _json_of(port, "/admin/api/group", "ann", "ann-pw",
                 {"name": "lab", "llm": llm})
        _json_of(port, "/admin/api/user", "ann", "ann-pw",
                 {"name": "eve", "llm": llm})
        _json_of(port, "/account/api/llm", "bob", "bob-pw", llm)
    assert refreshed == [["bob"], ["eve"], ["bob"]]


def test_the_console_exports_the_examples_of_each_engine(serve, tmp_path):
    from neurolang_apps.utils.llm.feedback_server import FeedbackStore

    (tmp_path / "bundle.json").write_text("{}")
    db = tmp_path / "feedback.sqlite"
    store = FeedbackStore(db)
    for engine, question in (("neurosynth", "Pain?"),
                             ("neurostore", "Fear?")):
        answer = store.add_answer({"engine": engine, "question": question,
                                   "language": "datalog", "query": "q",
                                   "valid": True})
        store.add_feedback(answer, {"verdict": "correct"})
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users,
                               feedback_db=db)
    with _serving(server) as port:
        state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
        assert state["feedback_engines"] == ["neurostore", "neurosynth"]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/admin/feedback.yaml?engine=neurosynth",
            headers=_basic("ann-pw", "ann"))
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode()
            assert "neurosynth" in response.headers["Content-Disposition"]
        assert "Pain?" in text and "Fear?" not in text
        for query in ("", "?engine=nothing", '?engine="x'):
            assert _status(port, f"/admin/feedback.yaml{query}",
                           _basic("ann-pw", "ann")) == 400


# --- signing in with ORCID -------------------------------------------------

ORCID_ANN = "0000-0002-1825-0097"  # ORCID's example iD
ORCID_NEW = "0000-0002-1694-233X"


def test_orcid_ids_and_their_check_digit(serve):
    assert serve.orcid.is_orcid(ORCID_ANN)
    assert serve.orcid.is_orcid(ORCID_NEW)
    for text in ("0000-0002-1825-0098", "0000-0002-1825-009", "ann",
                 "0000000218250097", None):
        assert not serve.orcid.is_orcid(text)


def test_a_user_named_by_an_orcid_id_needs_no_password(serve, tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(f"users:\n  ann:\n    password: "
                    f"{serve.hash_password('ann-pw')}\n    groups: [admin]\n"
                    f"  '{ORCID_ANN}':\n    groups: [lab]\n"
                    "    orcid_name: Josiah Carberry\n")
    users = serve.Users(path)
    assert users.groups(ORCID_ANN) == ["lab"]
    assert not users.check(ORCID_ANN, b"")  # no password to log in with
    path.write_text("users:\n  bob:\n    groups: [lab]\n")
    with pytest.raises(SystemExit, match="ORCID iD"):
        serve.read_users(path)
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    users.change(ORCID_NEW, groups=["lab"], create=True)
    with pytest.raises(ValueError, match="ORCID iD as their name"):
        users.change("carl", groups=["lab"], create=True)
    assert "password" not in users.current()[ORCID_NEW]


def test_the_orcid_client_of_its_file(serve, tmp_path):
    path = tmp_path / "orcid.yaml"
    path.write_text("client_id: APP-1\nclient_secret: s3cret\n"
                    "sandbox: true\n")
    client = serve.orcid.OrcidClient.from_file(path)
    url = client.authorize_url("https://x.org/login/orcid/callback", "st")
    assert url.startswith("https://sandbox.orcid.org/oauth/authorize?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query["scope"] == ["/authenticate"] and query["state"] == ["st"]
    assert query["client_id"] == ["APP-1"]
    path.write_text("client_id: APP-1\n")
    with pytest.raises(ValueError, match="client_secret"):
        serve.orcid.OrcidClient.from_file(path)


class _FakeOrcid:
    """ORCID, telling who signed in by the code it gave."""

    base = "https://orcid.example"
    redirect_uri = ""

    def __init__(self, people):
        self.people = people

    def authorize_url(self, redirect_uri, state):
        return f"{self.base}/authorize?" + urllib.parse.urlencode(
            {"redirect_uri": redirect_uri, "state": state})

    def exchange(self, code, redirect_uri):
        if code not in self.people:
            raise self.error("unknown code")
        return self.people[code]


@pytest.fixture
def orcid_server(serve, tmp_path):
    """A server of the admin ann and of the ORCID user of ORCID_ANN, whose
    users sign in with a fake ORCID."""
    (tmp_path / "bundle.json").write_text("{}")
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    users.change(ORCID_ANN, groups=["lab"], create=True)
    fake = _FakeOrcid({"c-ann": (ORCID_ANN, "Josiah Carberry"),
                       "c-new": (ORCID_NEW, "Sofia Garcia")})
    fake.error = serve.orcid.OrcidError
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users,
                               orcid_client=fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], users
    server.shutdown()
    server.server_close()


def _cookie_of(headers, name):
    for value in headers.get_all("Set-Cookie") or ():
        if value.startswith(name + "="):
            return value.split(";")[0]
    return None


def _sign_in(port, code, next_page="/admin"):
    """Sign in with the fake ORCID, as the browser does: the status, the
    headers of the way back, and the cookie of the start."""
    status, headers = _http(
        port, "GET", f"/login/orcid?next={urllib.parse.quote(next_page)}")
    assert status == 303
    state = urllib.parse.parse_qs(
        urllib.parse.urlsplit(headers["Location"]).query)["state"][0]
    started = _cookie_of(headers, "neurolang_orcid")
    status, back = _http(
        port, "GET", f"/login/orcid/callback?code={code}&state={state}",
        {"Cookie": started})
    return status, back


def _json_at(port, method, path, headers=None, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.load(error)


def test_a_user_signs_in_with_orcid(orcid_server):
    port, users = orcid_server
    assert _json_at(port, "GET", "/login/options")[1] == {
        "guests": False, "orcid": True}
    status, headers = _http(port, "GET", "/login/orcid?next=/admin")
    location = headers["Location"]
    assert location.startswith("https://orcid.example/authorize?")
    assert "redirect_uri=http%3A%2F%2F127.0.0.1" in location
    status, back = _sign_in(port, "c-ann", "/admin?x=1")
    assert (status, back["Location"]) == (303, "/admin?x=1")
    login = _cookie_of(back, "neurolang_login")
    assert _http(port, "GET", "/bundle.json", {"Cookie": login})[0] == 200
    # the name ORCID gives is kept, and shown on their account page
    assert users.current()[ORCID_ANN]["orcid_name"] == "Josiah Carberry"
    status, me = _json_at(port, "GET", "/account/api/me", {"Cookie": login})
    assert (me["name"], me["orcid_name"], me["orcid_only"]) == (
        ORCID_ANN, "Josiah Carberry", True)
    status, answer = _json_at(port, "POST", "/account/api/password",
                              {"Cookie": login},
                              {"current": "", "new": "long-enough"})
    assert status == 403 and "ORCID" in answer["error"]
    # a page off this site is not where they are sent
    assert _sign_in(port, "c-ann", "//evil.example/")[1]["Location"] == "/"


def test_a_sign_in_with_orcid_needs_its_state(orcid_server):
    port, _ = orcid_server
    status, headers = _http(port, "GET", "/login/orcid")
    state = urllib.parse.parse_qs(
        urllib.parse.urlsplit(headers["Location"]).query)["state"][0]
    # without the cookie of its start (another browser), or another state
    _, back = _http(port, "GET",
                    f"/login/orcid/callback?code=c-ann&state={state}")
    assert back["Location"] == "/login?orcid=expired"
    _, back = _http(port, "GET",
                    "/login/orcid/callback?code=c-ann&state=other",
                    {"Cookie": _cookie_of(headers, "neurolang_orcid")})
    assert back["Location"] == "/login?orcid=expired"
    assert _cookie_of(back, "neurolang_login") is None
    # a code ORCID does not know, or a sign-in cancelled at ORCID
    assert _sign_in(port, "c-bad")[1]["Location"] == "/login?orcid=failed"
    _, back = _http(port, "GET", "/login/orcid/callback?error=access_denied")
    assert back["Location"] == "/login?orcid=cancelled"


def test_an_unknown_orcid_id_asks_for_an_account(orcid_server, tmp_path):
    port, users = orcid_server
    status, back = _sign_in(port, "c-new")
    assert back["Location"] == "/login?orcid=request"
    assert _cookie_of(back, "neurolang_login") is None
    asking = {"Cookie": _cookie_of(back, "neurolang_request")}
    assert _json_at(port, "GET", "/login/request", asking)[1] == {
        "orcid": ORCID_NEW, "name": "Sofia Garcia", "requested": False}
    assert _json_at(port, "POST", "/login/request", asking,
                    {"message": "PhD student of the lab"})[0] == 200
    assert _json_at(port, "GET", "/login/request", asking)[1]["requested"]
    assert _json_at(port, "GET", "/login/request")[0] == 403  # no sign-in
    # the admins see it, and approve it in a group
    admin = _basic("ann-pw", "ann")
    state = _json_at(port, "GET", "/admin/api/state", admin)[1]
    assert state["orcid"] is True
    [asked] = state["requests"]
    assert (asked["orcid"], asked["name"], asked["message"]) == (
        ORCID_NEW, "Sofia Garcia", "PhD student of the lab")
    assert _json_at(port, "POST", "/admin/api/approve-request", admin,
                    {"name": ORCID_NEW, "groups": ["lab"]})[0] == 200
    assert users.groups(ORCID_NEW) == ["lab"]
    assert users.current()[ORCID_NEW]["orcid_name"] == "Sofia Garcia"
    assert _json_at(port, "GET", "/admin/api/state", admin)[1][
        "requests"] == []
    # and they sign in now
    _, back = _sign_in(port, "c-new", "/")
    assert back["Location"] == "/"
    assert _cookie_of(back, "neurolang_login")


def test_an_admin_declines_a_request(orcid_server):
    port, users = orcid_server
    _, back = _sign_in(port, "c-new")
    asking = {"Cookie": _cookie_of(back, "neurolang_request")}
    _json_at(port, "POST", "/login/request", asking, {"message": "hi"})
    admin = _basic("ann-pw", "ann")
    assert _json_at(port, "POST", "/admin/api/decline-request", admin,
                    {"name": ORCID_NEW})[0] == 200
    assert ORCID_NEW not in users.current()
    status, answer = _json_at(port, "POST", "/admin/api/approve-request",
                              admin, {"name": ORCID_NEW, "groups": []})
    assert status == 400 and "No request" in answer["error"]
    # a user who is no admin approves nothing
    _, back = _sign_in(port, "c-ann")
    lab = {"Cookie": _cookie_of(back, "neurolang_login")}
    assert _json_at(port, "POST", "/admin/api/approve-request", lab,
                    {"name": ORCID_NEW, "groups": ["admin"]})[0] == 403


def test_without_orcid_there_is_no_sign_in_with_it(console):
    port, _ = console
    assert _json_at(port, "GET", "/login/options")[1]["orcid"] is False
    assert _http(port, "GET", "/login/orcid")[0] == 404
    assert _http(port, "GET", "/login/request")[0] == 404


def test_the_requests_from_the_command_line(serve, tmp_path, capsys):
    path = tmp_path / "users.yaml"
    serve.main(["--users", str(path), "--add-user", ORCID_ANN,
                "--groups", "admin"])
    assert "signs in with ORCID" in capsys.readouterr().out
    assert "password" not in serve.read_users(path)[ORCID_ANN]
    requests = serve.orcid.AccountRequests(serve.orcid.requests_path(path))
    requests.add(ORCID_NEW, "Sofia Garcia", "hello")
    serve.main(["--users", str(path), "--list-requests"])
    assert f"{ORCID_NEW}\tSofia Garcia\t" in capsys.readouterr().out
    serve.main(["--users", str(path), "--approve-request", ORCID_NEW,
                "--groups", "lab"])
    assert serve.read_users(path)[ORCID_NEW]["groups"] == ["lab"]
    assert requests.list() == []
    requests.add("0000-0001-5109-3700", "", "")
    serve.main(["--users", str(path), "--decline-request",
                "0000-0001-5109-3700"])
    assert requests.list() == []
    with pytest.raises(SystemExit):
        serve.parse_args(["--orcid", str(path)])
    assert "--orcid signs in the users of --users" in capsys.readouterr().err


# --- the configuration and the secrets in YAML files ------------------------

def _secrets(tmp_path, text, mode=0o600):
    path = tmp_path / "secrets.yaml"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_options_of_a_configuration_file(serve, tmp_path):
    config = tmp_path / "conf" / "server.yaml"
    config.parent.mkdir()
    config.write_text(
        "port: 8443\nhost: 0.0.0.0\nlocal: true\ndata-dir: data\n"
        "engines: [neurosynth]\nusers: users.yaml\nmax_sessions: 3\n"
        "cert: /etc/ssl/cert.pem\nsecrets: secrets.yaml\n")
    _secrets(config.parent, "env:\n  ALBERT_API_KEY: k\n")
    args = serve.parse_args(["--config", str(config)])
    assert (args.port, args.host, args.local, args.max_sessions) == (
        8443, "0.0.0.0", True, 3)
    # paths relative to the file, absolute ones as they are
    assert args.data_dir == config.parent / "data"
    assert args.users == config.parent / "users.yaml"
    assert args.cert == Path("/etc/ssl/cert.pem")
    assert args.engines == ["neurosynth"]
    assert args.secret == {"env": {"ALBERT_API_KEY": "k"}}
    # an option given replaces the file's
    assert serve.parse_args(["--config", str(config), "--port", "9000"]
                            ).port == 9000


@pytest.mark.parametrize("text, says", [
    ("prt: 1\n", "no option 'prt'"),
    ("engines: neurosynth\n", "engines is a list"),
    ("port: [1, 2]\n", "port is one value"),
    ("local: yes please\n", "local is true or false"),
    ("port: eighty\n", "port is not a int"),
    ("engines: [nowhere]\n", "engines is one of"),
    ("daemon: install\n", "no option 'daemon'"),
    ("- port\n", "not a mapping"),
])
def test_a_configuration_file_that_is_not_one(serve, tmp_path, text, says):
    config = tmp_path / "server.yaml"
    config.write_text(text)
    with pytest.raises(SystemExit, match=re.escape(says)):
        serve.parse_args(["--config", str(config)])


def test_the_secrets_are_their_owners(serve, tmp_path, capsys):
    path = _secrets(tmp_path, "orcid:\n  client_id: APP-1\n"
                    "  client_secret: s\npassword: pw\nkey_password: kp\n")
    secret = serve.read_secrets(path)
    assert secret["password"] == "pw" and secret["key_password"] == "kp"
    client = serve.orcid.OrcidClient.from_mapping(secret["orcid"], "here")
    assert client.client_id == "APP-1"
    path.chmod(0o644)
    with pytest.raises(SystemExit, match="chmod 600"):
        serve.read_secrets(path)
    with pytest.raises(SystemExit, match="the secrets are those of"):
        serve.read_secrets(_secrets(tmp_path, "tokens: x\n"))
    with pytest.raises(SystemExit, match="env is a mapping"):
        serve.read_secrets(_secrets(tmp_path, "env: [a]\n"))
    # a secret given twice, or without what it is for
    path = _secrets(tmp_path, "password: pw\n")
    other = tmp_path / "password.txt"
    other.write_text("pw\n")
    with pytest.raises(SystemExit):
        serve.parse_args(["--secrets", str(path), "--password-file",
                          str(other)])
    assert "or the password of" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        serve.parse_args(["--secrets", str(_secrets(
            tmp_path, "orcid:\n  client_id: A\n  client_secret: s\n"))])
    assert "--orcid signs in the users of --users" in capsys.readouterr().err


def test_the_service_runs_its_configuration(serve, tmp_path, monkeypatch):
    config = tmp_path / "server.yaml"
    config.write_text("port: 8443\n")
    args = serve.parse_args(["--config", str(config), "--python",
                             sys.executable, "--daemon", "install"])
    command = serve.service_command(args)
    assert command[1:] == ["-m", "serve_repl", "--no-browser", "--config",
                           str(config.resolve())]
    # without one, its options and the file of its secrets
    secrets_file = _secrets(tmp_path, "password: pw\n")
    args = serve.parse_args(["--secrets", str(secrets_file), "--python",
                             sys.executable, "--daemon", "install"])
    command = serve.service_command(args)
    assert command[command.index("--secrets") + 1] == str(
        secrets_file.resolve())


def test_a_server_of_its_secrets(serve, tmp_path, monkeypatch, capsys):
    """The password of the secrets is that of the server; their variables
    are those of the sessions of --local, as --env-file's."""
    made = {}
    monkeypatch.setattr(serve, "make_server",
                        lambda *a, **k: made.update(args=a, kw=k)
                        or (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(serve, "write_index", lambda *a: None)
    monkeypatch.setattr(serve, "prepare_site", lambda *a, **k: tmp_path)
    path = _secrets(tmp_path, "password: s3cret-pw\n")
    with pytest.raises(KeyboardInterrupt):
        serve.main(["--secrets", str(path), "--no-browser"])
    assert made["args"][5] == "s3cret-pw"
    path = _secrets(tmp_path, "password: pw\nenv:\n  A_KEY: k\n")
    with pytest.raises(SystemExit):
        serve.parse_args(["--secrets", str(path)])
    assert "with --local" in capsys.readouterr().err


def test_the_console_shows_the_usage_and_the_log_of_each_user(
    serve, tmp_path
):
    from neurolang_apps.utils.tui.activity import ActivityLog

    (tmp_path / "bundle.json").write_text("{}")
    sessions = _FakeSessions()
    sessions.info = lambda: []
    sessions.builder_usage = {"pid": 1, "cpu": 2.0, "rss": 900.0,
                              "pss": 450.0, "private": 10.0}
    sessions.activity = ActivityLog(tmp_path / "activity.sqlite")
    sessions.activity.add("bob", "submit", text="ans(x) :- p(x)",
                          response="x\n1", failed=False, seconds=0.5,
                          cpu=0.4, memory=30.0)
    # its process's CPU in all, and the most memory of the session
    sessions.activity.add("bob", "end", text="idle", cpu=3.0, memory=80.0)
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]),
                   bob=("bob-pw", ["lab"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, sessions,
                               users=users)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
        assert state["activity"] and state["builder"]["rss"] == 900.0
        [bob] = _json_of(port, "/admin/api/usage", "ann", "ann-pw")
        assert (bob["user"], bob["inputs"], bob["cpu"], bob["seconds"],
                bob["memory"]) == ("bob", 1, 0.4, 0.5, 80.0)
        log = f"http://127.0.0.1:{port}/admin/activity.jsonl?user=bob"
        request = urllib.request.Request(log, headers=_basic("ann-pw",
                                                             "ann"))
        with urllib.request.urlopen(request, timeout=10) as response:
            events = [json.loads(line) for line in response]
        assert [e["text"] for e in events] == ["ans(x) :- p(x)", "idle"]
        assert _status(port, "/admin/activity.jsonl?user=bob",
                       _basic("bob-pw", "bob")) == 403
    finally:
        server.shutdown()
        server.server_close()


def test_the_activity_is_logged_with_local_users(serve, tmp_path,
                                                 monkeypatch):
    with pytest.raises(SystemExit):
        serve.parse_args(["--local", "--activity-db", "a.sqlite"])
    monkeypatch.chdir(tmp_path)
    command = serve.service_command(serve.parse_args([
        "--local", "--users", "users.yaml", "--activity-db", "a.sqlite",
        "--daemon", "install",
    ]))
    after = dict(zip(command, command[1:]))
    assert after["--activity-db"] == str(tmp_path / "a.sqlite")


def test_the_health_is_open_to_monitors_and_tells_a_dead_builder(
    serve, tmp_path
):
    (tmp_path / "bundle.json").write_text("{}")
    sessions = _FakeSessions()
    sessions.info = lambda: []
    sessions.dead = None
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    for options in ({"users": users}, {"password": "pw"}, {}):
        server = serve.make_server(tmp_path, "127.0.0.1", 0, sessions,
                                   **options)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=10) as answer:
                health = json.load(answer)
            assert health["ok"] and health["uptime_seconds"] >= 0
            assert set(health) == {"ok", "uptime_seconds"}
            sessions.dead = "The builder of the engines ended"
            assert _status(port, "/health") == 503
            sessions.dead = None
        finally:
            server.shutdown()
            server.server_close()


class _FakeDashboard:
    def __init__(self):
        self.asked = []

    def get(self, target):
        self.asked.append(target)
        return 200, [("Content-Type", "text/html")], b"<h1>logs</h1>"


def test_the_dashboard_is_the_admins_and_read_only(serve, tmp_path):
    from neurolang_apps.utils.tui.dashboard import PREFIX

    assert serve.DASHBOARD_PREFIX == PREFIX
    (tmp_path / "bundle.json").write_text("{}")
    board = _FakeDashboard()
    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]),
                   bob=("bob-pw", ["lab"]))
    server = serve.make_server(tmp_path, "127.0.0.1", 0, users=users,
                               dashboard=board)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        target = "/admin/logs/activity/by_user?days=7"
        assert _status(port, target, _basic("ann-pw", "ann")) == 200
        assert board.asked == [target]
        assert _status(port, target, _basic("bob-pw", "bob")) == 403
        assert _status(port, "/admin/logs/", _basic("ann-pw", "ann"),
                       "POST", {}) == 405
        assert _http(port, "GET", "/admin/logs",
                     _basic("ann-pw", "ann"))[0] == 301
        state = _json_of(port, "/admin/api/state", "ann", "ann-pw")
        assert state["dashboard"]
    finally:
        server.shutdown()
        server.server_close()


def test_the_dashboard_options(serve, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        serve.parse_args(["--local", "--users", "u", "--dashboard"])
    monkeypatch.chdir(tmp_path)
    command = serve.service_command(serve.parse_args([
        "--local", "--users", "u", "--activity-db", "a.sqlite",
        "--dashboard", "--daemon", "install",
    ]))
    assert "--dashboard" in command


def test_the_status_of_the_service_is_its_managers(serve, tmp_path,
                                                   monkeypatch, capsys):
    unit = tmp_path / "neurolang-repl.service"
    monkeypatch.setattr(serve, "service_file", lambda: unit)
    serve.main(["--daemon", "status"])
    assert "No service" in capsys.readouterr().out
    unit.write_text("")
    asked = []

    def run(command, **options):
        asked.append(command)
        return subprocess.CompletedProcess(
            command, 0, "ActiveState=active\nNRestarts=2\n", "")

    monkeypatch.setattr(serve.sys, "platform", "linux")
    monkeypatch.setattr(serve.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(serve.subprocess, "run", run)
    serve.main(["--daemon", "status"])
    out = capsys.readouterr().out
    assert "NRestarts=2" in out and "/health" in out
    assert asked[0][:4] == ["systemctl", "--user", "show", "neurolang-repl"]


def test_a_sigterm_stops_the_server_as_a_stop(serve, tmp_path):
    from neurolang_apps.utils.tui.dashboard import QUERIES

    users = _users(serve, tmp_path, ann=("ann-pw", ["admin"]))
    db = tmp_path / "activity.sqlite"
    process = subprocess.Popen([
        sys.executable, str(SCRIPT), "--local", "--users", str(users.path),
        "--activity-db", str(db), "--data-dir", str(tmp_path / "data"),
        "--site", str(tmp_path / "site"), "--port", "0", "--no-browser",
    ])
    try:
        deadline = time.monotonic() + 120
        while not db.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        import sqlite3

        while time.monotonic() < deadline:
            connection = sqlite3.connect(db)
            try:
                runs = connection.execute("SELECT COUNT(*) FROM runs"
                                          ).fetchone()[0]
            except sqlite3.OperationalError:
                runs = 0
            finally:
                connection.close()
            if runs:
                break
            time.sleep(0.2)
        time.sleep(1)  # into serve_forever
        process.send_signal(signal.SIGTERM)
        assert process.wait(60) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    connection = sqlite3.connect(db)
    try:
        hows = [row[3] for row in connection.execute(QUERIES["runs"][1])]
    finally:
        connection.close()
    assert hows == ["stopped"]
