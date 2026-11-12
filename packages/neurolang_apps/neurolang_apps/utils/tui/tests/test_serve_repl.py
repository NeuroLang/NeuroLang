"""Tests for the script serving the REPL in the browser on this machine."""

import importlib.util
import json
import plistlib
import shutil
import socket
import ssl
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

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
        assert error.value.code == 500
        assert json.load(error.value) == {"error": "ValueError: it broke"}
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


def test_a_key_with_a_passphrase_is_refused(serve, tmp_path, certificate):
    cert, key = certificate
    locked = tmp_path / "locked.pem"
    subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-out", str(locked),
         "-aes256", "-passout", "pass:secret"],
        check=True, capture_output=True,
    )
    with pytest.raises(SystemExit, match="passphrase"):
        serve.tls_context(cert, locked)


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
    assert "Environment=PATH=/opt/gh/bin:/usr/bin\n" in unit
    assert f"Environment=VIRTUAL_ENV={venv}\n" in unit
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
    assert "ExecStart=/py -m serve_repl --site '/a b/100%%'\n" in unit
    assert f"WorkingDirectory={tmp_path}\n" in unit
    assert "WantedBy=default.target" in unit


def test_the_launchd_agent_runs_the_command(serve, tmp_path):
    agent = plistlib.loads(serve.launchd_plist(["/py", "-m", "x"], tmp_path))
    assert agent["Label"] == serve.LAUNCHD_LABEL
    assert agent["ProgramArguments"] == ["/py", "-m", "x"]
    assert agent["RunAtLoad"] and agent["WorkingDirectory"] == str(tmp_path)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_service_is_installed_reinstalled_and_removed(
    serve, tmp_path, monkeypatch, platform
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(serve.sys, "platform", platform)
    monkeypatch.setattr(serve.shutil, "which", lambda name: name)
    managed = []
    monkeypatch.setattr(
        serve.subprocess, "run",
        lambda command, check=True: managed.append(command[1:]),
    )
    built = []
    monkeypatch.setattr(serve, "prepare_site",
                        lambda *args, **kwargs: built.append(args))
    monkeypatch.setattr(serve, "tls_context", lambda *args: object())
    path = serve.service_file()
    assert str(path).startswith(str(tmp_path))

    serve.main(["--daemon", "install", "--site", str(tmp_path / "site"),
                "--cert", "cert.pem"])
    assert path.is_file() and len(built) == 1
    assert "--cert" in path.read_bytes().decode()
    with pytest.raises(SystemExit, match="reinstall"):
        serve.main(["--daemon", "install"])
    serve.main(["--daemon", "reinstall", "--site", str(tmp_path / "site")])
    assert path.is_file() and len(built) == 2
    assert "--cert" not in path.read_bytes().decode()
    serve.main(["--daemon", "remove"])
    assert not path.exists()
    serve.main(["--daemon", "remove"])  # nothing installed: nothing done
    verbs = [command[1] if command[0] == "--user" else command[0]
             for command in managed]
    if platform == "linux":
        assert verbs == ["daemon-reload", "enable", "disable",
                         "daemon-reload", "daemon-reload", "enable",
                         "disable", "daemon-reload"]
    else:
        assert verbs == ["bootstrap", "bootout", "bootstrap", "bootout"]
