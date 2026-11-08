"""Tests for the script serving the REPL in the browser on this machine."""

import importlib.util
import json
import subprocess
import threading
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

    def build(repl, embeddings):
        built.append(embeddings)
        (repl / "bundle.json").write_text("{}")

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

    def build(repl, embeddings):
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
