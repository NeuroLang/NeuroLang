"""Tests for the script serving the REPL in the browser on this machine."""

import importlib.util
import json
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

    def build(repl, embeddings, engines=None):
        built.append(embeddings)
        (repl / "bundle.json").write_text(
            json.dumps({"engines": {"neurosynth": {}, "neurostore": {}}})
        )

    monkeypatch.setattr(serve, "build_bundle", build)
    serve.prepare_site(tmp_path)
    serve.prepare_site(tmp_path)
    assert built == [None]
    serve.prepare_site(tmp_path, rebuild=True, embeddings="emb")
    assert built == [None, "emb"]


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
        names = engines or ["neurosynth", "neurostore"]
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
