"""Tests of the REPL of the page run by this machine's Python."""

import json
import threading
import time

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.tui.local_backend import LocalBackend  # noqa: E402


# An engine of the registry, for the banner and the examples of the page,
# whose relations are a test's (one whose studies are not PubMed's: the
# page links none, and nothing is downloaded).
def _engine(name, say):
    say(f"Building {name}")
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    return nl


@pytest.fixture
def backend(tmp_path):
    return LocalBackend(tmp_path, build=_engine)


def _messages(listener):
    messages = []
    while not listener.empty():
        messages.append(json.loads(listener.get()))
    return messages


def test_start_builds_the_engine_and_tells_its_steps(backend):
    listener = backend.listen()
    ready = backend.start("neurostore")
    assert "neurostore" in ready["banner"]
    assert {"banner", "providers", "examples", "questions", "predicates",
            "commands", "canRate", "source", "assistant"} <= set(ready)
    assert any(p["name"] == "person" for p in ready["predicates"])
    statuses = [m["message"] for m in _messages(listener)]
    assert statuses[0] == "Building neurostore…"
    assert statuses[-1] == "Starting the REPL…"


def test_a_query_comes_back_as_the_page_takes_it(backend):
    backend.start("neurostore")
    reply = backend.submit("ans(n, a) :- person(n, a), (a > 4)")
    assert not reply["failed"]
    assert reply["tables"] and reply["mode"] == "datalog"
    assert "bob" in json.dumps(reply["tables"])
    assert backend.submit("ans(x) :- person(x, n")["failed"]


def test_each_start_has_a_new_session(backend):
    backend.start("neurostore")
    backend.submit("old(n) :- person(n, a)")
    assert backend.submit("ans(n) :- old(n)")["tables"]
    backend.start("neurostore")
    reply = backend.submit("ans(n) :- old(n)")
    assert "old has no facts or rules" in reply["html"]


def test_a_download_is_the_file_of_a_result(backend):
    backend.start("neurostore")
    backend.submit("ans(n) :- person(n, a)")
    file = backend.download(0, "csv")
    assert file.name.endswith(".csv") and b"alice" in file.data


def test_a_stop_interrupts_the_request_at_work(backend, monkeypatch):
    backend.start("neurostore")
    started = threading.Event()

    def busy(text):
        started.set()
        while True:
            time.sleep(0.01)

    monkeypatch.setattr(backend.repl, "submit", busy)
    replies = []
    worker = threading.Thread(target=lambda: replies.append(
        backend.submit("anything")
    ))
    worker.start()
    started.wait(5)
    assert backend.stop()
    worker.join(5)
    assert replies[0]["failed"] and "Stopped" in replies[0]["html"]
    assert not backend.stop()


def test_without_a_key_the_assistant_is_that_of_the_machine(backend):
    pytest.importorskip("httpx")
    backend.start("neurostore")
    state = backend.repl.app.assistant
    assert state.provider is None
    backend.set_llm("anthropic", "a-key")
    assert state.provider == "anthropic"
    backend.set_llm()
    assert state.provider is None


def test_a_request_before_the_start_says_so(backend):
    with pytest.raises(RuntimeError, match="not started"):
        backend.submit("ans(x) :- person(x, a)")
