"""Tests of the REPL of the page run by this machine's Python."""

import json
import os
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


def test_an_engine_builds_in_a_data_directory_not_made_yet(
    tmp_path, monkeypatch
):
    from neurolang_apps.utils.tui import local_backend

    built = []
    monkeypatch.setattr(
        local_backend.engine_registry, "build_engine",
        lambda name, data_dir, *args, **kwargs: built.append(data_dir),
    )
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "first" / "neurolang_data"
    LocalBackend(data)._build_engine("neurosynth", lambda message: None)
    assert built == [data] and data.is_dir()


def test_the_assistant_of_the_user_and_their_own_key(backend, monkeypatch):
    from neurolang_apps.utils.tui import local_backend

    monkeypatch.setenv(local_backend.ASSISTANT_ENV, json.dumps(
        {"provider": "anthropic", "model": "m", "key": "k"}))
    ready = backend.start("neurostore")
    assert ready["assistant"] and ready["ownKey"]
    provider = backend.repl.app.assistant.config().provider()
    assert (provider.name, provider.model) == ("anthropic", "m")
    monkeypatch.setenv(local_backend.OWN_KEY_ENV, "0")
    assert not backend.start("neurostore")["ownKey"]
    with pytest.raises(ValueError, match="admin"):
        backend.set_llm(provider="openai", key="mine")
    backend.set_llm()  # back to the user's: allowed


def test_the_users_new_assistant_keeps_the_pages_key(backend, monkeypatch):
    from neurolang_apps.utils.tui import local_backend

    # use_identity changes this process's environment: undone after.
    for name in (local_backend.ASSISTANT_ENV, local_backend.OWN_KEY_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    backend.start("neurostore")
    backend.set_llm(provider="openai", key="the-pages", model="gpt")
    user = json.dumps({"provider": "anthropic", "key": "k", "model": "m"})
    state = backend.use_identity({local_backend.ASSISTANT_ENV: user})
    assert state["ready"] and state["ownKey"]
    provider = backend.repl.app.assistant.config().provider()
    assert provider.name == "openai"  # the page's key stays
    # Forbidden their own key, the user has their admin's at once.
    backend.use_identity({local_backend.ASSISTANT_ENV: user,
                          local_backend.OWN_KEY_ENV: "0"})
    assert backend.repl.app.assistant.config().provider().name == "anthropic"


def test_a_variable_the_users_identity_no_longer_has_is_taken_out(
    backend, monkeypatch
):
    """A key an admin revoked leaves the session; a variable the builder
    had is as it was."""
    monkeypatch.setenv("TEST_REVOKED", "")
    monkeypatch.delenv("TEST_REVOKED")
    monkeypatch.setenv("TEST_BUILDERS", "the-builders")
    backend.use_identity({"TEST_REVOKED": "k", "TEST_BUILDERS": "the-users"})
    assert os.environ["TEST_REVOKED"] == "k"
    backend.use_identity({"TEST_BUILDERS": "the-users-new"})
    assert "TEST_REVOKED" not in os.environ
    assert os.environ["TEST_BUILDERS"] == "the-users-new"
    backend.use_identity({"TEST_BUILDERS": None})  # withheld (a guest's)
    assert "TEST_BUILDERS" not in os.environ
    backend.use_identity({})
    assert os.environ["TEST_BUILDERS"] == "the-builders"


def test_the_same_identity_keeps_the_assistant_chosen(backend, monkeypatch):
    from neurolang_apps.utils.tui import local_backend

    for name in (local_backend.ASSISTANT_ENV, local_backend.OWN_KEY_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    user = {local_backend.ASSISTANT_ENV: json.dumps(
        {"provider": "anthropic", "key": "k", "model": "m"})}
    backend.use_identity(user)
    backend.start("neurostore")
    backend.repl.app.assistant.provider = "chosen"  # .llm provider NAME
    backend.use_identity(dict(user))
    assert backend.repl.app.assistant.provider == "chosen"
    user[local_backend.ASSISTANT_ENV] = json.dumps(
        {"provider": "openai", "key": "k2"})
    backend.use_identity(user)  # changed: the user's new one
    assert backend.repl.app.assistant.provider == "openai"
    backend.use_identity({})


def test_the_ratings_are_signed_by_the_groups_of_now(backend, tmp_path):
    backend.feedback = {"path": str(tmp_path / "f.sqlite"), "user": "ann",
                        "groups": "lab"}
    backend.start("neurostore")
    backend.use_identity({}, "lab,zoo")
    assert backend.feedback["groups"] == "lab,zoo"
    assert backend.repl._rating.groups == "lab,zoo"


def test_the_tokens_of_an_answer_say_whose_key_paid(backend, monkeypatch):
    from neurolang_apps.utils.tui import web

    backend.start("neurostore")
    monkeypatch.setattr(backend.repl, "submit", lambda text: web.Reply(
        html="", tokens={"input": 7, "output": 2}))
    assert backend.submit(".ask who?")["tokens"] == {
        "input": 7, "output": 2, "key": "server"}
    backend.set_llm(provider="anthropic", key="mine")
    assert backend.submit(".ask who?")["tokens"]["key"] == "page"
    monkeypatch.setattr(backend.repl, "submit", lambda text: web.Reply(
        html=""))
    assert backend.submit("ans(x) :- person(x, 5)")["tokens"] == {}


def test_a_build_preloads_the_text_embeddings_if_asked(tmp_path, monkeypatch):
    from neurolang_apps.utils import text_embeddings

    preloaded = []
    monkeypatch.setattr(text_embeddings, "preload", preloaded.append)

    def build(name, say):
        nl = _engine(name, say)
        nl.text_embedding_indexes = {"studies": object()}
        return nl

    nl = LocalBackend(tmp_path, build=build).build("destrieux", print)
    assert preloaded == []
    backend = LocalBackend(tmp_path, build=build, preload_embeddings=True)
    backend.start("destrieux")
    assert len(preloaded) == 1 and preloaded[0] is not nl
