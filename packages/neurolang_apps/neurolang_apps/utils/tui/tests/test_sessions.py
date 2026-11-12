"""Tests of the sessions of the REPL page, each in a process of its own."""

import os
import signal
import time

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

if not hasattr(os, "fork"):
    pytest.skip("sessions fork their processes", allow_module_level=True)

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.tui.sessions import (  # noqa: E402
    SessionEnded,
    Sessions,
)


def build(name, say):
    """An engine of the registry's name with a test's relations (imported
    by name in the builder, which is started anew)."""
    say(f"Building {name}")
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    return nl


@pytest.fixture
def sessions(tmp_path):
    # Ann's sessions have the key of an assistant, Bob's its configuration
    # only.
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    sessions = Sessions(tmp_path, build=build, env={
        "ann": {"NEUROLANG_LLM_CONFIG": str(config), "TEST_LLM_KEY": "k"},
        "bob": {"NEUROLANG_LLM_CONFIG": str(config)},
    })
    yield sessions
    sessions.close()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _rows(reply):
    return [row for table in reply["tables"] for row in table["rows"]]


def test_each_session_has_its_own_rules(sessions):
    ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
    for session in (ann, bob):
        assert "neurostore" in session.start("neurostore")["banner"]
    assert ann.pid != bob.pid
    ann.submit("old(x) :- person(x, a), (a > 4)")
    assert _rows(ann.submit("ans(x) :- old(x)")) == [["bob"]]
    assert _rows(bob.submit("ans(x) :- old(x)")) == []  # no rule old


def test_a_session_is_that_of_a_user_and_a_tab(sessions):
    assert sessions.session("ann", "1") is sessions.session("ann", "1")
    assert sessions.session("ann", "1") is not sessions.session("bob", "1")
    assert sessions.session("ann", "1") is not sessions.session("ann", "2")


def test_a_session_has_the_variables_of_its_user(sessions, monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
    assert ann.start("neurostore")["assistant"]
    assert not bob.start("neurostore")["assistant"]


def test_a_stop_ends_one_session_only(sessions):
    ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
    for session in (ann, bob):
        session.start("neurostore")
    pid = ann.pid
    assert ann.stop()
    time.sleep(0.2)
    assert not _alive(pid)
    with pytest.raises(SessionEnded):
        ann.submit("ans(x) :- person(x, a)")
    assert _rows(bob.submit("ans(x) :- person(x, a)"))
    ann.start("neurostore")
    assert _rows(ann.submit("ans(x) :- person(x, a)"))


def test_a_dead_process_ends_its_session(sessions):
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    os.kill(ann.pid, signal.SIGKILL)
    time.sleep(0.5)
    with pytest.raises(SessionEnded):
        ann.submit("ans(x) :- person(x, a)")


def test_the_steps_of_a_start_are_told(sessions):
    ann = sessions.session("ann", "1")
    listener = ann.listen()
    ann.start("neurostore")
    messages = []
    while not listener.empty():
        messages.append(listener.get())
    assert any("Building neurostore" in m for m in messages)


def test_no_process_is_left_when_the_server_ends(tmp_path):
    sessions = Sessions(tmp_path, build=build)
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    pid, builder = ann.pid, sessions._builder
    sessions.close()
    time.sleep(0.5)
    assert not _alive(pid) and not builder.is_alive()


def test_an_idle_session_ends(tmp_path):
    sessions = Sessions(tmp_path, build=build, idle=0.5)
    try:
        ann = sessions.session("ann", "1")
        ann.start("neurostore")
        pid = ann.pid
        time.sleep(1.5)
        assert not _alive(pid)
        assert sessions.session("ann", "1") is not ann
    finally:
        sessions.close()


def test_the_sessions_are_at_most_so_many(tmp_path):
    sessions = Sessions(tmp_path, build=build, most=1)
    try:
        sessions.session("ann", "1").start("neurostore")
        with pytest.raises(RuntimeError, match="1 sessions"):
            sessions.session("bob", "1")
    finally:
        sessions.close()
