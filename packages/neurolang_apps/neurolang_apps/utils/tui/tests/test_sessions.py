"""Tests of the sessions of the REPL page, each in a process of its own."""

import json
import os
import signal
import subprocess
import sys
import threading
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
    free_memory,
)


def build(name, say):
    """An engine of the registry's name with a test's relations (imported
    by name in the builder, which is started anew); neurosynth takes 2 s,
    slow 60 s."""
    say(f"Building {name}")
    if name == "broken":
        raise RuntimeError("no data")
    if name == "neurosynth":
        time.sleep(2)
    if name == "slow":
        time.sleep(60)
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    nl.add_tuple_set([(name,)], name="engine")
    # wait(s) takes s seconds: a query of 2 (short) or 60 (long).
    nl.add_tuple_set([(2,)], name="short")
    nl.add_tuple_set([(60,)], name="long")
    nl.add_symbol(wait, name="wait")
    if name in ("destrieux", "tracts"):
        # Text embeddings, which count their loads (and fail to load in
        # tracts).
        nl.text_embedding_indexes = {"docs": _Index(name == "tracts")}
        nl.add_symbol(lambda engine: nl.text_embedding_indexes["docs"].loads,
                      name="loads")
    return nl


class _Index:
    """An index of text embeddings counting its loads, its own encoder."""

    def __init__(self, fails: bool):
        self.fails = fails
        self.loads = 0
        self.encoder = self

    def build(self):
        if self.fails:
            raise ImportError("no torch")
        self.loads += 1

    def _uses_numpy(self):
        return False

    def download(self):
        pass

    def _fork_safe(self):
        pass


def wait(seconds: int) -> int:
    time.sleep(seconds)
    return seconds


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


def _ends(pid, seconds=10.0):
    """Return whether the process *pid* ends within *seconds*."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _children(pid):
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                           text=True).stdout.split()
    return {int(child) for child in found}


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
    assert _ends(pid)
    with pytest.raises(SessionEnded):
        ann.submit("ans(x) :- person(x, a)")
    assert _rows(bob.submit("ans(x) :- person(x, a)"))
    ann.start("neurostore")
    assert _rows(ann.submit("ans(x) :- person(x, a)"))


def test_a_dead_process_ends_its_session(sessions):
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    pid = ann.pid
    os.kill(pid, signal.SIGKILL)
    assert _ends(pid)
    deadline = time.monotonic() + 10
    while ann._conn is not None and time.monotonic() < deadline:
        time.sleep(0.05)
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


def test_an_engine_built_before_its_sessions_starts_them_at_once(
        sessions, capsys):
    """The engines built when the server starts, the broken one told in the
    log: a first session of neurosynth starts without a build, and another
    engine still starts."""
    sessions.prebuild(["broken", "neurosynth"])
    deadline = time.monotonic() + 30
    while "neurosynth in" not in capsys.readouterr().out:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    ann = sessions.session("ann", "1")
    listener = ann.listen()
    ann.start("neurosynth")
    messages = []
    while not listener.empty():
        messages.append(listener.get())
    assert not any("Building" in m for m in messages)
    sessions.prebuild(["neurosynth"])  # built already: not again
    bob = sessions.session("bob", "1")
    bob.start("neurostore")
    assert bob.pid is not None


def test_a_query_leaves_no_garbage(sessions):
    """A session frees what its query left, after replying; freeing is
    harmless where malloc_trim does not exist (not glibc)."""
    free_memory()
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    assert sorted(_rows(ann.submit("ans(x) :- person(x, a)"))) == [
        ["alice"], ["bob"]]
    assert _rows(ann.submit("ans(x) :- person(x, 5)")) == [["bob"]]


def test_no_process_is_left_when_the_server_ends(tmp_path):
    sessions = Sessions(tmp_path, build=build)
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    pid, builder = ann.pid, sessions._builder
    engine_builder = sessions.builders["neurostore"]
    sessions.close()
    assert _ends(pid) and _ends(builder.pid) and _ends(engine_builder)


def test_an_idle_session_ends(tmp_path):
    sessions = Sessions(tmp_path, build=build, idle=0.5)
    try:
        ann = sessions.session("ann", "1")
        ann.start("neurostore")
        pid = ann.pid
        assert _ends(pid)
        assert sessions.session("ann", "1") is not ann
    finally:
        sessions.close()


def test_the_sessions_are_at_most_so_many(tmp_path):
    sessions = Sessions(tmp_path, build=build, most=1)
    try:
        sessions.session("ann", "1").start("neurostore")
        with pytest.raises(RuntimeError, match="1 sessions"):
            sessions.session("bob", "1").start("neurostore")
    finally:
        sessions.close()


def test_a_start_given_up_leaves_no_process(sessions):
    """A page reloaded while its engine builds starts again: the process of
    the start it gave up is not the session's, and ends."""
    ann = sessions.session("ann", "1")
    first = []
    thread = threading.Thread(target=lambda: first.append(
        pytest.raises(RuntimeError, ann.start, "neurosynth")))
    thread.start()
    time.sleep(0.5)  # building
    ann.stop()
    thread.join(10)
    assert first, "the start given up still waits"
    ann.start("neurosynth")
    assert _rows(ann.submit("ans(x) :- person(x, a)"))
    builder = sessions.builders["neurosynth"]
    deadline = time.monotonic() + 10
    while _children(builder) != {ann.pid} and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _children(builder) == {ann.pid}


def test_a_session_downloads_and_rates(sessions):
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    ann.submit("ans(x) :- person(x, a)")
    file = ann.download(block=0, kind="csv")
    assert file.name.endswith(".csv") and b"alice" in file.data
    assert "ok" in ann.rate(verdict="good")


def test_a_session_loads_what_its_page_uploads(sessions):
    ann = sessions.session("ann", "t1")
    ann.start("neurosynth")
    assert ann.upload(name="pets.csv", data=b"pet\ncat\n") == {
        "name": "pets.csv", "statement": ".load pets.csv"}
    ann.submit(".load pets.csv")
    assert _rows(ann.submit("ans(x) :- pets(x)")) == [["cat"]]
    with pytest.raises(RuntimeError, match="not the name of a file"):
        ann.upload(name="../pets.csv", data=b"pet\n")


def test_uploads_are_files_of_a_folder_removed_with_the_session(tmp_path):
    """With an uploads directory (neurolang-repl-serve --uploads-dir), a
    session keeps what its page uploads in a folder of its own, removed
    when it ends (stopped, or idle for too long); the server removes, at
    its start and from time to time, the folders of no session it runs."""
    directory = tmp_path / "uploads"
    directory.mkdir()
    (directory / "999").mkdir()  # left by an earlier run of the server
    sessions = Sessions(tmp_path, build=build, uploads={
        "directory": str(directory), "most_file": 1 << 20,
        "most_all": 2 << 20})
    try:
        assert list(directory.iterdir()) == []
        ann = sessions.session("ann", "t1")
        ann.start("neurosynth")
        ann.upload(name="pets.csv", data=b"pet\ncat\n")
        folder = directory / str(ann.starts)
        assert (folder / "pets.csv").read_bytes() == b"pet\ncat\n"
        ann.submit(".load pets.csv")
        assert _rows(ann.submit("ans(x) :- pets(x)")) == [["cat"]]
        with pytest.raises(RuntimeError, match="over 1 MB"):
            ann.upload(name="big.csv", data=b"x" * ((1 << 20) + 1))
        (directory / "123").mkdir()  # of a session that crashed
        sessions._sweep_uploads()
        assert sorted(p.name for p in directory.iterdir()) == [folder.name]
        ann.end()
        assert not folder.exists()
    finally:
        sessions.close()


def test_the_admins_see_and_erase_the_uploads(tmp_path):
    """The admin console lists the folders of the uploads, whose session
    each is, its files, their sizes and ages, and erases a file, a folder
    or all of them; never a path out of the uploads directory."""
    directory = tmp_path / "uploads"
    sessions = Sessions(tmp_path, build=build, uploads={
        "directory": str(directory), "most_file": 1 << 20,
        "most_all": 2 << 20})
    try:
        ann = sessions.session("ann", "t1")
        ann.start("neurosynth")
        ann.upload(name="pets.csv", data=b"pet\ncat\n")
        ann.upload(name="toys.csv", data=b"toy\n")
        (directory / "77").mkdir()  # of no session
        (directory / "77" / "old.csv").write_bytes(b"x")
        info = sessions.uploads_info()
        assert info["directory"] == str(directory)
        assert info["size"] == 13 and info["free"] > 0
        assert info["most_file"] == 1 << 20 and info["most_all"] == 2 << 20
        folders = {f["start"]: f for f in info["folders"]}
        mine = folders[str(ann.starts)]
        assert (mine["user"], mine["tab"], mine["running"]) == (
            "ann", "t1", True)
        assert [(f["name"], f["size"]) for f in mine["files"]] == [
            ("pets.csv", 8), ("toys.csv", 4)]
        assert all(f["age"] >= 0 for f in mine["files"])
        assert folders["77"]["user"] is None
        for start, name in (("..", None), ("1/..", None),
                            (str(ann.starts), "../x"), ("x", None)):
            with pytest.raises(ValueError):
                sessions.erase_uploads(start, name)
        ann.upload(name="gone.csv", data=b"g\n")
        ann.submit(".uploads remove gone.csv")  # kept for an undo, hidden
        info = sessions.uploads_info()
        mine = {f["start"]: f for f in info["folders"]}[str(ann.starts)]
        assert [f["name"] for f in mine["files"]] == ["pets.csv", "toys.csv"]
        assert sessions.erase_uploads(str(ann.starts), "toys.csv") == 1
        assert not (directory / str(ann.starts) / "toys.csv").exists()
        assert ann.submit(".load toys.csv")["failed"]
        assert sessions.erase_uploads("77") == 1
        assert not (directory / "77").exists()
        assert sessions.erase_uploads() == 1  # all: pets.csv
        assert list(directory.iterdir()) == []
    finally:
        sessions.close()


def test_a_busy_session_is_not_ended_for_another(tmp_path):
    sessions = Sessions(tmp_path, build=build, most=1)
    try:
        ann = sessions.session("ann", "1")
        ann.busy += 1  # at work
        ann.used -= 3600
        bob = sessions.session("bob", "1")
        bob.starts = 1000  # an earlier start, given up
        with pytest.raises(RuntimeError, match="1 sessions"):
            bob.start("neurostore")
        # The start refused leaves its process not the session's.
        assert sessions._owner(bob.key, 1000) is None
        ann.busy -= 1
    finally:
        sessions.close()


def _starting(session, engine):
    """Start *session* in a thread; return what it gave, or raised, and
    the thread."""
    done = []

    def run():
        try:
            done.append(session.start(engine))
        except Exception as error:
            done.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return done, thread


def _until(condition, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_build_does_not_hold_up_the_sessions_of_an_engine_built(sessions):
    """Each engine has a builder of its own: a session of an engine built
    starts while another engine builds, its start not behind that build."""
    sessions.session("ann", "1").start("neurostore")
    building, thread = _starting(sessions.session("bob", "1"), "slow")
    _until(lambda: "slow" in sessions.builders)  # building, for 60 s
    carl = sessions.session("carl", "1")
    carl.start("neurostore")
    assert thread.is_alive() and not building  # still building
    assert _rows(carl.submit("ans(x) :- engine(x)")) == [["neurostore"]]
    assert sessions.builders["slow"] != sessions.builders["neurostore"]


def test_a_builder_that_ends_ends_its_engine_only(sessions, capsys):
    """The end of the builder of an engine (killed, out of memory) fails
    the starts it had not answered, is told in the log, and leaves the
    sessions it forked and the other engines; the next start of its engine
    has a new builder."""
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    bob = sessions.session("bob", "1")
    building, thread = _starting(bob, "slow")
    _until(lambda: "slow" in sessions.builders)
    os.kill(sessions.builders["slow"], signal.SIGKILL)
    thread.join(30)
    assert isinstance(building[0], RuntimeError)
    assert "builder of slow ended" in str(building[0])
    _until(lambda: "slow" not in sessions.builders)
    assert "The builder of the engine slow ended" in capsys.readouterr().out
    old = sessions.builders["neurostore"]
    os.kill(old, signal.SIGKILL)
    _until(lambda: "neurostore" not in sessions.builders)
    assert _rows(ann.submit("ans(x) :- engine(x)")) == [["neurostore"]]
    bob.start("neurosynth")
    carl = sessions.session("carl", "1")
    carl.start("neurostore")  # built again
    assert sessions.builders["neurostore"] != old
    assert _rows(carl.submit("ans(x) :- engine(x)")) == [["neurostore"]]
    assert not sessions.dead


def test_without_the_builder_a_start_fails(sessions):
    sessions._builder.kill()
    sessions._builder.join(10)
    with pytest.raises(RuntimeError, match="builder"):
        sessions.session("ann", "1").start("neurostore")


def _start(session, engine):
    """Start *session* in a thread; return what it gave, or raised, and
    whether it still waits after 30 s."""
    done, thread = _starting(session, engine)
    thread.join(30)
    return done, thread.is_alive()


def test_a_start_of_a_session_ended_does_not_wait(sessions):
    ann = sessions.session("ann", "t")
    with sessions._lock:  # as the reaper does
        del sessions._sessions[ann.key]
    ann.end()
    done, waiting = _start(ann, "neurostore")
    assert not waiting
    assert isinstance(done[0], SessionEnded)


def test_a_start_stopped_while_its_engine_builds_has_ended(sessions):
    """The page stops a session before it starts one (a new engine, a
    restart): a start it cuts short ended, which the server tells without
    a traceback, not an error of the server."""
    session = sessions.session("ann", "t")
    done = []

    def run():
        try:
            done.append(session.start("neurosynth"))
        except Exception as error:
            done.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.5)  # building
    session.stop()
    thread.join(30)
    assert isinstance(done[0], SessionEnded), done


def test_a_new_session_does_not_take_the_process_of_the_one_it_replaces(
        sessions):
    """The session ended while its engine builds, a new one of its key
    starts: the process forked for the first is not the second's."""
    first = sessions.session("ann", "t")
    thread = threading.Thread(target=_start, args=(first, "neurosynth"))
    thread.start()
    time.sleep(0.5)  # building
    with sessions._lock:
        del sessions._sessions[first.key]
    first.end()
    second = sessions.session("ann", "t")
    done, waiting = _start(second, "neurostore")
    thread.join(30)
    assert not waiting and not isinstance(done[0], Exception), done
    assert _rows(second.submit("ans(x) :- engine(x)")) == [["neurostore"]]


def test_the_sessions_without_a_process_take_no_room(tmp_path):
    """Of *most* sessions, those of tabs that run none do not count; a
    user's start ends their idlest session first, another's only after a
    minute unused."""
    sessions = Sessions(tmp_path, build=build, most=1)
    try:
        sessions.session("carl", "1")  # its page listens, runs nothing
        ann = sessions.session("ann", "1")
        ann.start("neurostore")
        again = sessions.session("ann", "2")
        again.start("neurostore")
        assert ann.pid is None and again.pid is not None
        bob = sessions.session("bob", "1")
        with pytest.raises(RuntimeError, match="1 sessions"):
            bob.start("neurostore")
        again.used -= 120
        bob.start("neurostore")
        assert again.pid is None and bob.pid is not None
    finally:
        sessions.close()


class _Conn:
    def send(self, message):
        raise OSError("closed")


def test_a_process_ending_fails_its_replies_only(sessions):
    ann = sessions.session("ann", "1")
    old, new = object(), object()
    ann._replies = {1: [threading.Event(), False, "", old],
                    2: [threading.Event(), False, "", new]}
    ann._fail_replies(old)
    assert ann._replies[1][0].is_set() and not ann._replies[2][0].is_set()
    ann._replies.clear()
    ann._conn = _Conn()
    with pytest.raises(OSError):
        ann.submit("ans(x) :- person(x, a)")
    assert ann._replies == {}
    ann._conn = None


def test_a_wrong_client_does_not_stop_the_sessions(sessions):
    from multiprocessing.connection import Client
    from multiprocessing import AuthenticationError

    with pytest.raises(AuthenticationError):
        Client(sessions._listener.address, authkey=b"wrong")
    with Client(sessions._listener.address,
                authkey=sessions._listener._authkey) as conn:
        conn.send(("not a hello",))
    ann = sessions.session("ann", "1")
    done, waiting = _start(ann, "neurostore")
    assert not waiting and not isinstance(done[0], Exception), done


def test_a_process_late_to_connect_is_killed(sessions):
    child = subprocess.Popen(["sleep", "60"])
    try:
        ann = sessions.session("ann", "1")
        ann.starts = 99
        ann._forked.clear()
        Sessions._late(ann, 99, child.pid)
        assert child.wait(10) == -signal.SIGKILL
        assert "did not start" in ann._failure
    finally:
        child.kill()
        child.wait()


def test_a_restart_has_the_user_as_they_are_now(tmp_path, monkeypatch):
    """A key changed by the admin reaches the user's next start."""
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    env = {"NEUROLANG_LLM_CONFIG": str(config)}
    sessions = Sessions(tmp_path, build=build,
                        identity=lambda user: (dict(env), "lab"))
    try:
        ann = sessions.session("ann", "1")
        assert not ann.start("neurostore")["assistant"]
        env["TEST_LLM_KEY"] = "k"
        assert ann.start("neurostore")["assistant"]
    finally:
        sessions.close()


def test_the_sessions_ratings_are_signed_by_their_user(tmp_path):
    from neurolang_apps.utils.llm.feedback_server import FeedbackStore

    db = tmp_path / "feedback.sqlite"
    groups = {"ann": "admin", "bob": "lab"}
    sessions = Sessions(tmp_path, build=build, feedback_db=db,
                        identity=lambda user: ({}, groups[user]))
    try:
        for user in ("ann", "bob"):
            session = sessions.session(user, "1")
            assert session.start("neurostore")["ratingNote"]
            session.submit("ans(x) :- person(x, a)")
            assert session.rate(verdict="good")["ok"]
    finally:
        sessions.close()
    signed = {(a["user"], a["groups"]) for a in FeedbackStore(db).answers()}
    assert signed == {("ann", "admin"), ("bob", "lab")}


def test_a_running_session_takes_its_users_new_assistant(
    tmp_path, monkeypatch
):
    """A key saved on the account page, or by an admin, answers at once in
    the user's open pages, which are told."""
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    config = tmp_path / "llm.yaml"  # a configuration without its key
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    env = {"NEUROLANG_LLM_CONFIG": str(config)}
    sessions = Sessions(tmp_path, build=build,
                        identity=lambda user: (dict(env), ""))
    try:
        bob = sessions.session("bob", "1")
        listener = bob.listen()
        assert not bob.start("neurostore")["assistant"]
        env["NEUROLANG_REPL_LLM"] = (
            '{"provider": "anthropic", "key": "k", "model": "m"}')
        sessions.refresh()
        deadline = time.monotonic() + 10
        told = []
        while time.monotonic() < deadline and not any(
                '"assistant"' in m and '"ready": true' in m for m in told):
            while not listener.empty():
                told.append(listener.get())
            time.sleep(0.05)
        assert any('"ready": true' in m for m in told), told
    finally:
        sessions.close()


def test_a_guests_session_has_no_key_of_the_server(tmp_path, monkeypatch):
    """With the server's key in its environment, a user's session answers
    with it, a guest's does not (the key is not even in its process)."""
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    monkeypatch.setenv("NEUROLANG_LLM_CONFIG", str(config))
    monkeypatch.setenv("TEST_LLM_KEY", "the-servers")
    guest = ({"TEST_LLM_KEY": None, "NEUROLANG_REPL_BRING_KEY": "1"},
             "guest")
    user = ({"TEST_LLM_KEY": "the-servers"}, "")  # given, as the server does
    sessions = Sessions(tmp_path, build=build, withhold=["TEST_LLM_KEY"],
                        identity=lambda name: (
                            guest if name.startswith("guest:") else user))
    try:
        assert sessions.session("ann", "1").start("neurostore")["assistant"]
        visitor = sessions.session("guest:ab12", "1")
        ready = visitor.start("neurostore")
        assert not ready["assistant"] and ready["ownKey"]
        environ = subprocess.run(
            ["ps", "eww", "-o", "command=", "-p", str(visitor.pid)],
            capture_output=True, text=True).stdout
        assert "the-servers" not in environ
    finally:
        sessions.close()


def _config(tmp_path):
    config = tmp_path / "llm.yaml"  # a configuration without its key
    config.write_text(
        "default_provider: test\nproviders:\n  test:\n    api: openai\n"
        "    base_url: http://localhost\n    api_key_env: TEST_LLM_KEY\n"
        "    model: m\n"
    )
    return {"NEUROLANG_LLM_CONFIG": str(config)}


def _told(listener, seconds):
    """The messages of *listener* within *seconds*."""
    told = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        while not listener.empty():
            told.append(listener.get())
        time.sleep(0.05)
    return told


def test_a_refresh_is_that_of_the_users_changed_and_waits_for_none(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    monkeypatch.setattr(Sessions, "REFRESH_SECONDS", 1.0)
    env = {"ann": _config(tmp_path), "bob": _config(tmp_path)}
    sessions = Sessions(tmp_path, build=build,
                        identity=lambda user: (dict(env[user]), ""))
    try:
        ann, bob = (sessions.session(user, "1") for user in ("ann", "bob"))
        for session in (ann, bob):
            session.start("neurostore")
        heard = {"ann": ann.listen(), "bob": bob.listen()}
        for user in env:
            env[user]["TEST_LLM_KEY"] = "k"
        sessions.refresh(["bob"])
        assert any('"ready": true' in m for m in _told(heard["bob"], 3))
        assert not _told(heard["ann"], 0.5)
        # A session that answers nothing holds no request of the server.
        os.kill(bob.pid, signal.SIGSTOP)
        try:
            env["bob"]["TEST_LLM_KEY"] = "k2"
            begun = time.monotonic()
            sessions.refresh(["bob"])
            assert time.monotonic() - begun < 0.5
            deadline = time.monotonic() + 10
            while bob.busy and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not bob.busy  # given up
        finally:
            os.kill(bob.pid, signal.SIGCONT)
    finally:
        sessions.close()


def test_a_change_during_a_start_reaches_the_session(tmp_path, monkeypatch):
    """The identity of a session is read when its process is forked; one
    changed while the engine builds is given to it once it is there."""
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    env = _config(tmp_path)
    sessions = Sessions(tmp_path, build=build,
                        identity=lambda user: (dict(env), "lab"))
    try:
        ann = sessions.session("ann", "1")
        ready = []
        starting = threading.Thread(
            target=lambda: ready.append(ann.start("neurosynth")))
        starting.start()
        time.sleep(0.5)  # building neurosynth: 2 s
        env["TEST_LLM_KEY"] = "k"
        sessions.refresh()  # not running yet: nothing to refresh
        starting.join(30)
        assert ready and ready[0]["assistant"]
    finally:
        sessions.close()


def test_the_activity_of_the_users_is_logged(tmp_path):
    from neurolang_apps.utils.tui.activity import ActivityLog

    sessions = Sessions(tmp_path, build=build,
                        activity_db=tmp_path / "activity.sqlite",
                        identity=lambda user: ({}, "lab"))
    try:
        ann = sessions.session("ann", "1")
        assert "logs what you run" in ann.start("neurostore")["logNote"]
        ann.submit("ans(x) :- person(x, 5)")
        ann.submit("ans(x :- (")
        ann.stop()
        with pytest.raises(SessionEnded):  # not run: not logged
            ann.submit("ans(x) :- person(x, 5)")
    finally:
        sessions.close()
    log = ActivityLog(tmp_path / "activity.sqlite")
    events = log.entries("ann")
    kinds = [e["kind"] for e in events]
    assert kinds == ["start", "submit", "submit", "end"]
    start, good, bad, end = events
    assert (start["text"], start["engine"], start["groups"]) == \
        ("neurostore", "neurostore", "lab")
    assert good["text"] == "ans(x) :- person(x, 5)"
    assert "bob" in good["response"] and not good["failed"]
    assert bad["failed"] and good["seconds"] >= 0
    assert end["text"] == "stopped"
    if os.path.exists("/proc/self/smaps_rollup"):  # Linux
        assert good["cpu"] is not None and good["memory"] > 0
    [totals] = log.totals()
    assert (totals["user"], totals["inputs"], totals["failed"]) == \
        ("ann", 2, 1)
    if os.path.exists("/proc/self/smaps_rollup"):  # the inputs' CPU only
        # SQL's ROUND goes away from zero on a half (0.25 is 0.3), Python's
        # round to even (0.2): the total is that of the inputs, to the 0.05
        # that rounding to a tenth moves it.
        assert totals["cpu"] == pytest.approx(
            good["cpu"] + bad["cpu"], abs=0.051
        )


def test_the_own_memory_of_a_fork_is_apart_from_the_shared():
    from neurolang_apps.utils.tui.sessions import process_usage

    if process_usage(os.getpid()) is None:
        pytest.skip("the usage of a process is read on Linux")
    # The fork is of a fresh interpreter, as the builder is: that of the
    # test process, large after all the packages' tests, fails (ENOMEM).
    script = (
        "import os, sys\n"
        "shared = bytearray(os.urandom(1 << 20)) * 100  # 100 MB\n"
        "if os.fork() == 0:\n"
        "    own = bytearray(os.urandom(1 << 20)) * 50  # 50 MB of its own\n"
        "    print(os.getpid(), flush=True)\n"
        "    sys.stdin.read()\n"
        "    os._exit(0)\n"
        "os.wait()\n"
    )
    with subprocess.Popen([sys.executable, "-c", script], text=True,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
        usage = process_usage(int(p.stdout.readline()))
        p.stdin.close()
    assert 45 < usage["private"] < 80  # its 50 MB, not the parent's 100
    assert usage["rss"] > usage["private"] + 80  # the rss counts them


def test_the_queries_running_at_once_are_at_most_so_many(tmp_path):
    sessions = Sessions(tmp_path, build=build, running=1)
    try:
        ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
        for session in (ann, bob):
            session.start("neurostore")
        told = bob.listen()
        slow = threading.Thread(target=ann.submit,
                                args=("ans(y) :- short(s), (y == wait(s))",))
        slow.start()
        time.sleep(0.5)
        began = time.monotonic()
        assert _rows(bob.submit("ans(x) :- person(x, 5)")) == [["bob"]]
        assert time.monotonic() - began > 1  # it waited for ann's
        slow.join()
        messages = []
        while not told.empty():
            messages.append(told.get())
        assert any("Waiting: 1 queries run" in m for m in messages)
    finally:
        sessions.close()


def test_a_query_running_too_long_ends_its_session(tmp_path):
    sessions = Sessions(tmp_path, build=build, query_seconds=1)
    try:
        ann = sessions.session("ann", "1")
        ann.start("neurostore")
        pid = ann.pid
        with pytest.raises(RuntimeError, match="ran longer than this "
                           "server allows"):
            ann.submit("ans(y) :- long(s), (y == wait(s))")
        assert ann.pid is None and _ends(pid)
        with pytest.raises(SessionEnded):
            ann.submit("ans(x) :- person(x, 5)")
        ann.start("neurostore")  # its slot is free again
        assert _rows(ann.submit("ans(x) :- person(x, 5)")) == [["bob"]]
    finally:
        sessions.close()


def test_a_query_waiting_does_not_run_in_its_session_restarted(tmp_path):
    sessions = Sessions(tmp_path, build=build, running=1)
    try:
        ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
        for session in (ann, bob):
            session.start("neurostore")
        slow = threading.Thread(target=ann.submit,
                                args=("ans(y) :- short(s), (y == wait(s))",))
        slow.start()
        time.sleep(0.3)
        waited = []

        def asked():
            try:
                bob.submit("old(x) :- person(x, 5)")
                waited.append("ran")
            except SessionEnded:
                waited.append("ended")

        waiting = threading.Thread(target=asked)
        waiting.start()
        time.sleep(0.3)
        bob.stop()
        bob.start("neurostore")  # a start takes no slot
        slow.join()
        waiting.join()
        assert waited == ["ended"]
        # Its rule is not in the new process.
        assert _rows(bob.submit("ans(x) :- old(x)")) == []
    finally:
        sessions.close()


def test_the_tokens_of_an_answer_are_logged(tmp_path):
    from neurolang_apps.utils.tui.activity import ActivityLog

    sessions = Sessions(tmp_path, build=build,
                        activity_db=tmp_path / "activity.sqlite")
    try:
        ann = sessions.session("ann", "1")
        ann.start("neurostore")
        reply = {"html": "", "failed": False, "tokens": {
            "input": 10, "output": 3, "calls": 1, "provider": "albert",
            "model": "m", "key": "server"}}
        ann._log_submit(".ask who?", reply, 1.0, None)
    finally:
        sessions.close()
    [answer] = [e for e in ActivityLog(tmp_path / "activity.sqlite"
                                       ).entries("ann")
                if e["kind"] == "submit"]
    assert (answer["tokens_in"], answer["tokens_out"], answer["llm"],
            answer["llm_key"]) == (10, 3, "albert/m", "server")


def test_the_text_embeddings_are_loaded_once_before_the_sessions_fork(
    sessions,
):
    for user in ("ann", "bob"):
        session = sessions.session(user, "1")
        session.start("destrieux")
        assert _rows(session.submit(
            "ans(n) :- engine(e), (n == loads(e))")) == [[1]]
    told = sessions.session("ann", "2")
    listener = told.listen()
    told.start("tracts")  # its embeddings fail: the start does not
    assert _rows(told.submit("ans(n) :- engine(e), (n == loads(e))")) == [[0]]
    messages = []
    while not listener.empty():
        messages.append(listener.get())
    assert any("Loading the text embeddings" in m for m in messages)


def test_the_text_embeddings_may_be_left_to_the_sessions(tmp_path):
    sessions = Sessions(tmp_path, build=build, preload_embeddings=False)
    try:
        ann = sessions.session("ann", "1")
        ann.start("destrieux")
        assert _rows(ann.submit(
            "ans(n) :- engine(e), (n == loads(e))")) == [[0]]
    finally:
        sessions.close()


def _mcp(session, tmp_path, method, params=None):
    """The answer of the MCP server of *session* to a JSON-RPC request."""
    status, headers, body = session.mcp(
        verb="POST", path="/mcp", headers={
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "host": "repl.example.org",
            "mcp-protocol-version": "2025-06-18",
        },
        body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                         "params": params or {}}).encode(),
        base_url="https://repl.example.org/mcp/files/",
        results_dir=str(tmp_path / "mcp"),
    )
    assert status == 200, body
    return json.loads(body)["result"]


def test_an_mcp_client_has_a_session_in_plain_text(sessions, tmp_path):
    pytest.importorskip("mcp")
    session = sessions.session("ann", "mcp")
    session.start("neurostore", text=True)
    assert "NeuroLang" in _mcp(session, tmp_path, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    })["instructions"]
    text = _mcp(session, tmp_path, "tools/call", {
        "name": "repl", "arguments": {"text": "ans(x) :- person(x, 5)"},
    })["content"][0]["text"]
    assert "bob" in text and "<span" not in text  # text, not HTML
    answer = _mcp(session, tmp_path, "tools/call", {
        "name": "final_answer", "arguments": {
            "query": "ans(x) :- person(x, 5)", "explanation": "Bob.",
            "comments": ["the person of 5"],
        },
    })
    links = answer["structuredContent"]["links"]
    assert links["rows"].startswith("https://repl.example.org/mcp/files/")
    assert list((tmp_path / "mcp").glob("*/a1/rows.csv"))
