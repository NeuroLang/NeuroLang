"""Tests of the sessions of the REPL page, each in a process of its own."""

import os
import signal
import subprocess
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
)


def build(name, say):
    """An engine of the registry's name with a test's relations (imported
    by name in the builder, which is started anew); neurosynth takes 2 s."""
    say(f"Building {name}")
    if name == "neurosynth":
        time.sleep(2)
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


def test_no_process_is_left_when_the_server_ends(tmp_path):
    sessions = Sessions(tmp_path, build=build)
    ann = sessions.session("ann", "1")
    ann.start("neurostore")
    pid, builder = ann.pid, sessions._builder
    sessions.close()
    assert _ends(pid) and _ends(builder.pid)


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
    builder = sessions._builder.pid
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


def test_without_the_builder_a_start_fails(sessions):
    sessions._builder.kill()
    sessions._builder.join(10)
    with pytest.raises(RuntimeError, match="builder"):
        sessions.session("ann", "1").start("neurostore")


def _start(session, engine):
    """Start *session* in a thread; return what it gave, or raised, and
    whether it still waits after 30 s."""
    done = []

    def run():
        try:
            done.append(session.start(engine))
        except Exception as error:
            done.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
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
        assert totals["cpu"] == round(good["cpu"] + bad["cpu"], 1)


def test_the_own_memory_of_a_fork_is_apart_from_the_shared():
    from neurolang_apps.utils.tui.sessions import process_usage

    if process_usage(os.getpid()) is None:
        pytest.skip("the usage of a process is read on Linux")
    shared = bytearray(os.urandom(1 << 20)) * 100  # 100 MB, of the parent
    ready, done = os.pipe(), os.pipe()
    pid = os.fork()
    if pid == 0:
        own = bytearray(os.urandom(1 << 20)) * 50  # 50 MB of its own
        os.write(ready[1], b"x")
        os.read(done[0], 1)
        os._exit(len(own) and len(shared) and 0)
    try:
        os.read(ready[0], 1)
        usage = process_usage(pid)
    finally:
        os.write(done[1], b"x")
        os.waitpid(pid, 0)
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
