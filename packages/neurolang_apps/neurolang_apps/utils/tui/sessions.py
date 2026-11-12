"""
Sessions of the REPL page, each in a process of its own, for the users of
``neurolang-repl-serve --local --users FILE``.

The server's process answers the pages; a *builder* process, started
fresh (``spawn``) before the server answers anything, builds each engine
once, then forks a process per session from it: the sessions share the
memory of the engine's data, as the pages of a fork do until they are
written, and each has its own rules, AI assistant and requests. A session
is that of a tab of a user: its key is (user, the tab's id). Its process
talks to the server through a socket of this machine; a stop kills it (the
page starts a session again), as does a session idle for too long.

:class:`Sessions` is what the server holds; its :meth:`Sessions.session`
is a :class:`Session`, which answers as a
:class:`~neurolang_apps.utils.tui.local_backend.LocalBackend` does.
"""

import gc
import itertools
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener
from typing import Callable, Optional

from neurolang_apps.utils.tui.local_backend import LocalBackend


class SessionEnded(RuntimeError):
    """The process of a session is gone: the page starts it again."""


# -- In the builder's process ---------------------------------------------

def _builder(conn, address, authkey, options: dict, build) -> None:
    """Build the engines asked for, once each, and fork a session's
    process from them for each ``("fork", key, engine, env)`` of *conn*;
    end when the server does."""
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # the sessions reap
    backend = LocalBackend(build=build, **options)
    engines = {}
    while True:
        try:
            _, key, engine, env = conn.recv()
        except (EOFError, OSError):
            return
        try:
            if engine not in engines:
                engines[engine] = backend._build(
                    engine, lambda text: conn.send(("status", key, text))
                )
                gc.freeze()  # a session's process then writes fewer pages
                _warn_of_threads(engine)
        except Exception as error:
            traceback.print_exc()
            conn.send(("failed", key, f"{type(error).__name__}: {error}"))
            continue
        pid = os.fork()
        if pid == 0:
            conn.close()
            code = 1
            try:
                _session(address, authkey, key, env, options,
                         engines[engine])
                code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)
        conn.send(("forked", key, pid))


def _warn_of_threads(engine: str) -> None:
    """Warn if building *engine* left threads running in this process (on
    Linux, which lists them): a fork copies none of them, and a lock one of
    them holds stays held in the session's process."""
    try:
        threads = len(os.listdir("/proc/self/task"))
    except OSError:
        return
    if threads > 1:
        print(f"Building {engine} left {threads - 1} threads in the "
              "builder: a session's process may hang", file=sys.stderr,
              flush=True)


def _session(address, authkey, key, env: dict, options: dict, nl) -> None:
    """Serve the session *key* on the engine *nl*, with the variables
    *env* (the key of its user's assistant), until the server ends it."""
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.environ.update(env)
    backend = LocalBackend(build=lambda name, say: nl, **options)
    conn = Client(address, authkey=authkey)
    sending = threading.Lock()

    def send(message):
        with sending:
            conn.send(message)

    send(("hello", key, os.getpid()))
    events = backend.listen()

    def forward():
        while True:
            send(("event", events.get()))

    threading.Thread(target=forward, daemon=True).start()

    def call(request, method, arguments):
        try:
            value = getattr(backend, method)(**arguments)
            send(("reply", request, True, value))
        except BaseException as error:
            send(("reply", request, False,
                  f"{type(error).__name__}: {error}"))

    while True:
        try:
            _, request, method, arguments = conn.recv()
        except (EOFError, OSError):
            return
        threading.Thread(target=call, args=(request, method, arguments),
                         daemon=True).start()


# -- In the server's process ----------------------------------------------

class Session:
    """
    The session of a tab of a user, in its process: what the server's
    handler asks it, as of a
    :class:`~neurolang_apps.utils.tui.local_backend.LocalBackend`.
    """

    def __init__(self, sessions: "Sessions", key: tuple, env: dict):
        self.sessions = sessions
        self.key = key
        self.env = env
        self.used = time.monotonic()
        self.busy = 0  # the requests at work, which an idle end waits for
        self.pid: Optional[int] = None
        self._conn = None
        self._sending = threading.Lock()
        self._starting = threading.Lock()
        self._forked = threading.Event()
        self._failure: Optional[str] = None
        self._replies = {}
        self._requests = itertools.count()
        self._listeners = []
        self._listeners_lock = threading.Lock()

    # What the session tells while it works, to the pages listening: they
    # keep listening while its process changes.

    def listen(self):
        import queue

        listener = queue.Queue()
        with self._listeners_lock:
            self._listeners.append(listener)
        return listener

    def forget(self, listener) -> None:
        with self._listeners_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit_text(self, text: str) -> None:
        with self._listeners_lock:
            for listener in self._listeners:
                listener.put(text)

    # The requests of the page.

    def start(self, engine: str) -> dict:
        """Start the session anew, in a new process, on *engine*."""
        with self._starting:
            self.end()
            self._forked.clear()
            self._failure = None
            self.busy += 1
            try:
                self.sessions._fork(self, engine)
                self._forked.wait()
                if self._failure:
                    raise RuntimeError(self._failure)
                return self._call("start", engine=engine)
            finally:
                self.busy -= 1
                self.used = time.monotonic()

    def submit(self, text: str) -> dict:
        return self._call("submit", text=text)

    def set_llm(self, **options) -> None:
        return self._call("set_llm", **options)

    def rate(self, **options) -> dict:
        return self._call("rate", **options)

    def download(self, **options):
        return self._call("download", **options)

    def stop(self) -> bool:
        """End the session's process, which stops what it does: the page
        starts the session again."""
        running = self.pid is not None
        self.end()
        return running

    def end(self) -> None:
        """Kill the session's process, if any; what waits on it fails."""
        pid, conn, self.pid, self._conn = self.pid, self._conn, None, None
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if conn is not None:
            conn.close()
        self._fail_replies()
        if not self._forked.is_set():  # a start waiting for its process
            self._failed("The session ended")

    # Its process.

    def _call(self, method: str, **arguments):
        conn = self._conn
        if conn is None:
            raise SessionEnded(
                "This session ended (stopped, or idle for too long): "
                "restart it"
            )
        request = next(self._requests)
        done = threading.Event()
        self._replies[request] = [done, False, "The session ended"]
        self.busy += 1
        try:
            with self._sending:
                conn.send(("call", request, method, arguments))
            done.wait()
        finally:
            self.busy -= 1
            self.used = time.monotonic()
        _, ok, value = self._replies.pop(request)
        if not ok:
            raise RuntimeError(value)
        return value

    def _attach(self, conn, pid: int) -> None:
        self._conn, self.pid = conn, pid
        self._forked.set()
        threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn) -> None:
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            if message[0] == "event":
                self.emit_text(message[1])
            else:
                _, request, ok, value = message
                reply = self._replies.get(request)
                if reply is not None:
                    reply[1:] = [ok, value]
                    reply[0].set()
        if self._conn is conn:  # its process died
            self._conn = self.pid = None
        self._fail_replies()

    def _fail_replies(self) -> None:
        for done, *_ in list(self._replies.values()):
            done.set()

    def _failed(self, message: str) -> None:
        self._failure = message
        self._forked.set()


class Sessions:
    """
    The sessions of the users of the server, each in a process forked
    from the builder's, at most *most* at once; one unused for *idle*
    seconds ends. *env* gives, by user, the variables of their sessions
    (the key of their AI assistant). The other options are those of a
    :class:`~neurolang_apps.utils.tui.local_backend.LocalBackend`; *build*
    must be importable by name, as the builder is started anew.
    """

    def __init__(
        self, data_dir, resolution: float = 2.0, light: bool = False,
        feedback: Optional[tuple] = None, env: Optional[dict] = None,
        most: int = 8, idle: float = 7200.0,
        build: Optional[Callable] = None,
    ):
        self.env = env or {}
        self.most = most
        self.idle = idle
        self._sessions = {}
        self._lock = threading.Lock()
        self._forking = threading.Lock()
        authkey = os.urandom(32)
        self._listener = Listener(family="AF_UNIX", authkey=authkey)
        context = multiprocessing.get_context("spawn")
        self._builder_conn, theirs = context.Pipe()
        options = {"data_dir": str(data_dir), "resolution": resolution,
                   "light": light, "feedback": feedback}
        self._builder = context.Process(
            target=_builder, daemon=True, name="neurolang-repl-builder",
            args=(theirs, self._listener.address, authkey, options, build),
        )
        self._builder.start()
        theirs.close()
        self._closed = False
        for target in (self._accept, self._read_builder, self._reap):
            threading.Thread(target=target, daemon=True).start()

    def session(self, user: str, tab: str) -> Session:
        """Return the session of the tab *tab* of *user*, new if need be."""
        key = (user, tab)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                if len(self._sessions) >= self.most:
                    self._end_idlest()
                session = self._sessions[key] = Session(
                    self, key, self.env.get(user, {})
                )
            session.used = time.monotonic()
            return session

    def close(self) -> None:
        """End every session and the builder."""
        self._closed = True
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.end()
        self._builder_conn.close()
        self._builder.join(5)
        if self._builder.is_alive():
            self._builder.kill()
        self._listener.close()

    def _end_idlest(self) -> None:
        idlest = min(self._sessions.values(), key=lambda s: s.used)
        if idlest.pid is not None and \
                time.monotonic() - idlest.used < 60:
            raise RuntimeError(
                f"The server has {self.most} sessions at work: close a tab "
                "or wait"
            )
        idlest.end()
        del self._sessions[idlest.key]

    def _fork(self, session: Session, engine: str) -> None:
        with self._forking:
            self._builder_conn.send(
                ("fork", session.key, engine, session.env)
            )

    def _owner(self, key) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(tuple(key))

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn = self._listener.accept()
                _, key, pid = conn.recv()
            except (OSError, EOFError):
                continue
            session = self._owner(key)
            if session is None:
                conn.close()
                os.kill(pid, signal.SIGKILL)
            else:
                session._attach(conn, pid)

    def _read_builder(self) -> None:
        while True:
            try:
                kind, key, value = self._builder_conn.recv()
            except (EOFError, OSError):
                return
            session = self._owner(key)
            if session is None:
                continue
            if kind == "status":
                session.emit_text(json.dumps(
                    {"type": "status", "message": value + "…"}
                ))
            elif kind == "failed":
                session._failed(value)
            elif kind == "forked":  # its process says hello at once
                threading.Timer(self.HELLO_SECONDS, self._late, args=(
                    session, value)).start()

    #: How long a session's process may take to connect, in seconds.
    HELLO_SECONDS = 60.0

    @staticmethod
    def _late(session: Session, pid: int) -> None:
        if not session._forked.is_set():
            session._failed(f"The process of the session ({pid}) did not "
                            "start: see the server's log")

    def _reap(self) -> None:
        while not self._closed:
            time.sleep(min(60.0, self.idle / 2))
            now = time.monotonic()
            with self._lock:
                idle = [s for s in self._sessions.values()
                        if not s.busy and now - s.used > self.idle]
                for session in idle:
                    del self._sessions[session.key]
            for session in idle:
                session.end()
