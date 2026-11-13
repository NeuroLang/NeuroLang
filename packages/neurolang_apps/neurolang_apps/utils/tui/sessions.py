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
    """The process of a session is gone: the page starts it again, its
    rules given back, and runs again what was asked (the server answers
    409, ``{"ended": true}``)."""


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def process_usage(pid: int, memory: bool = True) -> Optional[dict]:
    """
    Return the CPU seconds of the process *pid* and, if *memory*, its
    memory in MB, on Linux (None elsewhere, or if it is gone): its resident
    memory (``rss``), which counts the pages a session shares with the
    builder, its proportional share of them (``pss``) and the memory of its
    own (``private``), which a session adds to the machine's. The memory
    takes longer to read, the larger the process.
    """
    try:
        with open(f"/proc/{pid}/stat") as file:
            # Its name, in parentheses, may hold spaces: the fields follow.
            fields = file.read().rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])  # utime, stime
        cpu = {"pid": pid, "cpu": round(ticks / os.sysconf("SC_CLK_TCK"), 2)}
        if not memory:
            return cpu
        kb = {}
        with open(f"/proc/{pid}/smaps_rollup") as file:
            for line in file:
                name, _, value = line.partition(":")
                value = value.split()
                if len(value) == 2 and value[1] == "kB":
                    kb[name] = int(value[0])
    except (OSError, IndexError, ValueError):
        return None
    return {
        **cpu,
        "rss": round(kb.get("Rss", 0) / 1024, 1),
        "pss": round(kb.get("Pss", 0) / 1024, 1),
        "private": round((kb.get("Private_Clean", 0)
                          + kb.get("Private_Dirty", 0)) / 1024, 1),
    }


# -- In the builder's process ---------------------------------------------

def _builder(conn, address, authkey, options: dict, build) -> None:
    """Build the engines asked for, once each, and fork a session's
    process from them for each ``("fork", key, start, engine, env)`` of
    *conn* (*start* numbering each start of a session once, *env* the
    variables of its user, *feedback* where its ratings go); end when the
    server does."""
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # the sessions reap
    backend = LocalBackend(build=build, **options)
    engines = {}
    while True:
        try:
            _, key, start, engine, env, feedback = conn.recv()
        except (EOFError, OSError):
            # The server ended: so does the builder, at once, rather than
            # wait for threads that building an engine may have left.
            os._exit(0)
        try:
            if engine not in engines:
                engines[engine] = backend._build(
                    engine,
                    lambda text: conn.send(("status", key, start, text)),
                )
                gc.freeze()  # a session's process then writes fewer pages
                _warn_of_threads(engine)
        except Exception as error:
            traceback.print_exc()
            conn.send(("failed", key, start,
                       f"{type(error).__name__}: {error}"))
            continue
        pid = os.fork()
        if pid == 0:
            conn.close()
            code = 1
            try:
                _session(address, authkey, key, start, env,
                         {**options, "feedback": feedback or
                          options["feedback"]}, engines[engine])
                code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)
        conn.send(("forked", key, start, pid))


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


def _session(address, authkey, key, start: int, env: dict, options: dict,
             nl) -> None:
    """Serve the session *key* on the engine *nl*, with the variables
    *env* (the key of its user's assistant), until the server ends it."""
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    backend = LocalBackend(build=lambda name, say: nl, **options)
    backend.use_identity(env)
    conn = Client(address, authkey=authkey)
    sending = threading.Lock()

    def send(message):
        with sending:
            conn.send(message)

    send(("hello", key, start, os.getpid()))
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

    def __init__(self, sessions: "Sessions", key: tuple):
        self.sessions = sessions
        self.key = key
        self.used = time.monotonic()
        self.busy = 0  # the requests at work, which an idle end waits for
        # The number of the latest start, unique among the sessions: a
        # process, or a message of the builder, of another is not this one's.
        self.starts = 0
        self.pid: Optional[int] = None
        self.engine = ""
        # The latest usage of its process (process_usage), and the most
        # private memory of its processes, in MB.
        self.usage: Optional[dict] = None
        self.peak = 0.0
        # The identity of its user its process has: (variables, groups).
        self.identity = None
        self._conn = None
        self._sending = threading.Lock()
        self._starting = threading.Lock()
        self._lock = threading.Lock()  # a process attaches, or it ends
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
            self.end("restarted")
            self.engine = engine
            self._forked.clear()
            self._failure = None
            self.busy += 1
            try:
                self.sessions._fork(self, engine)
                # The build may take minutes; the session may end meanwhile.
                while not self._forked.wait(5):
                    if self.sessions.dead or not self.sessions._owns(self):
                        self.end("ended while it started")
                if self._failure:
                    raise RuntimeError(self._failure)
                identity = self.sessions.identity(self.key[0])
                if identity != self.identity:  # changed while it was forked
                    self._call("use_identity", env=identity[0],
                               groups=identity[1])
                    self.identity = identity
                ready = self._call("start", engine=engine)
                self.sessions._log(self, "start", text=engine)
                if self.sessions.activity is not None:
                    from neurolang_apps.utils.tui.activity import LOG_NOTE

                    ready["logNote"] = LOG_NOTE
                return ready
            finally:
                self.busy -= 1
                self.used = time.monotonic()

    def submit(self, text: str) -> dict:
        pid = self.pid
        before = process_usage(pid, memory=False) if pid else None
        began = time.monotonic()
        reply = {"html": "", "failed": True}
        try:
            reply = self._run("submit", text=text)
            return reply
        except SessionEnded:
            reply = None  # not run: the page brings the session back
            raise
        except Exception as error:
            reply = {"html": f"{type(error).__name__}: {error}",
                     "failed": True}
            raise
        finally:
            self._log_submit(text, reply, time.monotonic() - began, before)

    def _sample(self) -> Optional[dict]:
        """Take the usage of the session's process now, if it runs."""
        pid = self.pid
        usage = process_usage(pid) if pid is not None else None
        if usage is not None:
            self.usage = usage
            self.peak = max(self.peak, usage["private"])
        return usage

    def _log_submit(self, text: str, reply: Optional[dict], seconds: float,
                    before: Optional[dict]) -> None:
        if self.sessions.activity is None or reply is None:
            return
        from neurolang_apps.utils.tui.activity import response_text

        # Its process's, after: as it ended (end), if it did.
        after = self._sample() or self.usage
        cpu = memory = None
        if before and after and before["pid"] == after["pid"]:
            cpu = round(after["cpu"] - before["cpu"], 2)
            memory = after["private"]
        self.sessions._log(
            self, "submit", text=text,
            response=response_text(reply),
            failed=bool(reply.get("failed")), seconds=round(seconds, 3),
            cpu=cpu, memory=memory,
        )

    def set_llm(self, **options) -> None:
        return self._call("set_llm", **options)

    def rate(self, **options) -> dict:
        return self._call("rate", **options)

    def download(self, **options):
        return self._run("download", **options)

    def _run(self, method: str, **arguments):
        """Return what the session's process answers to *method*, a query,
        once fewer than the server's most queries run (meanwhile the page
        is told it waits); a query running longer than the server allows
        ends the session's process, which stops it."""
        sessions = self.sessions
        conn = self._conn
        self.busy += 1  # waiting, it is at work: no idle end
        try:
            waited = not sessions._slots.acquire(blocking=False)
            if waited:
                self._say(f"Waiting: {sessions.running} queries run on "
                          "this server")
                while not sessions._slots.acquire(timeout=1):
                    if self._conn is not conn:
                        raise SessionEnded("This session ended while its "
                                           "query waited: restart it")
            # Not in the process of a restart meanwhile (a stop, then a
            # start), which is not the one it was asked of.
            if self._conn is not conn:
                sessions._slots.release()
                raise SessionEnded("This session ended while its query "
                                   "waited: restart it")
            if waited:
                self._say("Running")
            try:
                return self._call(method, timeout=sessions.query_seconds,
                                  **arguments)
            except TimeoutError:
                self.end("ran too long")
                raise RuntimeError(
                    "The query ran longer than this server allows "
                    f"({sessions.query_seconds / 60:g} minutes): it was "
                    "stopped, with its session, whose rules come back at the "
                    "next input"
                ) from None
            finally:
                sessions._slots.release()
        finally:
            self.busy -= 1

    def _say(self, text: str) -> None:
        """Tell the pages of the session a step, as their status."""
        self.emit_text(json.dumps({"type": "status", "message": text + "…"}))

    def stop(self) -> bool:
        """End the session's process, which stops what it does: the page
        starts the session again."""
        running = self.pid is not None
        self.end("stopped")
        return running

    def end(self, reason: str = "ended") -> None:
        """Kill the session's process, if any, logging why, *reason*; what
        waits on it fails."""
        with self._lock:
            self._sample()  # its last usage, which the log of a query takes
            pid, conn, self.pid, self._conn = self.pid, self._conn, None, None
            if pid is not None:
                _kill(pid)
            if conn is not None:
                conn.close()
                self._fail_replies(conn)
            if not self._forked.is_set():  # a start waiting for its process
                self._failed("The session ended")
        if pid is not None:
            usage = self.usage if self.usage and self.usage["pid"] == pid \
                else {}
            self.sessions._log(self, "end", text=reason,
                               cpu=usage.get("cpu"), memory=self.peak or None)

    # Its process.

    def _call(self, method: str, *, timeout: Optional[float] = None,
              **arguments):
        """Return what the session's process answers to *method* with
        *arguments*; raise TimeoutError past *timeout* seconds, if given."""
        conn = self._conn
        if conn is None:
            raise SessionEnded(
                "This session ended (stopped, or idle for too long): "
                "restart it"
            )
        request = next(self._requests)
        done = threading.Event()
        # The reply, of this process only: another's end does not fail it.
        reply = self._replies[request] = [done, False, "The session ended",
                                          conn]
        self.busy += 1
        try:
            with self._sending:
                conn.send(("call", request, method, arguments))
            if not done.wait(timeout):
                raise TimeoutError(f"The session did not answer {method}")
        finally:
            self._replies.pop(request, None)
            self.busy -= 1
            self.used = time.monotonic()
        _, ok, value, _ = reply
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
                    reply[1:3] = [ok, value]
                    reply[0].set()
        if self._conn is conn:  # its process died
            self._conn = self.pid = None
        self._fail_replies(conn)

    def _fail_replies(self, conn) -> None:
        """Fail the replies awaited from the process of *conn*."""
        for done, _, _, of in list(self._replies.values()):
            if of is conn:
                done.set()

    def _failed(self, message: str) -> None:
        self._failure = message
        self._forked.set()


class Sessions:
    """
    The sessions of the users of the server, each in a process forked
    from the builder's, at most *most* running at once; one unused for *idle*
    seconds ends. *identity*, ``identity(user)``, gives the variables of
    the sessions of *user* (the key of their AI assistant) and their groups,
    at each start; by default those of *env*, by user, and no groups. With
    *feedback_db*, the sessions store their ratings in that database,
    signed by their user and groups. With *activity_db*, the inputs of the
    sessions, their results and the resources they took are logged there
    (:class:`~neurolang_apps.utils.tui.activity.ActivityLog`). The
    variables *withhold* (the keys of
    the server) are not in the sessions' environment unless their identity
    gives them. The other options are those of a
    :class:`~neurolang_apps.utils.tui.local_backend.LocalBackend`. At most
    *running* queries (and downloads) run at once (by default, as many as
    the machine has CPUs), the others waiting; one running longer than
    *query_seconds*, if given, ends its session's process. *build* must be
    importable by name, as the builder is started anew.
    """

    def __init__(
        self, data_dir, resolution: float = 2.0, light: bool = False,
        feedback: Optional[tuple] = None, env: Optional[dict] = None,
        most: int = 8, idle: float = 7200.0,
        build: Optional[Callable] = None,
        identity: Optional[Callable] = None, feedback_db=None,
        withhold=(), activity_db=None, running: Optional[int] = None,
        query_seconds: Optional[float] = None,
    ):
        env = env or {}
        self.identity = identity or (lambda user: (env.get(user, {}), ""))
        self.feedback_db = feedback_db
        self.activity = None
        if activity_db:
            from neurolang_apps.utils.tui.activity import ActivityLog

            self.activity = ActivityLog(activity_db)
            # This run of the server, for its uptime (see _measure).
            self._run = self.activity.start_run()
        #: The latest usage of the builder's process (process_usage): the
        #: memory the sessions share.
        self.builder_usage: Optional[dict] = None
        self.most = most
        self.idle = idle
        self.running = running or os.cpu_count() or 1
        self._slots = threading.BoundedSemaphore(self.running)
        self.query_seconds = query_seconds or None
        self._sessions = {}
        self._lock = threading.Lock()
        self._starts = itertools.count(1)  # numbers every start once
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
        # The variables *withhold* (the server's keys) are not the
        # builder's, nor so of the sessions it forks: a session has those
        # its identity gives (a guest's none), not even in the environment
        # its process started with.
        withheld = {n: os.environ.pop(n) for n in withhold if n in os.environ}
        try:
            self._builder.start()
        finally:
            os.environ.update(withheld)
        theirs.close()
        self._closed = False
        for target in (self._accept, self._read_builder, self._reap,
                       self._measure):
            threading.Thread(target=target, daemon=True).start()

    def session(self, user: str, tab: str) -> Session:
        """Return the session of the tab *tab* of *user*, new if need be."""
        key = (user, tab)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = self._sessions[key] = Session(self, key)
            session.used = time.monotonic()
            return session

    def close(self) -> None:
        """End every session and the builder."""
        self._closed = True
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.end("the server stopped")
        if self.activity is not None:
            self.activity.beat(self._run, stopped=True)
        self._builder_conn.close()
        self._builder.join(5)
        if self._builder.is_alive():
            self._builder.kill()
            self._builder.join(5)
        self._listener.close()

    def _owns(self, session: Session) -> bool:
        """Return whether *session* is still that of its key."""
        with self._lock:
            return self._sessions.get(session.key) is session

    def _end_idlest(self, starting: Session) -> None:
        """Make room for *starting*, if *most* sessions run or start: end
        the idlest of its user's, or another's unused for a minute."""
        running = [s for s in self._sessions.values()
                   if s is not starting and (s.pid is not None or s.busy)]
        if len(running) < self.most:
            return
        now = time.monotonic()
        idle = [s for s in running if not s.busy]
        idle = [s for s in idle if s.key[0] == starting.key[0]] or \
            [s for s in idle if now - s.used >= 60]
        if not idle:
            raise RuntimeError(
                f"The server has {self.most} sessions at work: close a tab "
                "or wait"
            )
        min(idle, key=lambda s: s.used).end(
            f"made room for a session of {starting.key[0]}")

    def _fork(self, session: Session, engine: str) -> None:
        """Number a start of *session* and ask the builder for its
        process."""
        if self.dead:
            raise RuntimeError(self.dead)
        with self._lock:
            # A new number first: no earlier start's process is this one's.
            session.starts = start = next(self._starts)
            if self._sessions.get(session.key) is not session:
                raise SessionEnded("This session ended: restart it")
            self._end_idlest(session)
        # The user as they are now: their variables and groups may have
        # changed since their last start.
        env, groups = session.identity = self.identity(session.key[0])
        feedback = None
        if self.feedback_db:
            feedback = {"path": str(self.feedback_db),
                        "user": session.key[0], "groups": groups}
        try:
            with self._forking:
                self._builder_conn.send(
                    ("fork", session.key, start, engine, env, feedback)
                )
        except OSError:  # the builder is gone, its reader may not know yet
            raise RuntimeError("The builder of the engines ended: see the "
                               "server's log")

    def _owner(self, key, start: int) -> Optional[Session]:
        """Return the session *key* if its latest start is *start*."""
        with self._lock:
            session = self._sessions.get(tuple(key))
        return session if session and session.starts == start else None

    def info(self) -> list:
        """Return the sessions: the user and the tab of each, how long it
        has been unused, in seconds, whether it runs and is at work."""
        now = time.monotonic()
        with self._lock:
            sessions = list(self._sessions.values())
        return [{"user": s.key[0], "tab": s.key[1], "engine": s.engine,
                 "idle": round(now - s.used), "running": s.pid is not None,
                 "busy": bool(s.busy),
                 # the latest sample of its process, and the most memory
                 "cpu": s.usage and s.usage["cpu"],
                 "memory": s.usage and s.usage["private"],
                 "pss": s.usage and s.usage["pss"],
                 "peak": s.peak or None} for s in sessions]

    #: How long a refresh waits for a session's process, in seconds.
    REFRESH_SECONDS = 10.0

    def refresh(self, users=None) -> None:
        """Give the running sessions of *users* (all by default) their
        user as they are now, if it changed (an admin or the user changed
        their assistant): see
        :meth:`~neurolang_apps.utils.tui.local_backend.LocalBackend.use_identity`.
        Each is given it by a thread, which waits for it REFRESH_SECONDS at
        most; a session starting takes it once its process is there.
        """
        with self._lock:
            sessions = [s for s in self._sessions.values()
                        if s.pid is not None
                        and (users is None or s.key[0] in users)]
        for session in sessions:
            threading.Thread(target=self._refresh, args=(session,),
                             daemon=True).start()

    def _refresh(self, session: Session) -> None:
        identity = self.identity(session.key[0])
        if identity == session.identity:
            return
        try:
            session._call("use_identity", timeout=self.REFRESH_SECONDS,
                          env=identity[0], groups=identity[1])
            session.identity = identity
        except Exception:  # an ended session takes it at its start
            pass

    def end(self, user: str, tab: Optional[str] = None,
            reason: str = "ended by an admin") -> int:
        """End the sessions of *user* (that of the tab *tab* only, if
        given), for *reason*; return how many."""
        with self._lock:
            ended = [s for key, s in self._sessions.items()
                     if key[0] == user and tab in (None, key[1])]
            for session in ended:
                del self._sessions[session.key]
        for session in ended:
            session.end(reason)
        return len(ended)

    def _accept(self) -> None:
        while not self._closed:
            conn = None
            try:
                conn = self._listener.accept()
                _, key, start, pid = conn.recv()
                session = self._owner(key, start)
            except Exception:  # not a session's process, or the end
                if conn is not None:
                    conn.close()
                continue
            if session is None:
                conn.close()  # of a start given up, or an ended session
                _kill(pid)
                continue
            with session._lock:  # not after it ended
                if session._forked.is_set():
                    conn.close()
                    _kill(pid)
                else:
                    session._attach(conn, pid)

    def _read_builder(self) -> None:
        while True:
            try:
                kind, key, start, value = self._builder_conn.recv()
            except (EOFError, OSError):
                break
            session = self._owner(key, start)
            if session is None:
                continue
            if kind == "status":
                session.emit_text(json.dumps(
                    {"type": "status", "message": value + "…"}
                ))
            elif kind == "failed":
                session._failed(value)
            elif kind == "forked":  # its process says hello at once
                timer = threading.Timer(self.HELLO_SECONDS, self._late,
                                        args=(session, start, value))
                timer.daemon = True
                timer.start()

        # The builder is gone: no session starts any more.
        self.dead = "The builder of the engines ended: see the server's log"
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if not session._forked.is_set():
                session._failed(self.dead)

    #: Why no session starts any more, if the builder ended.
    dead: Optional[str] = None

    #: How long a session's process may take to connect, in seconds.
    HELLO_SECONDS = 60.0

    @staticmethod
    def _late(session: Session, start: int, pid: int) -> None:
        with session._lock:
            if session.starts == start and not session._forked.is_set():
                _kill(pid)
                session._failed(f"The process of the session ({pid}) did "
                                "not start: see the server's log")

    #: How often the usage of the processes is taken, in seconds.
    USAGE_SECONDS = 10.0

    #: How often the activity log is told the server is alive, in seconds.
    ALIVE_SECONDS = 60.0

    def _measure(self) -> None:
        """Take the usage of the processes of the sessions running, and of
        the builder, every USAGE_SECONDS; tell the activity log the server
        is alive every ALIVE_SECONDS."""
        beaten = time.monotonic()
        while not self._closed:
            if self.activity is not None and \
                    time.monotonic() - beaten >= self.ALIVE_SECONDS:
                beaten = time.monotonic()
                try:
                    self.activity.beat(self._run)
                except Exception:
                    traceback.print_exc()
            self.builder_usage = process_usage(self._builder.pid)
            with self._lock:
                sessions = [s for s in self._sessions.values()
                            if s.pid is not None]
            for session in sessions:
                session._sample()
            time.sleep(self.USAGE_SECONDS)

    def _log(self, session: Session, kind: str, **fields) -> None:
        """Log the event *kind* of *session*, if the activity is logged;
        a failure to is told in the server's log only."""
        if self.activity is None:
            return
        groups = session.identity[1] if session.identity else ""
        try:
            self.activity.add(session.key[0], kind, tab=session.key[1],
                              groups=groups, engine=session.engine,
                              **fields)
        except Exception:
            traceback.print_exc()

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
                session.end("idle")
