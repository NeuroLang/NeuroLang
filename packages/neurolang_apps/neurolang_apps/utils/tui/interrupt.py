"""
Interruption of long computations with Ctrl+C.

Python handles Ctrl+C (SIGINT) only between the bytecodes of the main
thread, so a query busy in a long native computation (e.g. a large
pandas join) cannot be interrupted where it runs. :func:`call_interruptibly`
runs the computation in a worker thread while the main thread waits,
ready to handle Ctrl+C: the first one raises KeyboardInterrupt in the
worker, which stops as soon as it runs Python code again; if it is still
busy, a second Ctrl+C exits the process.
"""

import ctypes
import os
import sys
import threading
import time

POLL_SECONDS = 0.1
GRACE_SECONDS = 1.0
EXIT_STATUS = 130


def _raise_in_thread(thread: threading.Thread, exc_type) -> None:
    """Raise *exc_type* in *thread* when it next runs Python code."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident), ctypes.py_object(exc_type)
    )


def _wait(done: threading.Event, timeout=None) -> None:
    """
    Wait until *done* is set, or for *timeout* seconds, handling signals.

    An event is waited for rather than joining the thread: a
    KeyboardInterrupt raised while joining marks the thread as stopped.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while not done.is_set() and (
        deadline is None or time.monotonic() < deadline
    ):
        done.wait(POLL_SECONDS)


def _stop(worker: threading.Thread, done: threading.Event, on_stuck) -> None:
    """
    Interrupt *worker*, which sets *done* when it ends, and wait for it.

    If it is still running after a grace period, *on_stuck* is called
    (e.g. to tell that a second Ctrl+C quits) and a second Ctrl+C exits
    the process with status 130.
    """
    _raise_in_thread(worker, KeyboardInterrupt)
    try:
        _wait(done, GRACE_SECONDS)
        if not done.is_set():
            on_stuck()
            _wait(done)
    except KeyboardInterrupt:
        os._exit(EXIT_STATUS)


def call_interruptibly(function, on_stuck=lambda: None):
    """
    Return ``function()``, run so that Ctrl+C interrupts it promptly.

    Exceptions of *function*, including the KeyboardInterrupt raised in
    it by Ctrl+C, are raised again. See :func:`_stop` for *on_stuck*.

    Where threads cannot be started, as in WebAssembly, *function* runs
    in the calling thread and cannot be interrupted.
    """
    if sys.platform == "emscripten":
        return function()
    outcome = {}
    done = threading.Event()

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=target, name="neurolang-query")
    worker.daemon = True
    worker.start()
    try:
        _wait(done)
    except KeyboardInterrupt:
        _stop(worker, done, on_stuck)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")
