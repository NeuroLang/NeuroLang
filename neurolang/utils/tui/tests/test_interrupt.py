"""Tests for the interruption of long computations with Ctrl+C."""

import _thread
import signal
import threading
import time

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.utils.tui import interrupt  # noqa: E402


@pytest.fixture(autouse=True)
def python_sigint_handler():
    """Make Ctrl+C raise KeyboardInterrupt, as in the REPL."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def _ctrl_c_after(*delays):
    """Simulate Ctrl+C presses after each of the *delays* in seconds."""
    for delay in delays:
        threading.Timer(delay, _thread.interrupt_main).start()


def test_returns_value_and_raises_errors():
    assert interrupt.call_interruptibly(lambda: 42) == 42
    with pytest.raises(ZeroDivisionError):
        interrupt.call_interruptibly(lambda: 1 / 0)


def test_ctrl_c_interrupts_python_code():
    def busy():
        while True:
            time.sleep(0.01)

    _ctrl_c_after(0.2)
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        interrupt.call_interruptibly(busy)
    assert time.monotonic() - start < interrupt.GRACE_SECONDS


def test_second_ctrl_c_exits_when_stuck_in_native_code(monkeypatch):
    release = threading.Event()
    stuck = []

    def fake_exit(status):
        release.set()
        raise SystemExit(status)

    monkeypatch.setattr(interrupt.os, "_exit", fake_exit)
    _ctrl_c_after(0.2, 0.2 + interrupt.GRACE_SECONDS + 0.5)
    try:
        with pytest.raises(SystemExit) as exc_info:
            interrupt.call_interruptibly(
                release.wait, on_stuck=lambda: stuck.append(True)
            )
    finally:
        release.set()
    assert exc_info.value.code == interrupt.EXIT_STATUS
    assert stuck == [True]
