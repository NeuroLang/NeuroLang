"""The shared heavy-job queue of this Mac (/private/tmp/nmethods-heavy.README):
queue, wait for the turn, hold the lock while running, release.

    with heavy_turn("S2 dev run"):
        ...
"""

import contextlib
import os
import time
from pathlib import Path

QUEUE = Path("/private/tmp/nmethods-heavy.queue")
LOCK = Path("/private/tmp/nmethods-heavy.lock")
SESSION = "neurosymbolic-probquery"
POLL = 5


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _lock_pid():
    try:
        for line in LOCK.read_text().splitlines():
            if line.startswith("pid "):
                return line.split()[1]
    except OSError:
        return None
    return None


def _first_alive():
    try:
        lines = QUEUE.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and _alive(parts[1]):
            return line
    return None


def _remove(mine):
    try:
        lines = QUEUE.read_text().splitlines()
    except OSError:
        return
    tmp = QUEUE.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text("".join(l + "\n" for l in lines if l != mine))
    tmp.replace(QUEUE)


@contextlib.contextmanager
def heavy_turn(purpose, enabled=True):
    if not enabled:
        yield
        return
    pid = os.getpid()
    mine = f"{SESSION} {pid} {purpose}"
    with open(QUEUE, "a") as f:
        f.write(mine + "\n")
    print(f"queued for the heavy-job lock: {mine}", flush=True)
    try:
        while True:
            holder = _lock_pid()
            if _first_alive() == mine and (holder is None or not _alive(holder)):
                tmp = LOCK.with_suffix(f".tmp{pid}")
                tmp.write_text(f"purpose {SESSION} {purpose}\npid {pid}\n")
                tmp.replace(LOCK)
                if _lock_pid() == str(pid):
                    break
            time.sleep(POLL)
        print("heavy-job lock taken", flush=True)
        yield
    finally:
        if _lock_pid() == str(pid):
            LOCK.unlink(missing_ok=True)
        _remove(mine)
