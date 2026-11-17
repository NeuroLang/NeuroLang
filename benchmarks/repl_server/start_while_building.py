"""
How long a session of the REPL server takes to start on an engine built
while another engine builds (``neurolang_apps.utils.tui.sessions``), and,
with ``--real``, the answers of queries run in such a session.

Run with the Python of the tree measured; it prints a JSON object::

    python benchmarks/repl_server/start_while_building.py --fake
    python benchmarks/repl_server/start_while_building.py --real \\
        --queries q1.dl q2.dl --out answers.json

``--fake`` builds test engines of the registry's names: ``neurosynth`` in
0.5 s, ``neurostore`` in ``--build-seconds``.
``--real`` builds ``--built`` (default neurosynth_iit) from the data
directory, then ``--building`` (default neurostore_iit) while a session
starts on the first. Each time is that of ``Session.start``, from the
request to the session's process answering.
"""

import argparse
import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

from neurolang_apps.utils.tui.sessions import Sessions, process_usage


def fake_build(name, say):
    """A test engine: neurosynth builds in 0.5 s, the others in
    BUILD_SECONDS."""
    from neurolang.frontend import NeurolangPDL

    say(f"Building {name}")
    time.sleep(0.5 if name == "neurosynth"
               else float(os.environ["BUILD_SECONDS"]))
    nl = NeurolangPDL()
    nl.add_tuple_set([(name,)], name="engine")
    return nl


def _timed_start(sessions, user, engine):
    began = time.monotonic()
    sessions.session(user, "1").start(engine)
    return time.monotonic() - began


def _while_building(sessions, built, building, wait):
    """Start a session on *built* (built first), then, once *building*
    has built for *wait* seconds, another: return the times of the first
    start on *built* (its build), of a start on it with nothing building
    and of that start, and the thread starting *building*."""
    first = _timed_start(sessions, "first", built)
    idle = _timed_start(sessions, "idle", built)
    other = threading.Thread(target=_timed_start,
                             args=(sessions, "other", building), daemon=True)
    other.start()
    time.sleep(wait)
    during = _timed_start(sessions, "during", built)
    return first, idle, during, other


def fake(args) -> dict:
    os.environ["BUILD_SECONDS"] = str(args.build_seconds)
    idle, during = [], []
    for _ in range(args.repeats):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Sessions(directory, build=fake_build)
            try:
                times = _while_building(sessions, "neurosynth",
                                        "neurostore", 1.0)
                idle.append(times[1])
                during.append(times[2])
            finally:
                sessions.close()
    return {"mode": "fake", "build_seconds": args.build_seconds,
            "idle": idle, "during": during,
            "idle_median": statistics.median(idle),
            "during_median": statistics.median(during)}


def _answers(session, queries) -> dict:
    """The full answers (CSV) of each query file run in *session*."""
    answers, block = {}, 0
    for path in queries:
        began = time.monotonic()
        reply = session.submit(Path(path).read_text())
        seconds = time.monotonic() - began
        files = []
        while True:
            try:
                files.append(session.download(block=block, kind="csv")
                             .data.decode())
            except Exception:
                break
            block += 1
        answers[Path(path).name] = {"failed": bool(reply.get("failed")),
                                    "seconds": seconds, "csv": files}
    return answers


def real(args) -> dict:
    sessions = Sessions(Path.home() / "neurolang_data", resolution=2.0)
    try:
        first, idle, during, other = _while_building(
            sessions, args.built, args.building, args.wait)
        building = other.is_alive()  # still, after the start measured
        other.join()
        # The memory of the builders (with the coordinator, on this
        # branch), both engines built, summed: on Linux only.
        pids = [sessions._builder.pid,
                *getattr(sessions, "builders", {}).values()]
        usages = [process_usage(pid) for pid in pids]
        memory = {name: sum(u[name] for u in usages if u)
                  for name in ("rss", "pss", "private")}
        answers = _answers(sessions.session("during", "1"), args.queries)
    finally:
        sessions.close()
    if args.out:
        Path(args.out).write_text(json.dumps(answers))
    return {"mode": "real", "built": args.built, "building": args.building,
            "first": first, "idle": idle, "during": during,
            "still_building_after": building, "builders_mb": memory,
            "processes": len(pids),
            "queries": {name: {"failed": a["failed"],
                               "seconds": round(a["seconds"], 2),
                               "rows": [f.count("\n") - 1 for f in a["csv"]]}
                        for name, a in answers.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--build-seconds", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--built", default="neurosynth_iit")
    parser.add_argument("--building", default="neurostore_iit")
    parser.add_argument("--wait", type=float, default=5.0)
    parser.add_argument("--queries", nargs="*", default=[])
    parser.add_argument("--out")
    args = parser.parse_args()
    print(json.dumps(fake(args) if args.fake else real(args)), flush=True)


if __name__ == "__main__":
    main()
