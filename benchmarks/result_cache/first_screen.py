"""Time the first-screen queries of the REPL page, as sessions of a server.

    python benchmarks/result_cache/first_screen.py ENGINE DATA_DIR OUT_DIR
        [--cache DIR] [--sessions N] [--queries FILE ...]

Builds ENGINE from DATA_DIR as the builder of ``neurolang-repl-serve
--users --local`` does (``LocalBackend.build``; with ``--cache DIR``, and a
checkout that has it, the results shared by the sessions are kept in DIR),
then forks N sessions (default 2) one after the other, as the server forks
one per tab. Each starts the REPL (``LocalBackend.start``) and submits, by
``LocalBackend.submit`` as the page does, the engine's Datalog examples and
the Datalog queries of its example questions (the page's first screen),
then the programs of the files ``--queries``, freeing the memory after
each as the server does.

``OUT_DIR/times.jsonl`` gets a line per session and query: its wall time,
whether it failed, its number of rows, whether the reply says it was kept,
the private memory of the session before and after it (Linux,
``smaps_rollup``) and the peak resident memory of the session so far
(``ru_maxrss``). ``OUT_DIR/s{N}-q{i}.pkl`` holds the frames of its result
(``app.results``) for comparing answers across checkouts.
"""
import argparse
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path


def private_mb() -> float:
    try:
        with open("/proc/self/smaps_rollup") as smaps:
            return sum(int(line.split()[1]) for line in smaps if line.startswith(
                ("Private_Clean:", "Private_Dirty:"))) / 1024
    except OSError:
        return float("nan")


def first_screen(repl) -> list:
    examples = repl.examples().get("datalog", [])
    by_question = repl.question_queries()
    questions = [by_question.get(q, {}).get("datalog")
                 for q in repl.questions()]
    return [("example", q) for q in examples] + [
        ("question", q) for q in questions if q]


def free_memory():
    """Free what the query left, as a session of the server does after each
    one (``sessions.free_memory``: ``gc.collect`` and ``malloc_trim``)."""
    from neurolang_apps.utils.tui.sessions import free_memory

    free_memory()


def session(backend_class, options, engine, nl, queries_files, out, number):
    backend = backend_class(build=lambda name, say: nl, **options)
    backend.start(engine)
    queries = first_screen(backend.repl) + [
        ("file", Path(f).read_text()) for f in queries_files]
    with open(out / "times.jsonl", "a") as log:
        for index, (kind, query) in enumerate(queries):
            before = private_mb()
            began = time.perf_counter()
            reply = backend.submit(query)
            seconds = time.perf_counter() - began
            free_memory()  # as the server does after each query
            frames = [(b.title, b.frame) for b in backend.repl.app.results]
            with open(out / f"s{number}-q{index}.pkl", "wb") as file:
                pickle.dump({"query": query, "failed": reply["failed"],
                             "frames": frames}, file)
            log.write(json.dumps({
                "session": number, "index": index, "kind": kind,
                "query": query, "seconds": seconds,
                "failed": reply["failed"],
                "rows": sum(len(f) for _, f in frames),
                "kept": "kept from an earlier run" in reply["html"],
                "private_mb_before": before, "private_mb_after": private_mb(),
                "maxrss_mb": resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1024,
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("data_dir")
    parser.add_argument("out")
    parser.add_argument("--cache")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--queries", nargs="*", default=[])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import inspect

    from neurolang_apps.utils.tui.local_backend import LocalBackend

    options = {"data_dir": args.data_dir, "resolution": 2.0}
    if args.cache and "result_cache" in inspect.signature(
            LocalBackend).parameters:
        options["result_cache"] = {"directory": args.cache, "most_mb": 1024}
    builder = LocalBackend(**options)
    began = time.perf_counter()
    nl = builder.build(args.engine, lambda text: None)
    built = time.perf_counter() - began
    began = time.perf_counter()
    attached = None
    if "result_cache" in options:  # what attach adds to the build, timed
        from neurolang_apps.utils import result_cache

        engine = result_cache.engine_digest(
            args.engine, 2.0, Path(args.data_dir).resolve(),
            skip=(Path(args.cache),))
        attached = time.perf_counter() - began
        assert engine == nl.result_cache.engine
    with open(out / "build.json", "w") as file:
        json.dump({"seconds": built, "engine_digest_seconds": attached,
                   "cache": "result_cache" in options}, file)
    import gc

    gc.freeze()
    for number in range(1, args.sessions + 1):
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                session(LocalBackend, options, args.engine, nl, args.queries,
                        out, number)
                code = 0
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        if status:
            print(f"session {number} failed: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
