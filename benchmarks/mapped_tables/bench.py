"""
Build ENGINE once in this process and report what the build of its tables
costs and what it gives: the time of each phase, the memory of the process
after the build (private and file-backed, from /proc/self/status), how many
tables were read from the Arrow files of the build and how many computed,
how much of each table is a view of a mapped file, and a signature of each
table (rows, dtypes, row type, duplicates flag and an order-free hash of
its rows). With --queries, run the example queries of the engine and the
benchmark queries and keep each result as a parquet file.

    python bench.py ENGINE OUT_DIR [--tables DIR] [--queries Q.json]
        [--timeout S] [--prepare] [--no-signatures]

--tables points the Arrow files of the build (DATA_DIR/derived_tables) at
DIR; without it the data directory's is used (a branch without them builds
everything). Measurement only.
"""
import argparse
import functools
import json
import os
import resource
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

TIMES = {}
COUNTS = {"read": 0, "computed": 0, "kinds": {}}


def status() -> dict:
    """The memory lines of /proc/self/status, in MiB (Linux)."""
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                out[key] = round(int(value.split()[0]) / 1024, 1)
    except OSError:
        out["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return out


def timed(module, name):
    original = getattr(module, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        began = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TIMES[name] = TIMES.get(name, 0) + round(
                time.perf_counter() - began, 2)

    setattr(module, name, wrapper)


def count_tables(engine_cache):
    """Count the tables the build reads from its Arrow files and those it
    computes (a branch without them has neither)."""
    tables = getattr(engine_cache, "DerivedTables", None)
    if tables is None:
        return
    call = tables.__call__

    def counted(self, kind, frames, parameters, compute):
        computed = []

        def wrapped():
            computed.append(1)
            return compute()

        result = call(self, kind, frames, parameters, wrapped)
        how = "computed" if computed else "read"
        COUNTS[how] += 1
        COUNTS["kinds"].setdefault(kind, {"read": 0, "computed": 0})[how] += 1
        return result

    tables.__call__ = counted


def mapped_ranges():
    """The address ranges of the Arrow files mapped in this process."""
    ranges = []
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 6 and parts[5].endswith(".arrow"):
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                ranges.append((lo, hi))
    except OSError:
        pass
    return ranges


def buffers(series):
    """(address, size) of the buffers holding the values of *series*."""
    array = series.array
    pa_array = getattr(array, "_pa_array", None)
    if pa_array is not None:
        return [
            (buffer.address, buffer.size)
            for chunk in pa_array.chunks for buffer in chunk.buffers()
            if buffer is not None
        ]
    values = np.asarray(array)
    if values.dtype == object:
        return [(0, int(series.memory_usage(index=False, deep=True)))]
    return [(values.__array_interface__["data"][0], values.nbytes)]


def tables(nl):
    seen = {}
    for symbol, value in nl.program_ir.symbol_table.items():
        relation = getattr(value, "value", None)
        frame = getattr(relation, "_container", None)
        if not isinstance(frame, pd.DataFrame):
            continue
        seen.setdefault(id(frame), []).append((symbol.name, relation, frame))
    return seen


def mapping(nl) -> dict:
    """The bytes of the tables that are views of mapped files, and the
    others, of the tables of at least 1 MiB."""
    ranges = mapped_ranges()
    out = {"mapped_bytes": 0, "private_bytes": 0, "tables": {}}
    for entries in tables(nl).values():
        name, _, frame = entries[0]
        mapped = private = 0
        for column in frame.columns:
            for address, size in buffers(frame[column]):
                if any(lo <= address < hi for lo, hi in ranges):
                    mapped += size
                else:
                    private += size
        out["mapped_bytes"] += mapped
        out["private_bytes"] += private
        if mapped + private >= 2 ** 20:
            out["tables"][name] = {
                "mapped_mib": round(mapped / 2 ** 20, 1),
                "private_mib": round(private / 2 ** 20, 1),
            }
    return out


def multiset_hash(frame) -> str:
    """A hash of the rows of *frame* that does not depend on their order."""
    if len(frame) == 0:
        return "empty"
    rows = pd.util.hash_pandas_object(
        frame.set_axis(range(frame.shape[1]), axis=1), index=False
    ).to_numpy()
    total = int(rows.sum(dtype=np.uint64))
    squares = int((rows * rows).sum(dtype=np.uint64))
    return f"{total:016x}{squares:016x}"


def signatures(nl) -> dict:
    out = {}
    for entries in tables(nl).values():
        for name, relation, frame in entries:
            if name.startswith("fresh_"):  # named anew at each build
                name = f"fresh:{len(frame)}:{frame.shape[1]}"
            out[name] = {
                "rows": len(frame),
                "dtypes": [str(d) for d in frame.dtypes],
                "row_type": str(getattr(relation, "row_type", None)),
                "might_have_duplicates": bool(
                    getattr(relation, "might_have_duplicates", True)),
                "hash": multiset_hash(frame),
            }
    return out


class Timeout(Exception):
    pass


def run_queries(nl, queries, out_dir: Path, timeout: int) -> dict:
    def alarm(signum, frame):
        raise Timeout()

    signal.signal(signal.SIGALRM, alarm)
    results = {}
    for name, program in queries.items():
        began = time.perf_counter()
        signal.alarm(timeout)
        try:
            result = nl.execute_datalog_program(program)
            frame = result.as_pandas_dataframe() if result is not None \
                else pd.DataFrame()
            frame = frame.set_axis([str(c) for c in frame.columns], axis=1)
            frame.reset_index(drop=True).to_parquet(
                out_dir / f"q-{name}.parquet")
            results[name] = {"rows": len(frame), "status": "ok"}
        except Timeout:
            results[name] = {"status": "timeout"}
        except Exception as error:  # noqa: BLE001 - recorded, compared
            results[name] = {"status": f"error: {type(error).__name__}: "
                                       f"{str(error)[:200]}"}
        finally:
            signal.alarm(0)
        results[name]["seconds"] = round(time.perf_counter() - began, 2)
        print(name, results[name], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("out_dir")
    parser.add_argument("--tables")
    parser.add_argument("--queries")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-signatures", action="store_true")
    parser.add_argument(
        "--prepare", action="store_true",
        help="then constant_notes.prepare, as the REPL server's builder")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    began = time.perf_counter()
    from neurolang_apps.utils import engine_cache, engine_registry
    imported = time.perf_counter() - began
    if args.tables:
        engine_cache.DERIVED_DIRECTORY = args.tables
    count_tables(engine_cache)
    for name in dir(engine_registry):
        if name.startswith("_phase_") or name == "_resolve_mask":
            timed(engine_registry, name)

    before = status()
    began = time.perf_counter()
    nl = engine_registry.build_engine(
        args.engine, Path.home() / "neurolang_data", resolution=2)
    seconds = time.perf_counter() - began
    built = status()
    report = {
        "engine": args.engine, "pid": os.getpid(),
        "import_seconds": round(imported, 2),
        "build_seconds": round(seconds, 2), "phases": TIMES,
        "status_before": before, "status_built": built,
        "tables": COUNTS, "mapping": mapping(nl),
    }
    print(json.dumps({k: report[k] for k in (
        "build_seconds", "phases", "status_built", "tables")}), flush=True)
    if args.prepare:
        from neurolang_apps.utils.tui import constant_notes
        began = time.perf_counter()
        constant_notes.prepare(nl, args.engine)
        report["prepare_seconds"] = round(time.perf_counter() - began, 2)
        report["status_prepared"] = status()
        report["mapping_prepared"] = mapping(nl)
        print("prepared", report["prepare_seconds"],
              report["status_prepared"], flush=True)
    if not args.no_signatures:
        report["signatures"] = signatures(nl)
    if args.queries:
        queries = json.loads(Path(args.queries).read_text())
        report["queries"] = run_queries(nl, queries, out_dir, args.timeout)
        report["status_queries"] = status()
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
