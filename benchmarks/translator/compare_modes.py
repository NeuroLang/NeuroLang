"""Run queries with and without the translator to tensor contractions.

    python benchmarks/translator/compare_modes.py ENGINE OUT [--file q.dl ...]
        [--grep REGEX] [--reps N] [--profile REGEX]

Builds the engine once, then for each query forks a child per mode
(contractions on, off), alternating, as the REPL server forks sessions.
Each child records its wall time, peak memory above its start (Linux
VmHWM after /proc/self/clear_refs) and whether a contraction ran, and
pickles its answer. The parent then compares the answers of the two
modes: same rows, floats within rtol 1e-9. The children of the queries
named by ``--profile`` run under cProfile and keep their statistics in
``OUT/NAME-MODE.prof`` (the profiler slows them: their times are not
comparable to the others').
"""
import argparse
import cProfile
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def vm(field):
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return 0


def run_child(nl, name, text, mode, out, profile=False):
    from nl_core.probabilistic import sums_of_products
    from nl_expr.config import config

    config.set("PROBABILISTIC_SOLVER", "contractions", str(mode == "on"))
    ran = []
    evaluate = sums_of_products.evaluate
    sums_of_products.evaluate = lambda *a: ran.append(1) or evaluate(*a)
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass
    start = vm("VmRSS")
    t = time.perf_counter()
    error = None
    profiler = cProfile.Profile() if profile else None
    try:
        with nl.scope:
            if profiler:
                profiler.enable()
            result = nl.execute_datalog_program(text)
        frame = None if result is None else result.as_pandas_dataframe()
    except Exception as e:  # recorded, compared as such
        frame, error = None, f"{type(e).__name__}: {e}"[:300]
    if profiler:
        profiler.disable()
        profiler.dump_stats(out / f"{name}-{mode}.prof")
    seconds = time.perf_counter() - t
    with open(out / f"{name}-{mode}.pkl", "wb") as f:
        pickle.dump((frame, error), f)
    with open(out / "times.jsonl", "a") as f:
        f.write(json.dumps({
            "name": name, "mode": mode, "seconds": seconds,
            "peak_mb": (vm("VmHWM") - start) / 2**20,
            "contractions": len(ran), "error": error,
            "profiled": profile,
        }) + "\n")


def same(a, b):
    (fa, ea), (fb, eb) = a, b
    if ea or eb:
        return ea == eb, f"errors {ea!r} / {eb!r}"
    if fa is None or fb is None:
        return fa is None and fb is None, "none"
    if fa.shape != fb.shape:
        return False, f"shape {fa.shape} / {fb.shape}"
    fa = fa.sort_values(list(fa.columns)).reset_index(drop=True)
    fb = fb.sort_values(list(fb.columns)).reset_index(drop=True)
    worst = 0.0
    for c in fa.columns:
        x, y = fa[c], fb[c]
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            x, y = x.to_numpy(float), y.to_numpy(float)
            if not np.allclose(x, y, rtol=1e-9, atol=0, equal_nan=True):
                return False, f"column {c} differs"
            worst = max(worst, float(np.nanmax(np.abs(x - y))) if len(x) else 0)
        elif not (x.astype(str) == y.astype(str)).all():
            return False, f"column {c} differs"
    return True, f"rows {len(fa)}, max |diff| {worst:.2e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("out")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--grep")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--profile")
    args = parser.parse_args()
    from neurolang_apps.utils.engine_registry import build_engine
    from neurolang_apps.utils import engine_registry

    queries = {Path(f).stem: Path(f).read_text() for f in args.file}
    if args.grep:
        config = engine_registry.get_engine_config(args.engine)
        for key in ("examples", "assistant_examples"):
            for i, ex in enumerate(config.get(key) or []):
                text = ex.get("datalog") if isinstance(ex, dict) else None
                if text and re.search(args.grep, text):
                    queries[f"{key[:3]}{i:03d}"] = text
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nl = build_engine(args.engine, Path.home() / "neurolang_data", resolution=2)
    for name, text in queries.items():
        for rep in range(args.reps):
            for mode in (("on", "off") if rep % 2 == 0 else ("off", "on")):
                pid = os.fork()
                if pid == 0:
                    try:
                        run_child(
                            nl, name, text, mode, out, profile=bool(
                                args.profile and re.fullmatch(args.profile, name)
                            ) and rep == args.reps - 1,
                        )
                    finally:
                        os._exit(0)
                os.waitpid(pid, 0)
        ok, note = same(*(
            pickle.load(open(out / f"{name}-{m}.pkl", "rb")) for m in ("on", "off")
        ))
        print(f"{name}: {'SAME' if ok else 'DIFFERENT'} ({note})", flush=True)


if __name__ == "__main__":
    main()
