"""
Trace how the probabilistic queries of the coordinate meta-analysis
engines are solved: which solver of the chain answers each PROB rule (or
which one raised, falling back to the next), the RA plan it runs (show_ra
with analyze, row counts per node), the wall time and the peak RSS of the
query, and its result table, saved for comparison with a direct counting
(sparse matrix) computation (counting.py).

    python probtrace.py ENGINE OUTDIR [QUERY_NAME ...]

Measurement only.
"""
import contextlib
import gc
import io
import json
import resource
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

from neurolang.frontend.show_ra import ShowRAOptions
from neurolang_apps.utils.engine_registry import build_engine

from queries import QUERIES


class PeakRSS:
    """Sample the RSS of this process every 20 ms."""

    def __enter__(self):
        self.proc = psutil.Process()
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.02)

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()


SOLVER_LOG = []


def wrap(name, fun):
    def wrapped(*args, **kwargs):
        rule = args[0]
        t = time.time()
        try:
            res = fun(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - logging the fallback
            SOLVER_LOG.append({
                "solver": name, "rule": str(rule)[:300],
                "raised": f"{type(exc).__name__}: {str(exc)[:200]}",
                "s": round(time.time() - t, 3),
            })
            raise
        SOLVER_LOG.append({
            "solver": name, "rule": str(rule)[:300], "raised": None,
            "s": round(time.time() - t, 3),
        })
        return res
    wrapped.__name__ = getattr(fun, "__name__", name)
    return wrapped


def instrument(nl):
    names = ("small_dichotomy", "dalvi_suciu", "wmc")
    nl.probabilistic_solvers = tuple(
        wrap(f"{n}.succ", f) for n, f in zip(names, nl.probabilistic_solvers)
    )
    nl.probabilistic_marg_solvers = tuple(
        wrap(f"{n}.marg", f)
        for n, f in zip(names, nl.probabilistic_marg_solvers)
    )


def main():
    engine, outdir = sys.argv[1], Path(sys.argv[2])
    names = sys.argv[3:] or list(QUERIES)
    outdir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    nl = build_engine(engine, Path.home() / "neurolang_data", resolution=2)
    print(f"build {time.time() - t:.1f} s", flush=True)
    instrument(nl)
    gc.collect()
    summary = {}
    for name in names:
        program = QUERIES[name]
        # The plan, with the row counts of each node (runs the query).
        buf = io.StringIO()
        SOLVER_LOG.clear()
        with contextlib.redirect_stdout(buf):
            try:
                with nl.scope:
                    nl.execute_datalog_program(
                        program, show_ra=ShowRAOptions(analyze=True)
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"show_ra failed: {type(exc).__name__}: {exc}")
        (outdir / f"{name}.plan.txt").write_text(buf.getvalue())
        plan_log = list(SOLVER_LOG)
        gc.collect()
        SOLVER_LOG.clear()
        with PeakRSS() as mem:
            t = time.time()
            with nl.scope:
                res = nl.execute_datalog_program(program)
            elapsed = time.time() - t
        df = res.as_pandas_dataframe()
        df.to_pickle(outdir / f"{name}.result.pkl")
        summary[name] = {
            "seconds": round(elapsed, 2),
            "peak_rss_delta_gb": round((mem.peak - mem.base) / 2**30, 3),
            "rows": len(df),
            "solvers": list(SOLVER_LOG),
            "show_ra_solvers": plan_log,
        }
        print(name, json.dumps(summary[name], indent=1), flush=True)
        del res, df
        gc.collect()
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
          / 2**30)


if __name__ == "__main__":
    main()
