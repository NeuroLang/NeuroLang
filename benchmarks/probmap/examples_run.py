"""
Run the PROB programs of an engine's example banks (as tally.py) and the
query shapes of queries.py, saving each answer table, the solver that
answered each PROB rule, the wall time and the peak RSS, for a comparison
of two versions of NeuroLang (compare.py).

    python examples_run.py ENGINE OUTDIR [NAME_SUBSTRING ...]

Measurement only.
"""
import gc
import json
import pickle
import signal
import sys
import time
from pathlib import Path

from neurolang_apps.utils.engine_registry import build_engine, get_engine_config

from probtrace import SOLVER_LOG, PeakRSS, instrument
from queries import QUERIES
from tally import Timeout, _alarm, programs


def frames(result):
    if result is None or isinstance(result, bool):
        return {"ans": result}
    if hasattr(result, "as_pandas_dataframe"):
        return {"ans": result.as_pandas_dataframe()}
    if hasattr(result, "items"):
        return {str(k): frames(v)["ans"] for k, v in result.items()}
    return {"ans": repr(result)}


def main():
    engine, outdir = sys.argv[1], Path(sys.argv[2])
    selected = sys.argv[3:]
    outdir.mkdir(parents=True, exist_ok=True)
    progs = programs(get_engine_config(engine)) + [
        (f"query {name}", text) for name, text in QUERIES.items()
    ]
    if selected:
        progs = [(n, t) for n, t in progs if any(s in n for s in selected)]
    t = time.time()
    nl = build_engine(engine, Path.home() / "neurolang_data", resolution=2)
    print(f"build {time.time() - t:.1f} s, {len(progs)} programs", flush=True)
    instrument(nl)
    signal.signal(signal.SIGALRM, _alarm)
    summary, answers = {}, {}
    for name, text in progs:
        SOLVER_LOG.clear()
        gc.collect()
        error, result = None, None
        signal.alarm(300)
        with PeakRSS() as mem:
            t = time.time()
            try:
                with nl.scope:
                    result = frames(nl.execute_datalog_program(text))
            except Timeout:
                error = "timeout 300 s"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            finally:
                signal.alarm(0)
            elapsed = time.time() - t
        answers[name] = result
        accepted = {}
        for entry in SOLVER_LOG:
            if entry["raised"] is None:
                accepted.setdefault(entry["rule"], entry["solver"])
        summary[name] = {
            "error": error, "seconds": round(elapsed, 3),
            "peak_rss_delta_mb": round((mem.peak - mem.base) / 2**20, 1),
            "solvers": sorted(accepted.values()),
            "tries": [(e["solver"], e["rule"][:120], e["raised"], e["s"])
                      for e in SOLVER_LOG],
        }
        print(name[:60], error, summary[name]["seconds"],
              summary[name]["solvers"], flush=True)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(outdir / "answers.pkl", "wb") as f:
        pickle.dump(answers, f)


if __name__ == "__main__":
    main()
