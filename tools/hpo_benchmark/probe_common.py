"""Shared helpers for the neurosymbolic dataset probe.

``make_nl`` builds a NeurolangPDL whose probabilistic solvers exclude WMC:
a query that lifted inference cannot solve raises instead of falling back.
``run_item`` runs one item in a subprocess-free, timed way and appends a
JSONL record.
"""
import json
import signal
import time
import traceback
from pathlib import Path

from neurolang.frontend import NeurolangPDL
from neurolang.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
)

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def make_nl():
    return NeurolangPDL(
        probabilistic_solvers=(
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
        ),
        probabilistic_marg_solvers=(
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
        ),
    )


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def run_item(dataset, item_id, fn, timeout=120, out=None, **meta):
    """Call ``fn()``; record parse/solve status, error type and time."""
    rec = dict(dataset=dataset, item=item_id, **meta)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    t0 = time.time()
    try:
        rec["result"] = fn()
        rec["status"] = "ok"
    except Timeout:
        rec["status"] = "timeout"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)[:300]
        rec["tb"] = traceback.format_exc()[-600:]
    finally:
        signal.alarm(0)
    rec["seconds"] = round(time.time() - t0, 3)
    path = RESULTS / (out or f"{dataset}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
