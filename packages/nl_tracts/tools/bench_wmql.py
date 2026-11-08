"""Time WMQL programs, by tract_querier's evaluator and as Datalog.

The same synthetic indexing feeds both: ``crossing`` (labels per streamline),
the labels of the ends, and the geometry for the directional terms. For each
size and program it prints the seconds tract_querier's ``EvaluateQueries``
takes (its indexing is built beforehand, as in a real run), and those that
NeuroLang takes, split into loading the relations, compiling the rules and
evaluating them. One tractogram, one atlas of 100 labels.

Usage (the reference is found as in tests/reference_tq.py, through
``TRACT_QUERIER_DIR``)::

    python tools/bench_wmql.py [--backend pandas|polars] [N ...]
    # N streamlines, default 20000 100000
"""

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "nl_tracts" / "tests"))

from nl_tracts.wmql import transpile  # noqa: E402
from reference_tq import (  # noqa: E402
    Indexing, load_reference, reference_dir,
)

LABELS = 100
PROGRAMS = {
    "a and b not in c": "a = 1\nb = 2\nc = 3\nx = a and b not in c\n",
    "only(a or b)": "a = 1\nb = 2\nx = only(a or b)\n",
    "not a": "a = 1\nx = not a\n",
    "endpoints_in(a) and endpoints_in(b)": (
        "a = 1\nb = 2\nx = endpoints_in(a) and endpoints_in(b)\n"
    ),
    "anterior_of(a) (box)": "a = 1\nx = anterior_of(a)\n",
    "uncinate-like bundle": (
        "amy.left = 4\namy.right = 5\ntmp.left = 6\ntmp.right = 7\n"
        "uf.side = endpoints_in(tmp.side and anterior_of(amy.side))"
        " not in only(tmp.opposite)\n"
    ),
}


def synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    pairs = pd.DataFrame(
        {"s": np.repeat(np.arange(n), 8),
         "l": rng.integers(0, LABELS + 1, 8 * n)}
    ).drop_duplicates()
    endings = tuple(
        {s: int(l) for s, l in enumerate(rng.integers(1, LABELS + 1, n))}
        for _ in range(2)
    )
    low = rng.normal(size=(n, 3)) * 20
    high = low + rng.random((n, 3)) * 40
    boxes = {}
    for label in range(1, LABELS + 1):
        lo = rng.normal(size=3) * 20
        boxes[label] = (lo, lo + rng.random(3) * 20)
    ends = np.stack([low + rng.random((n, 3)) * (high - low) for _ in "ab"],
                    axis=1)
    return pairs, endings, boxes, np.stack([low, high], axis=1), ends


def reference_index(n, data):
    pairs, endings, boxes, tract_boxes, ends = data
    crossing = {s: set() for s in range(n)}
    for s, l in zip(pairs["s"].to_numpy(), pairs["l"].to_numpy()):
        crossing[s].add(int(l))
    return Indexing(crossing, endings, boxes, tract_boxes, ends)


def relations(n, data):
    pairs, endings, boxes, tract_boxes, ends = data
    ss = np.arange(n)
    return {
        "traverses": pd.DataFrame({
            "a": "fs", "t": "t1", "s": pairs["s"].to_numpy(np.int32),
            "l": pairs["l"].to_numpy(np.int32)}),
        "endpoint": pd.DataFrame({
            "a": "fs", "t": "t1",
            "s": np.tile(ss, 2).astype(np.int32),
            "e": np.repeat([0, 1], n).astype(np.int8),
            "l": np.r_[[endings[0][s] for s in ss],
                       [endings[1][s] for s in ss]].astype(np.int32)}),
        "tractogram": pd.DataFrame({"t": ["t1"]}),
        "label_extent": pd.DataFrame(
            [("fs", label, k, lo[k], hi[k])
             for label, (lo, hi) in boxes.items() for k in range(3)],
            columns=["a", "l", "axis", "lo", "hi"]
        ).astype({"l": np.int32, "axis": np.int8, "lo": np.float32,
                  "hi": np.float32}),
        "streamline_extent": pd.DataFrame({
            "t": "t1", "s": np.repeat(ss, 3).astype(np.int32),
            "axis": np.tile(np.arange(3), n).astype(np.int8),
            "lo": tract_boxes[:, 0, :].reshape(-1).astype(np.float32),
            "hi": tract_boxes[:, 1, :].reshape(-1).astype(np.float32)}),
        "endpoint_coord": pd.DataFrame({
            "t": "t1", "s": np.repeat(ss, 6).astype(np.int32),
            "e": np.tile(np.repeat([0, 1], 3), n).astype(np.int8),
            "axis": np.tile(np.arange(3), 2 * n).astype(np.int8),
            "v": ends.reshape(-1).astype(np.float32)}),
    }


def timed(f):
    start = time.perf_counter()
    result = f()
    return result, time.perf_counter() - start


def main(sizes, backend="pandas"):
    from nl_expr.config import config

    config.set_query_backend(backend)  # before the engine is made
    from neurolang.frontend import NeurolangDL

    reference = load_reference()
    print(f"tract_querier: {reference_dir()}; NeuroLang backend: {backend}")
    for n in sizes:
        data = synthetic(n)
        index, t_index = timed(lambda: reference_index(n, data))
        frames = relations(n, data)
        rows = len(frames["traverses"])
        print(f"\n{n} streamlines, {rows} traverses rows "
              f"(reference indexing built in {t_index:.1f} s)")
        print(f"{'program':36s} {'tract_querier':>13s} {'load':>6s} "
              f"{'compile':>8s} {'evaluate':>9s} {'datalog':>8s} "
              f"{'ratio':>6s} {'streamlines only':>17s}  rules")
        for name, source in PROGRAMS.items():
            body = reference.queries_preprocess(source)
            _, t_ref = timed(lambda: reference.EvaluateQueries(
                index).visit(ast.Module(body, type_ignores=[])))
            program = transpile(source, atlas="fs")
            saved = program.saved[-1]

            def run(text):
                nl = NeurolangDL()
                _, t_load = timed(lambda: [
                    nl.add_tuple_set(f, name=k) for k, f in frames.items()])
                _, t_compile = timed(
                    lambda: nl.execute_datalog_program(text))
                _, t_eval = timed(nl.solve_all)
                return t_load, t_compile, t_eval

            text = program.program(saved)
            t_load, t_compile, t_eval = run(text)
            total = t_load + t_compile + t_eval
            only = sum(run(program.program(saved, components=("tracts",))))
            print(f"{name:36s} {t_ref:12.3f}s {t_load:5.2f}s "
                  f"{t_compile:7.2f}s {t_eval:8.2f}s {total:7.2f}s "
                  f"{total / t_ref:5.0f}x {only:16.2f}s  "
                  f"{text.count(chr(10))}")


if __name__ == "__main__":
    args = sys.argv[1:]
    backend = "pandas"
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1]
        del args[i:i + 2]
    main([int(a) for a in args] or [20000, 100000], backend)
