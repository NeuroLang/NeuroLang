"""Fit the prior (a, b) of PROTOCOL_PRIOR.md by log-loss.

    python fit_prior.py SMR.parquet PREDICTIONS.parquet

Uses every prediction with support > 0 of the genes of ``items.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6
CLIP = (0.02, 0.98)
GRID_A = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2]
GRID_B = [0.1, 0.25, 0.5, 1, 2, 3, 5]


def main(smr_path, predictions_path):
    smr = pd.read_parquet(
        smr_path, columns=["Omic", "Gene", "Disease", "p_SMR_multi"]
    )
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    tests["sig"] = tests.p_SMR_multi < THRESHOLD
    counts = tests.groupby(["Gene", "Disease"]).sig.agg(
        n_all="size", k_all="sum"
    )
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    rows = preds[preds.gene.isin(genes)].join(counts, on=["gene", "disease"])
    n = (rows.n_all - 1).values
    k = (rows.k_all - rows.y).values
    y = rows.y.values
    results = []
    for a in GRID_A:
        for b in GRID_B:
            p = np.clip((k + a) / (n + a + b), *CLIP)
            loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            results.append((loss, a, b))
    results.sort()
    p0 = np.clip(k / n, *CLIP)
    base = -(y * np.log(p0) + (1 - y) * np.log(1 - p0)).mean()
    print(
        f"{len(rows)} predictions of {rows.gene.nunique()} genes; "
        f"unsmoothed log-loss {base:.3f}"
    )
    for loss, a, b in results[:5]:
        print(f"  a={a} b={b} log-loss {loss:.3f}")
    loss, a, b = results[0]
    edge = a in (GRID_A[0], GRID_A[-1]) or b in (GRID_B[0], GRID_B[-1])
    print(f"chosen a={a}, b={b}; on the edge of the grid: {edge}")


if __name__ == "__main__":
    main(*sys.argv[1:])
