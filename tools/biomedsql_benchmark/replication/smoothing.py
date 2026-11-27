"""POST HOC (decided after the results of PROTOCOL.md were seen).

The unsmoothed share is overconfident when a gene was tested in few datasets
(a single agreeing dataset gives support 1). A prior of a pseudo-datasets that
found the gene significant and b that did not (a Beta prior, i.e. a + b
imaginary datasets among the units of evidence) gives
(k + a) / (n + a + b), still an exact conditional probability over a uniform
choice of unit of evidence. a and b are fitted by log-loss on the predictions
of genes that are not among the genes of the 240 items, then applied to them.

    python smoothing.py SMR.parquet PREDICTIONS.parquet RESULTS_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import CLIP, load_answers, metrics

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6


def counts(smr_path):
    smr = pd.read_parquet(
        smr_path, columns=["Omic", "Gene", "Disease", "p_SMR_multi"]
    )
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    tests["sig"] = tests.p_SMR_multi < THRESHOLD
    return tests.groupby(["Gene", "Disease"]).sig.agg(
        n_all="size", k_all="sum"
    )


def fit(k, n, y, grid):
    best = None
    for a in grid:
        for b in grid:
            p = np.clip((k + a) / (n + a + b), *CLIP)
            loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            if best is None or loss < best[0]:
                best = (loss, a, b)
    return best


def main(smr_path, predictions_path, results):
    items = json.load(open(HERE / "items.json"))
    item_genes = {i["gene"] for i in items}
    per_pair = counts(smr_path)
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    rows = preds.join(per_pair, on=["gene", "disease"])
    rows["n"] = rows.n_all - 1
    rows["k"] = rows.k_all - rows.y
    train = rows[~rows.gene.isin(item_genes)]
    grid = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8]
    loss, a, b = fit(train.k.values, train.n.values, train.y.values, grid)
    print(
        f"fitted on {len(train)} predictions of {train.gene.nunique()} genes"
        f" (none of the items' genes): a={a}, b={b}, log-loss {loss:.3f}"
    )
    y = np.array([i["y"] for i in items])
    k = np.array([round(i["support"] * i["n_other"]) for i in items])
    n = np.array([i["n_other"] for i in items])
    smoothed = (k + a) / (n + a + b)
    answers = load_answers(results)
    l2 = np.array([answers["L2"][i["id"]] for i in items])
    for name, v in (
        ("share (unsmoothed)", k / n),
        (f"share + prior ({a}, {b})", smoothed),
        ("L2 LLM + evidence", l2),
    ):
        m = metrics(v, y)
        print(
            f"{name:24} Brier {m['brier']:.3f} skill {m['skill']:+.2f}"
            f" logloss {m['logloss']:.3f} ECE {m['ece']:.3f} AUC {m['auc']:.3f}"
        )
    for label, mask in (
        ("n = 1", n == 1),
        ("n 2-3", (n >= 2) & (n <= 3)),
        ("n 4-9", (n >= 4) & (n <= 9)),
        ("n >= 10", n >= 10),
    ):

        def brier(v):
            return ((np.clip(v[mask], *CLIP) - y[mask]) ** 2).mean()

        print(
            f"  {label:8} items {mask.sum():3d}: share {brier(k / n):.3f}"
            f" prior {brier(smoothed):.3f} L2 {brier(l2):.3f}"
        )
    rng = np.random.default_rng(0)
    genes = [i["gene"] for i in items]
    by = {}
    for j, g in enumerate(genes):
        by.setdefault(g, []).append(j)
    keys = sorted(by)
    diffs = []
    for _ in range(2000):
        idx = [j for g in rng.choice(keys, len(keys)) for j in by[g]]
        diffs.append(
            metrics(smoothed[idx], y[idx])["brier"]
            - metrics(l2[idx], y[idx])["brier"]
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    est = metrics(smoothed, y)["brier"] - metrics(l2, y)["brier"]
    print(f"Brier, share + prior minus L2: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main(*sys.argv[1:])
