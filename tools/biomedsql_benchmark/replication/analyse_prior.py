"""Metrics of PROTOCOL_PRIOR.md.

    python analyse_prior.py ENGINE_VALUES.json LLM.jsonl

ENGINE_VALUES.json maps an item id to the engine's ``support_prior``.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

from analyse import metrics

HERE = Path(__file__).resolve().parent
B = 2000
A_PRIOR, B_PRIOR = 0.01, 0.5


def main(engine_path, llm_path):
    items = json.load(open(HERE / "items_eval.json"))
    engine = json.load(open(engine_path))
    llm = {}
    for line in open(llm_path):
        r = json.loads(line)
        if r["system"] == "L2":
            llm[r["id"]] = r["probability"]
    y = np.array([i["y"] for i in items])
    k = np.array([round(i["support"] * i["n_other"]) for i in items])
    n = np.array([i["n_other"] for i in items])
    closed = (k + A_PRIOR) / (n + A_PRIOR + B_PRIOR)
    eng = np.array([engine[i["id"]] for i in items])
    miss = sum(llm.get(i["id"]) is None for i in items)
    l2 = np.array([llm.get(i["id"]) or 0.5 for i in items])
    print(
        f"{len(items)} items, {len({i['gene'] for i in items})} genes, base "
        f"rate {y.mean():.3f}; engine - closed form max |diff| "
        f"{np.abs(eng - closed).max():.2e}; L2 answers missing {miss}"
    )
    systems = {
        "share (unsmoothed)": k / n,
        "support_prior (engine)": eng,
        "L2 LLM + evidence": l2,
    }
    genes = [i["gene"] for i in items]
    by = {}
    for j, g in enumerate(genes):
        by.setdefault(g, []).append(j)
    keys = sorted(by)
    rng = random.Random(0)
    draws = [
        [j for g in (rng.choice(keys) for _ in keys) for j in by[g]]
        for _ in range(B)
    ]
    cache = {}

    def boot(name, metric):
        if (name, metric) not in cache:
            cache[(name, metric)] = np.array(
                [metrics(systems[name][d], y[d])[metric] for d in draws]
            )
        return cache[(name, metric)]

    print(
        f"\n{'system':24} {'Brier':>6} {'95% CI':>14} {'skill':>6} "
        f"{'logloss':>7} {'ECE':>6} {'AUC':>6}"
    )
    for name, v in systems.items():
        m = metrics(v, y)
        lo, hi = np.percentile(boot(name, "brier"), [2.5, 97.5])
        print(
            f"{name:24} {m['brier']:6.3f} [{lo:5.3f},{hi:5.3f}] "
            f"{m['skill']:+6.2f} {m['logloss']:7.3f} {m['ece']:6.3f} "
            f"{m['auc']:6.3f}"
        )
    print("\npaired differences (Brier, log-loss: negative = first better)")
    for a, b in (
        ("support_prior (engine)", "share (unsmoothed)"),
        ("support_prior (engine)", "L2 LLM + evidence"),
    ):
        for metric in ("brier", "logloss", "auc"):
            d = boot(a, metric) - boot(b, metric)
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = (
                metrics(systems[a], y)[metric] - metrics(systems[b], y)[metric]
            )
            print(
                f"  {a} - {b:20} {metric:8} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )
    print("\nBrier by the number of other datasets that tested the gene")
    for label, mask in (
        ("1", n == 1),
        ("2-3", (n >= 2) & (n <= 3)),
        ("4-9", (n >= 4) & (n <= 9)),
        (">=10", n >= 10),
    ):

        def brier(v):
            return ((np.clip(v[mask], 0.02, 0.98) - y[mask]) ** 2).mean()

        print(
            f"  n {label:5} items {mask.sum():4d}: share {brier(k / n):.3f} "
            f"prior {brier(eng):.3f} L2 {brier(l2):.3f}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
