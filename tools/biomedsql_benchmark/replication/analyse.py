"""Metrics of the replication experiment, as PROTOCOL.md fixes them.

python analyse.py RESULTS_DIR
"""

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLIP = (0.02, 0.98)
B = 2000


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def ece(p, y):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, 6)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def metrics(p, y):
    p = np.clip(np.asarray(p, float), *CLIP)
    y = np.asarray(y, float)
    brier = float(((p - y) ** 2).mean())
    base = float(((y.mean() - y) ** 2).mean())
    return {
        "brier": brier,
        "skill": 1 - brier / base if base else float("nan"),
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "ece": ece(p, y),
        "auc": auc(p, y),
    }


def load_answers(results):
    answers = {}
    for line in open(f"{results}/llm.jsonl"):
        r = json.loads(line)
        answers.setdefault(r["system"], {})[r["id"]] = r["probability"]
    for path in glob.glob(f"{results}/agent_*.jsonl"):
        for line in open(path):
            r = json.loads(line)
            answers.setdefault("P", {})[r["id"]] = r["probability"]
    return answers


def main(results):
    items = json.load(open(HERE / "items.json"))
    ids = [i["id"] for i in items]
    y = np.array([i["y"] for i in items])
    genes = [i["gene"] for i in items]
    answers = load_answers(results)
    direct = {i["id"]: i["support"] for i in items}
    systems, missing = {}, {}
    systems["D1 any signal"] = np.ones(len(items))
    systems["D2 majority"] = np.array([float(direct[i] >= 0.5) for i in ids])
    for name, key in (
        ("L1 LLM, no data", "L1"),
        ("L2 LLM + evidence", "L2"),
        ("P NeuroLang", "P"),
    ):
        got = answers.get(key, {})
        vals, miss = [], 0
        for i in ids:
            v = got.get(i)
            if v is None:
                miss += 1
                v = 0.5 if key != "P" else direct[i]
            vals.append(v)
        systems[name] = np.array(vals)
        missing[name] = miss
    systems["P direct computation"] = np.array([direct[i] for i in ids])
    off = sum(
        abs(a - b) > 1e-6
        for a, b in zip(
            systems["P NeuroLang"], systems["P direct computation"]
        )
    )
    print(
        f"{len(items)} items, {len(set(genes))} genes, base rate {y.mean():.3f};"
        f" missing answers {missing}; the agent's value differs from the direct"
        f" computation on {off} items"
    )
    point = {k: metrics(v, y) for k, v in systems.items()}
    by_gene = {}
    for k, g in enumerate(genes):
        by_gene.setdefault(g, []).append(k)
    keys = sorted(by_gene)
    rng = random.Random(0)
    draws = [
        [j for g in (rng.choice(keys) for _ in keys) for j in by_gene[g]]
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
        f"\n{'system':22} {'Brier':>7} {'95% CI':>15} {'skill':>6}"
        f" {'logloss':>7} {'ECE':>6} {'AUC':>6}"
    )
    for name in systems:
        m = point[name]
        lo, hi = np.percentile(boot(name, "brier"), [2.5, 97.5])
        print(
            f"{name:22} {m['brier']:7.3f} [{lo:5.3f},{hi:5.3f}]"
            f" {m['skill']:+6.2f} {m['logloss']:7.3f} {m['ece']:6.3f}"
            f" {m['auc']:6.3f}"
        )
    print(
        "\npaired differences, P NeuroLang minus the other (Brier and"
        " log-loss: negative = P better; AUC: positive = P better);"
        " 95% CI over genes"
    )
    for other in ("D2 majority", "L1 LLM, no data", "L2 LLM + evidence"):
        for metric in ("brier", "logloss", "auc"):
            d = boot("P NeuroLang", metric) - boot(other, metric)
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = point["P NeuroLang"][metric] - point[other][metric]
            print(
                f"  P - {other:19} {metric:8} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )
    for name in ("P NeuroLang", "L2 LLM + evidence", "L1 LLM, no data"):
        v = np.clip(systems[name], *CLIP)
        print(f"\nreliability of {name}: bin | n | mean predicted | observed")
        edges = np.linspace(0, 1, 6)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & ((v < hi) if hi < 1 else (v <= hi))
            if m.any():
                print(
                    f"  [{lo:.1f},{hi:.1f}] {m.sum():4d} |"
                    f" {v[m].mean():.3f} | {y[m].mean():.3f}"
                )


if __name__ == "__main__":
    main(sys.argv[1])
