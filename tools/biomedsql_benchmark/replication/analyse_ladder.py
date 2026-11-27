"""Readings 1-6 of PROTOCOL_LADDER.md.

    python analyse_ladder.py [ITEMS.json [ENGINE.json [OUT.txt [LLM.jsonl]]]]

Without LLM.jsonl (rows: id, question, arm in {flags, full}, probability)
only the engine is reported. Brier is clipped to [0.02, 0.98]; intervals are
95%, 2,000 draws, resampling loci (leak_checks.cluster_ci).
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from leak_checks import B, cluster_ci

HERE = Path(__file__).resolve().parent
CLIP = (0.02, 0.98)
TOL = 0.01
RUNGS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
POOLED = ["Q4", "Q5", "Q6"]
SAMPLE = {"Q1": "q14", "Q2": "q14", "Q3": "q14", "Q4": "q14", "Q5": "q5",
          "Q6": "q6"}


def brier(p, y):
    return (np.clip(p, *CLIP) - y) ** 2


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def load(items_file, engine_file, llm_file):
    data = json.load(open(HERE / items_file))
    engine = json.load(open(HERE / engine_file))["values"]
    llm = {}
    if llm_file:
        for line in open(llm_file):
            r = json.loads(line)
            if r["probability"] is not None:
                llm[(r["id"], r["question"], r["arm"])] = r["probability"]
    table = {}
    for q in RUNGS:
        rows = []
        for it in data[SAMPLE[q]]:
            e = engine.get(it["id"], {}).get(q)
            if e is None:
                continue
            rows.append({
                "id": it["id"], "y": it["gold"][q], "locus": it["locus"],
                "engine": e["engine"],
                "flags": llm.get((it["id"], q, "flags"), np.nan),
                "full": llm.get((it["id"], q, "full"), np.nan)})
        table[q] = {k: np.array([r[k] for r in rows])
                    for k in ("id", "y", "locus", "engine", "flags", "full")}
    return table


def skill(p, y):
    base = y.mean() * (1 - y.mean())
    if base == 0:
        return None
    return 1 - brier(p, y).mean() / base


def per_rung(table, systems):
    print("1. Per rung: Brier [95% CI over loci] and skill vs base rate")
    for q in RUNGS:
        t = table[q]
        y = t["y"].astype(float)
        print(f"\n{q}: n = {len(y)}, base rate {y.mean():.3f}, loci "
              f"{len(set(t['locus']))}; Brier of the base rate "
              f"{y.mean() * (1 - y.mean()):.4f}")
        for s in systems:
            ok = ~np.isnan(t[s])
            if not ok.any():
                print(f"  {s:7} no answers")
                continue
            m = brier(t[s][ok], y[ok])
            c = cluster_ci(m, t["locus"][ok])
            sk = skill(t[s][ok], y[ok])
            sk = "undefined (no positives)" if sk is None else f"{sk:+.3f}"
            print(f"  {s:7} Brier {c[0]:.4f} [{c[1]:.4f}, {c[2]:.4f}] "
                  f"(n {ok.sum()})  skill {sk}")
        for other in ("full", "flags"):
            if "engine" in systems and other in systems:
                ok = ~np.isnan(t[other])
                if ok.any():
                    d = (brier(t["engine"][ok], y[ok])
                         - brier(t[other][ok], y[ok]))
                    print(f"  engine - {other:5} "
                          f"{fmt(cluster_ci(d, t['locus'][ok]))}")


def diffs(table, q, other):
    t = table[q]
    y = t["y"].astype(float)
    ok = ~np.isnan(t[other])
    return (brier(t["engine"][ok], y[ok]) - brier(t[other][ok], y[ok]),
            t["locus"][ok])


def pooled(table, qs, other):
    parts = [diffs(table, q, other) for q in qs]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def joint_ci(a, la, b, lb):
    """mean(a) - mean(b), resampling loci jointly (a, b: per-item terms)."""
    labels = np.unique(np.concatenate([la, lb]))
    ia, ib = np.searchsorted(labels, la), np.searchsorted(labels, lb)
    n = len(labels)
    sa, ca = np.bincount(ia, a, n), np.bincount(ia, minlength=n).astype(float)
    sb, cb = np.bincount(ib, b, n), np.bincount(ib, minlength=n).astype(float)
    draws = np.random.default_rng(0).integers(0, n, (B, n))
    ok = (ca[draws].sum(1) > 0) & (cb[draws].sum(1) > 0)
    bs = (sa[draws].sum(1)[ok] / ca[draws].sum(1)[ok]
          - sb[draws].sum(1)[ok] / cb[draws].sum(1)[ok])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi


def hypotheses(table):
    print("\n2. H1: (engine - full) on pooled Q4-Q6 minus (engine - full) "
          "on Q1")
    a, la = pooled(table, POOLED, "full")
    b, lb = diffs(table, "Q1", "full")
    if len(a) and len(b):
        print(f"  pooled Q4-Q6 {a.mean():+.4f} (n {len(a)}); Q1 "
              f"{b.mean():+.4f} (n {len(b)}); difference "
              f"{fmt(joint_ci(a, la, b, lb))}")
    else:
        print("  no LLM answers")
    print("\n3. H2: engine - flags on pooled Q4-Q6")
    a, la = pooled(table, POOLED, "flags")
    if len(a):
        print(f"  {fmt(cluster_ci(a, la))} (n {len(a)})")
    else:
        print("  no LLM answers")


def coherence(table, systems):
    print(f"\n4. H3: share of the Q1-Q4 items violating P(Q3) <= P(Q4) <= "
          f"P(Q2) or P(Q3) <= P(Q1) <= P(Q2) (tolerance {TOL})")
    ids = [list(table[q]["id"]) for q in ("Q1", "Q2", "Q3", "Q4")]
    if not all(i == ids[0] for i in ids):
        print("  Q1-Q4 items differ; skipped")
        return
    t = table
    for s in systems:
        p1, p2, p3, p4 = (t[q][s] for q in ("Q1", "Q2", "Q3", "Q4"))
        ok = ~(np.isnan(p1) | np.isnan(p2) | np.isnan(p3) | np.isnan(p4))
        if not ok.any():
            print(f"  {s:7} no complete items")
            continue
        p1, p2, p3, p4 = p1[ok], p2[ok], p3[ok], p4[ok]
        v = ((p3 > p4 + TOL) | (p4 > p2 + TOL) | (p3 > p1 + TOL)
             | (p1 > p2 + TOL))
        c = cluster_ci(v.astype(float), t["Q1"]["locus"][ok])
        print(f"  {s:7} {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}] "
              f"({v.sum()} of {ok.sum()} items)")


def main(items_file="results/ladder_items.json",
         engine_file="results/ladder_engine.json",
         out_file="results/ladder_analysis.txt", llm_file=None):
    table = load(items_file, engine_file, llm_file)
    systems = ["engine"] + (["flags", "full"] if llm_file else [])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"items {items_file}; engine {engine_file}; LLM answers "
              f"{llm_file or 'none (engine-only mode)'}\n")
        per_rung(table, systems)
        if llm_file:
            hypotheses(table)
        coherence(table, systems)
        print("\n5. Skill is Brier skill against each rung's base rate "
              "(1 - Brier / base-rate Brier), printed beside the Brier "
              "above.\n6. Every rung and system with answers is reported "
              "above.")
    (HERE / out_file).write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main(*sys.argv[1:])
