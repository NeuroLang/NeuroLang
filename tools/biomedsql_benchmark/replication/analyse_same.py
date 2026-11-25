"""The readings of PROTOCOL_SAME.md, on the 1,437 comparison items.

    python analyse_same.py SMR.parquet

Reads the out-of-fold model predictions (results/same_models_oof.json,
from same_evidence_models.py), the OpenRouter answers of L2-full and
L2-flags (results/same_llm.jsonl, from run_same_llm.py) and the earlier
direct L2 answers (results/answers.jsonl, results/eval_answers.jsonl).
Writes results/same_analysis.txt.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import CLIP, auc
from leak_checks import cluster_ci, loci_of

HERE = Path(__file__).resolve().parent
MARGIN = 0.010
MODELS = ["F1", "F2", "F3", "E1", "E2", "E3", "Estar"]


def main(smr_path):
    oof = json.load(open(HERE / "results/same_models_oof.json"))
    chosen = oof.pop("_chosen", None)
    llm = {}
    for line in open(HERE / "results/same_llm.jsonl"):
        r = json.loads(line)
        llm.setdefault(r["system"], {})[r["id"]] = r["probability"]
    old = {}
    for line in open(HERE / "results/answers.jsonl"):
        r = json.loads(line)
        if r["system"] == "L2":
            old[r["id"]] = r["probability"]
    for line in open(HERE / "results/eval_answers.jsonl"):
        r = json.loads(line)
        old[r["id"]] = r["L2"]

    items = (json.load(open(HERE / "items.json"))
             + json.load(open(HERE / "items_eval.json")))
    ids = [i["id"] for i in items]
    y = np.array([i["y"] for i in items], float)
    smr = pd.read_parquet(smr_path, columns=["Gene", "ProbeChr", "Probe_bp"])
    loci = loci_of(smr, {i["gene"] for i in items}).locus
    clusters = np.array([loci[i["gene"]] for i in items])

    systems = {m: np.array([oof[i][m] for i in ids], float) for m in MODELS}
    missing = {}
    for name, src in (("L2-full", llm["L2-full"]), ("L2-flags", llm["L2-flags"]),
                      ("L2 direct (earlier)", old)):
        v = [src.get(i) for i in ids]
        missing[name] = sum(x is None for x in v)
        systems[name] = np.array([0.5 if x is None else x for x in v], float)

    lines = [f"{len(ids)} items, {len(set(clusters))} loci, base rate "
             f"{y.mean():.3f}; answers missing (scored as 0.5): {missing}; "
             f"forms chosen per fold: {chosen}",
             "", f"{'system':22} {'Brier':>6} {'95% CI':>17} {'logloss':>8}"
             f" {'AUC':>6}"]

    def brier_terms(p):
        return (np.clip(p, *CLIP) - y) ** 2

    for name, p in systems.items():
        m, lo, hi = cluster_ci(brier_terms(p), clusters)
        pc = np.clip(p, *CLIP)
        ll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        lines.append(f"{name:22} {m:6.3f} [{lo:6.3f}, {hi:6.3f}] {ll:8.3f} "
                     f"{auc(p, y):6.3f}")

    def diff(a, b):
        return cluster_ci(brier_terms(systems[a]) - brier_terms(systems[b]),
                          clusters)

    lines += ["", "paired Brier differences (negative = first better), 95% CI"
                  " resampling loci"]
    pairs = [("F1", "L2-full"), ("F2", "L2-full"), ("F3", "L2-full"),
             ("Estar", "L2-flags"), ("E1", "L2-flags"),
             ("Estar", "L2-full"), ("Estar", "F2"),
             ("L2-full", "L2 direct (earlier)"), ("L2-flags", "L2-full")]
    res = {}
    for a, b in pairs:
        res[(a, b)] = diff(a, b)
        m, lo, hi = res[(a, b)]
        lines.append(f"  {a:8} - {b:20} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    def verdict(lo, hi):
        return ("first better" if hi < 0 else "second better" if lo > 0
                else "equal (interval contains 0)")

    ceiling = [f for f in ("F1", "F2", "F3") if res[(f, "L2-full")][2] < 0]
    lines += ["", "readings (PROTOCOL_SAME.md)",
              "  1. ceiling: " + (f"{', '.join(ceiling)} beat L2-full" if ceiling
                                  else "no model fitted on the table beats "
                                  "L2-full: 'the engine beats an LLM reading "
                                  "the full evidence' is not the case here"),
              "  2. same evidence: E* vs L2-flags "
              + verdict(*res[("Estar", "L2-flags")][1:])
              + "; E1 vs L2-flags " + verdict(*res[("E1", "L2-flags")][1:]),
              f"  3. parity: E* - L2-full upper bound "
              f"{res[('Estar', 'L2-full')][2]:+.4f} against margin {MARGIN}: "
              + ("non-inferior" if res[("Estar", "L2-full")][2] < MARGIN
                 else "not non-inferior")]
    text = "\n".join(lines)
    print(text)
    (HERE / "results/same_analysis.txt").write_text(text + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
