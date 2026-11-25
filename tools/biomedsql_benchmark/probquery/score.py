"""The readings of PROTOCOL.md, from results/<arm>.jsonl (--dev:
results/dev/<arm>.jsonl).

    uv run --no-sync python score.py [--dev]

Every reading is reported twice: over all templates, and without the four
restored ones (P4d, P4e, P5d, P6d: the protocol's example shapes, three of
which the lifted solvers refuse; S2 is wrong where it gives no answer).

Correct: |answer - gold| <= 1e-3 (P7: <= 1% of gold); no answer is wrong.
Intervals are 95% percentile intervals of 10,000 bootstrap draws of the
templates (clusters), seed 5. The arm differences use the questions both arms
have answered or attempted.

1. per level: accuracy of S0, S1, S2 (with abs error, no-answer share, model
   calls, cost);
2. H7: (S2 - S1 accuracy on P4-P7 pooled) - (the same on P1-P2 pooled);
3. secondary: S2 - S1 on P4-P7 pooled, and on all levels;
amendment: the effort of S2 against S1, per level and pooled (P1-P2, P4-P7,
all): mean cost (list price), model calls, failed tool calls, time (model,
tool), cost per correct answer and its ratio S2 / S1, and the final query's
length (characters, lines), with the same template intervals.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from arms import RESULTS_TAG
HERE = Path(__file__).resolve().parent
LEVELS = [f"P{i}" for i in range(1, 8)]
HARD, EASY = {"P4", "P5", "P6", "P7"}, {"P1", "P2"}
DRAWS = 10_000


def correct(level, answer, gold):
    if answer is None or not np.isfinite(answer):
        return False
    if level == "P7":
        return abs(answer - gold) <= 0.01 * abs(gold)
    return abs(answer - gold) <= 1e-3


def load(dev, exclude_restored=False):
    """The records of each arm; without the restored templates (P4d, P4e,
    P5d, P6d) if *exclude_restored*."""
    folder = HERE / "results" / RESULTS_TAG / ("dev" if dev else "")
    gold = json.load(open(HERE / "gold.json"))
    restored = {q["id"] for q in json.load(open(HERE / "questions.json"))
                if q.get("restored")}
    arms = {}
    for arm in ("S0", "S1", "S2"):
        path = folder / f"{arm}.jsonl"
        if not path.exists():
            continue
        recs = {}
        for line in open(path):
            r = json.loads(line)
            if exclude_restored and r["id"] in restored:
                continue
            r["gold"] = gold[r["id"]]["gold"]
            r["correct"] = correct(r["level"], r["answer"], r["gold"])
            recs[r["id"]] = r
        arms[arm] = recs
    return arms


def bootstrap(stat, templates, rng):
    """Percentile 95% interval of stat(multiplicity of each template)."""
    idx = rng.integers(0, len(templates), (DRAWS, len(templates)))
    mult = np.zeros((DRAWS, len(templates)))
    np.add.at(mult, (np.arange(DRAWS)[:, None], idx), 1)
    values = np.array([stat(m) for m in mult])
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), DRAWS - len(values)


def per_template(recs, ids, templates):
    """(correct, n) per template over the questions *ids*."""
    c, n = np.zeros(len(templates)), np.zeros(len(templates))
    pos = {t: i for i, t in enumerate(templates)}
    for i in ids:
        r = recs[i]
        c[pos[r["template"]]] += r["correct"]
        n[pos[r["template"]]] += 1
    return c, n


def accuracy(c, n, mask, m=None):
    m = np.ones(len(c)) if m is None else m
    den = (m * n * mask).sum()
    return (m * c * mask).sum() / den if den else float("nan")


EFFORT = {
    "cost $": lambda r: r["usage"]["list_cost"],
    "calls": lambda r: r["steps"],
    "failed tools": lambda r: r.get("tools", {}).get("failed", np.nan),
    # Model + tool time; the wall-clock time of S2 also holds per-question
    # harness time (its prompt built over the 9 GB engine, the final query
    # run again), which is not the agent's.
    "agent s": lambda r: r["usage"].get("model_seconds", np.nan)
    + r.get("tools", {}).get("seconds", np.nan),
    "wall s": lambda r: r["seconds"],
    "model s": lambda r: r["usage"].get("model_seconds", np.nan),
    "tool s": lambda r: r.get("tools", {}).get("seconds", np.nan),
    "query chars": lambda r: (r.get("query_size") or {}).get("chars", np.nan),
    "query lines": lambda r: (r.get("query_size") or {}).get("lines", np.nan),
}


def sums(recs, ids, templates, metric):
    """(sum of metric, questions with it) per template."""
    pos = {t: i for i, t in enumerate(templates)}
    s, n = np.zeros(len(templates)), np.zeros(len(templates))
    for i in ids:
        v = metric(recs[i])
        if v is not None and np.isfinite(v):
            s[pos[recs[i]["template"]]] += v
            n[pos[recs[i]["template"]]] += 1
    return s, n


def effort(arms, ids, templates, rng):
    level = np.array([t[:2] for t in templates])
    pools = [(lv, level == lv) for lv in LEVELS] + [
        ("P1-P2", np.isin(level, list(EASY))),
        ("P4-P7", np.isin(level, list(HARD))),
        ("all", np.ones(len(templates), bool)),
    ]
    print("\nAmendment: effort of S2 against S1 (means per question; "
          "[95% CI over templates]; query length of the answers given)")
    for lv, mask in pools:
        if not mask.any():
            continue
        print(f"  {lv}:")
        for name, metric in EFFORT.items():
            cells = []
            for arm in ("S1", "S2"):
                s_, n_ = sums(arms[arm], ids, templates, metric)
                stat = lambda m=None: accuracy(s_, n_, mask, m)
                lo, hi, _ = bootstrap(stat, templates, rng)
                cells.append(f"{arm} {stat():.3f} [{lo:.3f}, {hi:.3f}]")
            print(f"    {name:>12}: " + "   ".join(cells))
        cost = {a: sums(arms[a], ids, templates, EFFORT["cost $"])[0]
                for a in ("S1", "S2")}
        right = {a: per_template(arms[a], ids, templates)[0]
                 for a in ("S1", "S2")}
        per = {a: (lambda m=None, a=a: accuracy(cost[a], right[a], mask, m))
               for a in ("S1", "S2")}
        ratio = lambda m=None: per["S2"](m) / per["S1"](m)
        lo, hi, dropped = bootstrap(ratio, templates, rng)
        print(f"    $ / correct: S1 {per['S1']():.4f}   S2 {per['S2']():.4f}"
              f"   S2/S1 {ratio():.2f} [{lo:.2f}, {hi:.2f}]"
              + (f" ({dropped} draws undefined)" if dropped else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    for exclude, title in (
        (False, "ALL TEMPLATES (25: the 21 and the restored P4d, P4e, P5d, "
                "P6d)"),
        (True, "EXCLUDING THE RESTORED TEMPLATES (the 21)"),
    ):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        report(load(args.dev, exclude))


def report(arms):
    rng = np.random.default_rng(5)
    print("Reading 1: per level (accuracy [95% CI over templates], mean |error| "
          "of answers, no-answer share, model calls, cost)")
    for arm, recs in arms.items():
        templates = sorted({r["template"] for r in recs.values()})
        c, n = per_template(recs, recs, templates)
        level = np.array([t[:2] for t in templates])
        print(f"\n{arm}: {len(recs)} questions, cost "
              f"${sum(r['usage']['cost'] for r in recs.values()):.2f}")
        for lv in LEVELS + ["all"]:
            rs = [r for r in recs.values() if lv == "all" or r["level"] == lv]
            if not rs:
                continue
            mask = (level == lv) if lv != "all" else np.ones(len(templates), bool)
            acc = accuracy(c, n, mask)
            lo, hi, _ = bootstrap(lambda m: accuracy(c, n, mask, m), templates, rng)
            errs = [abs(r["answer"] - r["gold"]) for r in rs
                    if r["answer"] is not None]
            none = sum(r["answer"] is None for r in rs) / len(rs)
            print(f"  {lv:>3}: {acc:.3f} [{lo:.3f}, {hi:.3f}]  n={len(rs):3d}  "
                  f"|err| {np.mean(errs) if errs else float('nan'):.4f}  "
                  f"none {none:.2f}  calls {np.mean([r['steps'] for r in rs]):.1f}  "
                  f"${sum(r['usage']['cost'] for r in rs):.2f}")

    if "S1" not in arms or "S2" not in arms:
        return
    ids = sorted(set(arms["S1"]) & set(arms["S2"]))
    templates = sorted({arms["S1"][i]["template"] for i in ids})
    level = np.array([t[:2] for t in templates])
    c1, n = per_template(arms["S1"], ids, templates)
    c2, _ = per_template(arms["S2"], ids, templates)
    hard, easy = np.isin(level, list(HARD)), np.isin(level, list(EASY))
    every = np.ones(len(templates), bool)

    def diff(mask, m=None):
        return accuracy(c2, n, mask, m) - accuracy(c1, n, mask, m)

    def dod(m=None):
        return diff(hard, m) - diff(easy, m)

    print(f"\nS2 - S1 on {len(ids)} questions of both arms, {len(templates)} "
          "templates")
    for name, stat in (
        ("Reading 2, H7: (S2-S1 on P4-P7) - (S2-S1 on P1-P2)", dod),
        ("Reading 3: S2-S1 on P4-P7", lambda m=None: diff(hard, m)),
        ("Reading 3: S2-S1 on all levels", lambda m=None: diff(every, m)),
    ):
        lo, hi, dropped = bootstrap(stat, templates, rng)
        note = f" ({dropped} draws without both pools dropped)" if dropped else ""
        print(f"  {name}: {stat():+.3f} [{lo:+.3f}, {hi:+.3f}]{note}")
    with np.errstate(divide="ignore", invalid="ignore"):
        effort(arms, ids, templates, rng)


if __name__ == "__main__":
    main()
