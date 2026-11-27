"""Results of the strategies experiment (PROTOCOL.md).

    python analyse.py BiomedSQL.csv RESULTS_DIR

RESULTS_DIR holds A_1..3, S1, S2 and S4_f (f = 0..3) as ``*.judged.jsonl``
(S2 may be absent) and the raw ``*.jsonl`` with the answers (for the vote).
"""

import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import gold_table, load_biomedsql  # noqa: E402

B = 10000


def load(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def signature(rec):
    parts = [str(rec.get("boolean"))]
    for rel in rec.get("answer") or []:
        parts.append(
            json.dumps(
                [rel["n"], sorted(map(tuple, rel["rows"]))], default=str
            )
        )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def vote(runs):
    """For each id, the judged record of the run whose answer is most common."""
    out = {}
    for i in runs[0]:
        recs = [r[i] for r in runs if i in r]
        sigs = [signature(r) for r in recs]
        best = max(set(sigs), key=lambda s: (sigs.count(s), -sigs.index(s)))
        out[i] = recs[sigs.index(best)]
    return out


def per_template(rec_by_id, correct, ids):
    out = {}
    for i in ids:
        r = rec_by_id.get(i)
        if r is None:
            continue
        out.setdefault(r["template"], []).append(float(correct(r)))
    return {t: sum(v) / len(v) for t, v in out.items()}


def main(csv_path, results):
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    items = json.load(open(HERE.parent / "items.json"))["items"]
    ids = [i["id"] for i in items]
    nonempty = {i for i in ids if gold_table(by_id[i])[1]}
    templates = sorted(
        {i["template"] for i in items}, key=lambda t: int(t[1:])
    )

    def judged(tag):
        recs = {}
        for path in sorted(glob.glob(f"{results}/{tag}.judged.jsonl")):
            recs.update({r["id"]: r for r in load(path)})
        return recs

    def raw(tag):
        recs = {}
        for path in sorted(glob.glob(f"{results}/{tag}.jsonl")):
            recs.update({r["id"]: r for r in load(path)})
        return recs

    systems = {}
    for r in (1, 2, 3):
        systems[f"A run {r}"] = judged(f"A_{r}")
    a_raw = [raw(f"A_{r}") for r in (1, 2, 3)]
    a_judged = [systems[f"A run {r}"] for r in (1, 2, 3)]
    if all(a_judged):
        merged = []
        for j, rw in zip(a_judged, a_raw):
            merged.append({i: {**rw[i], "judge": j[i]["judge"]} for i in j})
        systems["S3 vote of A runs 1-3"] = vote(merged)
    systems["S1 stat-instruct"] = judged("S1")
    systems["S2 verify"] = judged("S2")
    s4 = {}
    for f in range(4):
        s4.update(judged(f"S4_{f}"))
    systems["S4 learned guidance (CV)"] = s4
    systems = {k: v for k, v in systems.items() if v}
    cost = {
        k: sum(r.get("usd", 0) or 0 for r in v.values())
        for k, v in systems.items()
    }
    metrics = {
        "judge": lambda r: r.get("judge") == "correct",
        "strict": lambda r: bool(r.get("strict")),
    }
    rng = random.Random(0)
    draws = [[rng.choice(templates) for _ in templates] for _ in range(B)]

    def table(name, subset, metric):
        return per_template(systems[name], metrics[metric], subset)

    def score(per, sample):
        vals = [per[t] for t in sample if t in per]
        return float(np.mean(vals)) if vals else float("nan")

    for label, subset in (
        ("all items", ids),
        ("non-empty gold", sorted(nonempty)),
    ):
        print(
            f"\n== {label}: accuracy over {len(templates)} templates, 95% CI over templates"
        )
        print(
            f"{'system':26} {'n':>4} | {'judge':>6} {'95% CI':>15} | {'strict':>6} | {'diff vs A run 1 (judge)':>26}"
        )
        base = table("A run 1", subset, "judge")
        for name in systems:
            pj = table(name, subset, "judge")
            ps = table(name, subset, "strict")
            n = sum(1 for i in subset if i in systems[name])
            bs = sorted(score(pj, d) for d in draws)
            if name == "A run 1":
                diff = ""
            else:
                common = [t for t in templates if t in pj and t in base]
                ds = sorted(
                    np.mean([pj[t] - base[t] for t in d if t in common])
                    for d in draws
                    if any(t in common for t in d)
                )
                est = np.mean([pj[t] - base[t] for t in common])
                diff = f"{100*est:+.1f} [{100*ds[int(.025*len(ds))]:+.1f}, {100*ds[int(.975*len(ds))-1]:+.1f}]"
            print(
                f"{name:26} {n:4d} | {100*score(pj, templates):6.1f} [{100*bs[int(.025*B)]:5.1f},{100*bs[int(.975*B)-1]:5.1f}] | {100*score(ps, templates):6.1f} | {diff:>26}"
            )
    print("\n== cost (USD, all items of the system)")
    for k, v in cost.items():
        print(f"  {k:26} ${v:.2f}")
    if "S4 learned guidance (CV)" in systems:
        folds = json.load(open(f"{results}/s4/folds.json"))
        base = table("A run 1", ids, "judge")
        s4t = table("S4 learned guidance (CV)", ids, "judge")
        print("\n== S4 per fold (templates held out): A run 1 vs S4")
        for f, held in enumerate(folds):
            print(
                f"  fold {f}: {100*np.mean([base[t] for t in held if t in base]):5.1f} -> {100*np.mean([s4t[t] for t in held if t in s4t]):5.1f}"
            )
    print("\n== per template (judge): A1 A2 A3 | S1 S2 vote S4")
    cols = [
        "A run 1",
        "A run 2",
        "A run 3",
        "S1 stat-instruct",
        "S2 verify",
        "S3 vote of A runs 1-3",
        "S4 learned guidance (CV)",
    ]
    per = {c: table(c, ids, "judge") for c in cols if c in systems}
    for t in templates:
        print(
            "  "
            + t.ljust(4)
            + " ".join(
                f"{100*per[c].get(t, float('nan')):4.0f}"
                for c in cols
                if c in per
            )
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
