"""Results of the benchmark, as the protocol fixes them.

    python analyse.py BiomedSQL.csv RESULTS_DIR SPLIT

Reads ``RESULTS_DIR/{SPLIT}_{A,B,C}_{run}.judged.jsonl``, ``baseline.nodb.judged.jsonl``,
and ``oracle.jsonl``; prints, for each condition and baseline, the accuracy over
templates with a 95% bootstrap interval (templates resampled, 10,000 times), for the
judge and the strict scorer, on all items and on items with a non-empty gold answer,
the paired differences between conditions, the runs apart, per template, cost and time.
"""

import collections
import glob
import json
import random
import statistics
import sys
from pathlib import Path

from common import HERE, gold_table, load_biomedsql

B = 10000
RNG = random.Random(0)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def per_template(records, correct, nonempty, split, runs=None):
    """{template: (hits, count)} over records, summed over runs, each item counted once per run."""
    out = collections.defaultdict(lambda: [0.0, 0])
    for r in records:
        if r["split"] != split or (nonempty and not r["_nonempty"]):
            continue
        out[r["template"]][0] += correct(r)
        out[r["template"]][1] += 1
    return {t: tuple(v) for t, v in out.items()}


def score(per, templates):
    vals = [per[t][0] / per[t][1] for t in templates if t in per and per[t][1]]
    return sum(vals) / len(vals) if vals else float("nan")


def boot(pers, templates, fn=None):
    """Bootstrap over templates; *pers* a list of per-template tables, fn combines their scores."""
    point = [score(p, templates) for p in pers]
    draws = []
    for _ in range(B):
        sample = [RNG.choice(templates) for _ in templates]
        draws.append([score(p, sample) for p in pers])
    return point, draws


def interval(values):
    values = sorted(v for v in values if v == v)
    return (
        values[int(0.025 * len(values))],
        values[int(0.975 * len(values)) - 1],
    )


def main(csv_path, results, split):
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    meta = {}
    for i in json.load(open(HERE / "items.json"))["items"]:
        if i["split"] == split:
            _, rows, truncated = gold_table(by_id[i["id"]])
            meta[i["id"]] = {"nonempty": bool(rows), "truncated": truncated}
    templates = sorted(
        {
            i["template"]
            for i in json.load(open(HERE / "items.json"))["items"]
            if i["split"] == split
        },
        key=lambda t: int(t[1:]),
    )
    systems = {}
    for c in "ABC":
        runs = sorted(glob.glob(f"{results}/{split}_{c}_*.judged.jsonl"))
        recs = [r for p in runs for r in load_jsonl(p)]
        if recs:
            systems[f"agent {c}"] = (recs, len(runs))
    nodb = load_jsonl(f"{results}/baseline.nodb.judged.jsonl")
    if nodb:
        systems["no database"] = (nodb, 1)
    for name, (recs, _) in systems.items():
        for r in recs:
            r["_nonempty"] = meta.get(r["id"], {}).get("nonempty", False)
    oracle = [
        r for r in load_jsonl(f"{results}/oracle.jsonl") if r["split"] == split
    ]
    ids = [i for i in meta]
    synthetic = {
        "always empty": [
            {
                "id": i,
                "template": next(
                    t for t in templates if i.startswith(t + ".")
                ),
                "split": split,
                "strict": not meta[i]["nonempty"],
                "_nonempty": meta[i]["nonempty"],
            }
            for i in ids
        ],
        "oracle": [
            dict(r, _nonempty=meta[r["id"]]["nonempty"]) for r in oracle
        ],
    }
    systems.update({k: (v, 1) for k, v in synthetic.items()})
    metrics = {
        "judge": lambda r: r.get("judge") == "correct",
        "strict": lambda r: bool(r.get("strict")),
    }
    results_table = {}
    print(
        f"{split}: {len(templates)} templates, {len(meta)} items, {sum(m['nonempty'] for m in meta.values())} with a non-empty answer, "
        f"{sum(m['truncated'] for m in meta.values())} truncated"
    )
    for subset in (False, True):
        print(
            f"\n== {'items with a non-empty gold answer' if subset else 'all items'}"
        )
        print(
            f"{'system':14} {'runs':>4} | {'judge':>6} {'95% CI':>15} | {'strict':>6} {'95% CI':>15}"
        )
        for name, (recs, runs) in systems.items():
            cells = []
            for mname, fn in metrics.items():
                if mname == "judge" and name in ("always empty", "oracle"):
                    cells.append("      -")
                    cells.append(" " * 15)
                    continue
                per = per_template(recs, fn, subset, split)
                point, draws = boot([per], templates)
                lo, hi = interval(d[0] for d in draws)
                results_table[(name, mname, subset)] = (point[0], draws)
                cells.append(f"{100 * point[0]:6.1f}")
                cells.append(f"[{100 * lo:5.1f},{100 * hi:5.1f}]")
            print(
                f"{name:14} {runs:4d} | {cells[0]} {cells[1]:>15} | {cells[2]} {cells[3]:>15}"
            )
    print(
        "\n== paired differences in judge accuracy (percentage points, all items)"
    )
    for a, b in (
        ("agent B", "agent A"),
        ("agent C", "agent B"),
        ("agent C", "agent A"),
        ("agent A", "no database"),
    ):
        if (a, "judge", False) in results_table and (
            b,
            "judge",
            False,
        ) in results_table:
            pa, da = results_table[(a, "judge", False)]
            pb, db = results_table[(b, "judge", False)]
            lo, hi = interval(x[0] - y[0] for x, y in zip(da, db))
            print(
                f"  {a} - {b}: {100 * (pa - pb):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            )
    print("\n== the runs apart (judge accuracy over templates)")
    for name in ("agent A", "agent B", "agent C"):
        if name in systems:
            recs, _ = systems[name]
            runs = sorted({r["run"] for r in recs})
            vals = [
                score(
                    per_template(
                        [r for r in recs if r["run"] == k],
                        metrics["judge"],
                        False,
                        split,
                    ),
                    templates,
                )
                for k in runs
            ]
            print(
                f"  {name}: "
                + ", ".join(
                    f"run {k}: {100 * v:.1f}" for k, v in zip(runs, vals)
                )
            )
    print(
        "\n== per template (judge, mean over items and runs): "
        + " ".join(["A", "B", "C"])
    )
    for t in templates:
        cells = []
        for name in ("agent A", "agent B", "agent C", "no database"):
            if name in systems:
                p = per_template(
                    systems[name][0], metrics["judge"], False, split
                )
                cells.append(
                    f"{100 * p[t][0] / p[t][1]:3.0f}" if t in p else "  -"
                )
        orc = [r for r in oracle if r["template"] == t]
        print(
            f"  {t:4} "
            + " ".join(cells)
            + f"   (oracle {sum(bool(r.get('strict')) for r in orc)}/{len(orc)})"
        )
    from families import twins

    twin = twins(csv_path)
    print(
        "\n== judge accuracy by family: templates with a development twin / without"
    )
    for name in ("agent A", "agent B", "agent C", "no database"):
        if name in systems:
            per = per_template(
                systems[name][0], metrics["judge"], False, split
            )
            for label, flag in (("with twin", True), ("without", False)):
                ts = [t for t in templates if twin.get(t) == flag]
                print(
                    f"  {name:12} {label:9} ({len(ts)} templates): {100 * score(per, ts):5.1f}"
                )
    print("\n== cost and time")
    for name in ("agent A", "agent B", "agent C"):
        if name in systems:
            recs, _ = systems[name]
            print(
                f"  {name}: ${sum(r.get('usd', 0) for r in recs):.2f} for {len(recs)} answers, median {statistics.median(r['seconds'] for r in recs):.0f} s, "
                f"median calls {statistics.median(r.get('calls', 0) for r in recs):.0f}, no valid answer {sum(not r.get('valid') for r in recs)}"
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
