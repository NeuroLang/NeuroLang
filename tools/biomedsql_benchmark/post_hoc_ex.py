"""POST HOC (defined after the test results were seen; not part of the frozen protocol).

An analogue of the paper's execution accuracy (EX: the set of row identities returned equals the
gold result's), for answers that do not carry BiomedSQL's UUID column. An answer is accepted when
one of its relations has the same number of distinct rows as the gold table and each of its columns,
as a set of values, equals the value set of some gold column (so it invents no value, and it may
leave out columns); an empty gold needs an empty answer; a truncated gold needs at least its rows
and value sets that contain the gold's. Reported next to, never in place of, the judge.

    python post_hoc_ex.py BiomedSQL.csv RESULTS_DIR
"""

import glob
import json
import sys

from common import gold_table, load_biomedsql
from analyse import boot, interval, per_template


def projection_ex(columns, rows, truncated, record):
    gold = set(rows)
    relations = record.get("answer") or []
    if not gold:
        return all(rel["n"] == 0 for rel in relations) and record.get(
            "boolean"
        ) in (None, False)
    gold_cols = [{r[i] for r in gold} for i in range(len(columns))]
    for rel in relations:
        got = {tuple(r) for r in rel["rows"]}
        n = rel["n"]
        if rel["n"] > len(
            rel["rows"]
        ):  # stored rows are cut: only the count is known
            n_ok = n >= len(gold) if truncated else n == len(gold)
            got_cols = [
                {r[i] for r in got} for i in range(len(rel["columns"]))
            ]
            count_ok = n_ok
        else:
            count_ok = (
                len(got) >= len(gold) if truncated else len(got) == len(gold)
            )
            got_cols = [
                {r[i] for r in got} for i in range(len(rel["columns"]))
            ]
        if not count_ok or not got_cols:
            continue
        match = (lambda g, c: g <= c) if truncated else (lambda g, c: g == c)
        if all(any(match(g, c) for g in gold_cols) for c in got_cols):
            return True
    return False


def main(csv_path, results):
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    spec = [
        i
        for i in json.load(open("items.json"))["items"]
        if i["split"] == "test"
    ]
    templates = sorted({i["template"] for i in spec}, key=lambda t: int(t[1:]))
    meta = {i["id"]: gold_table(by_id[i["id"]]) for i in spec}
    out = {}
    for c in "ABC":
        recs = []
        for path in sorted(glob.glob(f"{results}/test_{c}_*.jsonl")):
            if path.endswith(".judged.jsonl"):
                continue
            for line in open(path):
                r = json.loads(line)
                cols, rows, trunc = meta[r["id"]]
                r["_ex"] = projection_ex(cols, rows, trunc, r)
                r["_nonempty"] = bool(rows)
                recs.append(r)
        out[c] = recs
    print(
        "projection-EX (post hoc), test split, accuracy over templates [95% CI over templates]"
    )
    for subset in (False, True):
        print(
            f"\n{'items with a non-empty gold answer' if subset else 'all items'}"
        )
        for c, recs in out.items():
            per = per_template(recs, lambda r: r["_ex"], subset, "test")
            point, draws = boot([per], templates)
            lo, hi = interval(d[0] for d in draws)
            print(
                f"  agent {c} ({len(recs)} answers): {100 * point[0]:5.1f}  [{100 * lo:5.1f}, {100 * hi:5.1f}]"
            )
    empty = sum(not meta[i["id"]][1] for i in spec)
    print(
        f"\nalways empty: {100 * empty / len(spec):.1f}% (the gold is empty for {empty} of {len(spec)} items)"
    )
    agree = [
        (r["_ex"], r.get("strict")) for recs in out.values() for r in recs
    ]
    print(
        "where the projection-EX and the strict scorer disagree: projection-only",
        sum(a and not b for a, b in agree),
        "strict-only",
        sum(b and not a for a, b in agree),
        "of",
        len(agree),
    )
    judged = {}
    for path in glob.glob(f"{results}/test_?_?.judged.jsonl"):
        for line in open(path):
            r = json.loads(line)
            judged[(r["condition"], r["run"], r["id"])] = (
                r["judge"] == "correct"
            )
    both = [
        (r["_ex"], judged[(r["condition"], r["run"], r["id"])])
        for recs in out.values()
        for r in recs
    ]
    print(
        "projection-EX vs judge: both",
        sum(a and b for a, b in both),
        "| judge only",
        sum(b and not a for a, b in both),
        "| projection only",
        sum(a and not b for a, b in both),
        "| neither",
        sum(not a and not b for a, b in both),
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
