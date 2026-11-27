"""Validate the scorer on controls built from the development items only.

    python validate_judge.py BiomedSQL.csv

Each development item gives answers whose verdict is known by construction:

  correct:   the stored result; the same with shuffled rows and columns renamed;
             the same plus a computed extra column; (truncated) the result plus
             a row the list would have continued with
  incorrect: one row removed; a row added; a number changed by 5%; the result of
             another item of the same template (when different); an empty result
             (when the stored one is not); (truncated) one row removed

Reports the accuracy of the judge and of the strict mechanical scorer on them.
"""

import collections
import json
from concurrent.futures import ThreadPoolExecutor
import random
import sys

from common import HERE, canon, gold_table, load_biomedsql
from scorer import call_judge, judge_prompt, result_text, strict


def essential_columns(item, columns, rows):
    """Columns with a value that BiomedSQL's prose answer states and the question
    does not: the information the question asks for (not an echo of its entity).
    """
    answer, question = item["answer"].lower(), item["question"].lower()
    out = []
    for i, c in enumerate(columns):
        for r in rows:
            v = r[i].lower()
            if len(v) >= 3 and v in answer and v not in question:
                out.append(i)
                break
    return out


def controls(item, other, rng):
    """(name, expected verdict, columns, rows) for answers whose verdict is
    known: edits that keep the answer, and edits of its essential columns."""
    columns, rows, truncated = gold_table(item)
    out = [("gold", True, columns, rows)]
    if not truncated:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        out.append(
            (
                "shuffled+renamed",
                True,
                [f"col{i}" for i in range(len(columns))],
                shuffled,
            )
        )
    if rows:
        out.append(
            (
                "extra column",
                True,
                columns + ["n_columns"],
                [r + (str(len(r)),) for r in rows],
            )
        )
    ocols, orows, _ = other
    if set(orows) != set(rows):
        out.append(("other item", False, ocols, orows))
    out.append(("empty", not rows, columns, []))
    out.append(("bare boolean true", False, ["answer"], [("True",)]))
    if truncated and len(rows) > 1:
        out.append(("truncated: one missing", False, columns, rows[:-1]))
    if truncated or not rows:
        return out
    essential = essential_columns(item, columns, rows)
    if essential and len(rows) >= 2:
        for k, row in enumerate(rows):
            rest = rows[:k] + rows[k + 1 :]
            if any(row[i] not in {r[i] for r in rest} for i in essential):
                out.append(
                    ("row removed (essential value)", False, columns, rest)
                )
                break
    if essential:
        i = essential[0]
        value = rows[0][i]
        try:
            changed_value = (
                canon(float(value) * 1.05) if float(value) != 0 else None
            )
        except ValueError:
            changed_value = value + "_x"
        if changed_value is not None:
            changed = [
                rows[0][:i] + (changed_value,) + rows[0][i + 1 :]
            ] + rows[1:]
            out.append(("essential value changed", False, columns, changed))
    return out


def main(path):
    rows = load_biomedsql(path)
    by_id = {r["uuid"]: r for r in rows}
    spec = json.load(open(HERE / "items.json"))
    dev = [i for i in spec["items"] if i["split"] == "dev"]
    by_t = collections.defaultdict(list)
    for i in dev:
        by_t[i["template"]].append(i["id"])
    rng = random.Random(1)
    stats = collections.defaultdict(
        lambda: [0, 0, 0, 0]
    )  # n, judge ok, strict ok, unused
    jobs = []
    for i in dev:
        item = by_id[i["id"]]
        pool = [x for x in by_t[i["template"]] if x != i["id"]]
        other = gold_table(by_id[rng.choice(pool)])
        columns, grows, truncated = gold_table(item)
        for name, expected, cols, crows in controls(item, other, rng):
            prompt = judge_prompt(
                item, columns, grows, truncated, result_text([(cols, crows)])
            )
            ok_strict = (
                strict(columns, grows, truncated, [(cols, crows)]) == expected
            )
            jobs.append((i["id"], name, expected, prompt, ok_strict))
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(lambda j: call_judge(j[3])["verdict"], jobs))
    wrong = []
    for (iid, name, expected, _, ok_strict), verdict in zip(jobs, verdicts):
        s = stats[name]
        s[0] += 1
        s[1] += (verdict == "correct") == expected
        s[2] += ok_strict
        if (verdict == "correct") != expected:
            wrong.append((iid, name, expected, verdict))
    print(f"{'control':26} {'n':>4} {'judge ok':>9} {'strict ok':>10}")
    tot = [0, 0, 0]
    for name, (n, j, st, _) in stats.items():
        print(f"{name:26} {n:4d} {j:9d} {st:10d}")
        tot[0] += n
        tot[1] += j
        tot[2] += st
    print(
        f"{'ALL':26} {tot[0]:4d} {tot[1]:9d} {tot[2]:10d}   judge {100*tot[1]/tot[0]:.1f}%  strict {100*tot[2]/tot[0]:.1f}%"
    )
    for w in wrong:
        print("JUDGE WRONG", w)


if __name__ == "__main__":
    main(sys.argv[1])
