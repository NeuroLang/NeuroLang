"""Template split and blind item selection of the BiomedSQL benchmark.

    python items.py BiomedSQL.csv

writes ``items.json``: for each of the 40 templates, its split (development
or test, stratified by table, seed 0) and 4 items drawn uniformly at random
(seed 0), whatever their answer (empty and truncated ones included). Nothing
here looks at an agent's output.
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

from common import HERE, load_biomedsql, sha256

SEED = 0
PER_TEMPLATE = 4


def table_of(sql):
    tables = sorted(set(re.findall(r"\.(\w+)`", sql)))
    return "+".join(tables)


def main(path):
    rows = load_biomedsql(path)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["template_uuid"]].append(r)
    for items in by.values():
        items.sort(key=lambda r: r["uuid"])
    strata = collections.defaultdict(list)
    for t, items in by.items():
        strata[table_of(items[0]["benchmark_query"])].append(t)
    rng = random.Random(SEED)
    split = {}
    flip = rng.random() < 0.5
    for stratum in sorted(strata):
        templates = sorted(strata[stratum], key=lambda t: int(t[1:]))
        rng.shuffle(templates)
        for k, t in enumerate(templates):
            dev = (k % 2 == 0) == flip
            split[t] = "dev" if dev else "test"
        if len(templates) % 2:
            flip = not flip
    out = []
    for t in sorted(by, key=lambda t: int(t[1:])):
        picks = random.Random(f"{SEED}/{t}").sample(by[t], PER_TEMPLATE)
        for r in picks:
            out.append(
                {
                    "id": r["uuid"],
                    "template": t,
                    "split": split[t],
                    "table": table_of(r["benchmark_query"]),
                }
            )
    spec = {
        "seed": SEED,
        "per_template": PER_TEMPLATE,
        "source_sha256": sha256(path),
        "items": out,
    }
    Path(HERE / "items.json").write_text(json.dumps(spec, indent=1))
    count = collections.Counter(split.values())
    print("templates:", dict(count), "| items:", len(out))
    for s in ("dev", "test"):
        print(
            s,
            sorted(
                (t for t in split if split[t] == s), key=lambda t: int(t[1:])
            ),
        )
    print(
        "per table and split:",
        dict(
            collections.Counter(
                (table_of(by[t][0]["benchmark_query"])[:28], split[t])
                for t in split
            )
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1])
