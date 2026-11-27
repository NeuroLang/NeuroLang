"""Build bank.json: one worked example per development template.

For each development template, one item that is not among the benchmark's four
for it (drawn at random, seed 0); its question is BiomedSQL's, its query the oracle's
(BiomedSQL's gold SQL translated mechanically), kept only if it runs on the engine
and reproduces the gold result (strict). Test templates have no example.
"""

import collections
import json
import random
import sys
from pathlib import Path

from common import HERE, gold_table, load_biomedsql
from oracle import oracle
from run_oracle import columns_of, relations
from scorer import strict


def main(csv_path, data_dir):
    from neurolang_apps.utils import engine_registry

    nl = engine_registry.build_engine(
        "biomedsql", data_dir=Path(data_dir), use_cache=False
    )
    rows = load_biomedsql(csv_path)
    chosen = {i["id"] for i in json.load(open(HERE / "items.json"))["items"]}
    split = {
        i["template"]: i["split"]
        for i in json.load(open(HERE / "items.json"))["items"]
    }
    by = collections.defaultdict(list)
    for r in rows:
        if split[r["template_uuid"]] == "dev" and r["uuid"] not in chosen:
            by[r["template_uuid"]].append(r)
    cols = columns_of()
    bank = []
    for t in sorted(by, key=lambda t: int(t[1:])):
        pool = sorted(by[t], key=lambda r: r["uuid"])
        random.Random(f"bank/{t}").shuffle(pool)
        for item in pool[:6]:
            columns, grows, truncated = gold_table(item)
            if not grows:  # an example with an empty answer teaches little
                continue
            program = oracle(item["benchmark_query"], cols)
            with nl.scope:
                result = nl.execute_datalog_program(program)
            if strict(columns, grows, truncated, relations(result)):
                bank.append(
                    {
                        "template": t,
                        "question": item["question"],
                        "datalog": program,
                    }
                )
                break
        print(
            t,
            "ok" if bank and bank[-1]["template"] == t else "no example",
            flush=True,
        )
    (HERE / "bank.json").write_text(json.dumps(bank, indent=1))
    print(len(bank), "examples")


if __name__ == "__main__":
    main(*sys.argv[1:])
