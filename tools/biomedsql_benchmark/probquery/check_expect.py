"""The expected-number questions (P7) written with EXPECT.

An agent writes "the expected number of units" as a sum over a PROB rule,

    p(u, PROB) :- <event with u and projected variables>
    ans(s) :- AGGREGATE[](p(u, q) @ sum(q)) = s

``EXPECT[1 | <event>] = s`` is that program: it sums over the variables of
its event, so the projected ones are written under an ``exists``. This runs
both forms of every P7 question on the tables of the benchmark, with the
lifted solvers of ``engine.build``, and compares them to each other and to
the gold values.

    uv run --no-sync python check_expect.py

P7a needs ``drug_indication`` only of the BiomedSQL tables (7 MB), which is
read here directly; the other four (1.1 GB) are not loaded.
"""
import json
import sys
import time

import pandas as pd

from build_tables import DB
from check_lifted import REFERENCE, listed
from engine import build
from gold import HERE

INDICATION = "DrugTargets_IndicationsAndTherapeuticUses.parquet"

EXPECT = {
    "P7a": "{listed}ans(s) :- EXPECT[1 | listed(u), drug_approved(u), "
           "exists(g, d; drug_target(u, g), test_evidence(d, g, '{dis}'))]"
           " = s",
    "P7b": "ans(s) :- EXPECT[1 | drug_target('{x}', u), "
           "exists(d; test_evidence(d, u, '{dis}'), dataset_reliable(d))]"
           " = s",
    "P7c": "ans(s) :- EXPECT[1 | dataset_reliable(u), "
           "exists(g; drug_target('{x}', g), test_evidence(u, g, '{dis}'))]"
           " = s",
}


def value(nl, program):
    with nl.scope:
        rows = [tuple(r) for r in nl.execute_datalog_program(program)]
    return float(rows[0][-1]) if rows else 0.0


def main():
    nl = build(with_biomedsql=False)
    nl.add_tuple_set(
        pd.read_parquet(DB / INDICATION).itertuples(index=False, name=None),
        name="drug_indication",
    )
    questions = json.load(open(HERE / "questions.json"))
    gold = json.load(open(HERE / "gold.json"))
    bad = 0
    for q in questions:
        template = q["template"]
        if template not in EXPECT:
            continue
        if q.get("engine_refused"):
            continue
        p = q["params"]
        fields = dict(listed=listed(p["m"]) if "m" in p else "", **p)
        t = time.time()
        before = value(nl, REFERENCE[template].format(**fields))
        after = value(nl, EXPECT[template].format(**fields))
        expected = gold[q["id"]]["gold"]
        ok = abs(before - after) <= 1e-9 * max(1, before) and (
            abs(after - expected) <= 1e-9 * max(1, after)
        )
        bad += not ok
        print(f"{q['id']}: {'ok' if ok else 'MISMATCH'} before {before:.9f} "
              f"EXPECT {after:.9f} gold {expected:.9f} "
              f"{time.time() - t:.1f}s", flush=True)
    print(f"{bad} mismatches")
    return bad


if __name__ == "__main__":
    sys.exit(bool(main()))
