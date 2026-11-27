"""Run the oracle on every item through the engine; score it with the strict scorer.

python run_oracle.py BiomedSQL.csv DATA_DIR OUT.jsonl [split]
"""

import json
import signal
import sys
import time
from pathlib import Path

import yaml

from common import HERE, canon, gold_table, load_biomedsql
from oracle import RELATIONS, oracle
from scorer import strict


class Timeout(Exception):
    pass


def columns_of():
    path = (
        HERE.parents[1]
        / "packages/neurolang_apps/neurolang_apps/utils/engines/engines.yaml"
    )
    predicates = yaml.safe_load(open(path))["engines"]["biomedsql"][
        "predicates"
    ]
    return {r: predicates[r]["columns"] for r in RELATIONS.values()}


def relations(result):
    """Answer relations (columns, rows) of an execution result, rows canonical."""
    if isinstance(result, bool):
        return []
    if isinstance(result, dict):
        return [r for v in result.values() for r in relations(v)]
    rows = [tuple(canon(x) for x in row) for row in result]
    width = len(rows[0]) if rows else 0
    return [([f"c{i}" for i in range(width)], rows)]


def main(csv_path, data_dir, out_path, split=None):
    from neurolang_apps.utils import engine_registry

    nl = engine_registry.build_engine(
        "biomedsql", data_dir=Path(data_dir), use_cache=False
    )
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    items = [
        i
        for i in json.load(open(HERE / "items.json"))["items"]
        if split in (None, i["split"])
    ]
    cols = columns_of()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
    done = (
        {json.loads(l)["id"] for l in open(out_path)}
        if Path(out_path).exists()
        else set()
    )
    with open(out_path, "a") as out:
        for i in items:
            if i["id"] in done:
                continue
            item = by_id[i["id"]]
            columns, rows, truncated = gold_table(item)
            rec = {
                "id": i["id"],
                "template": i["template"],
                "split": i["split"],
                "n_gold": len(rows),
                "truncated": truncated,
            }
            t0 = time.time()
            try:
                program = oracle(item["benchmark_query"], cols)
                rec["program"] = program
                signal.alarm(900)
                with nl.scope:
                    result = nl.execute_datalog_program(program)
                rec["strict"] = strict(
                    columns, rows, truncated, relations(result)
                )
            except Timeout:
                rec["error"] = "timeout"
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                signal.alarm(0)
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                i["id"],
                rec.get("strict"),
                rec.get("error", ""),
                rec["seconds"],
                flush=True,
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
