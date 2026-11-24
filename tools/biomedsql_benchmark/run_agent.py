"""Run the ReAct agent on the benchmark items, in one of three conditions.

    python run_agent.py BiomedSQL.csv DATA_DIR CONDITION SPLIT RUN OUT.jsonl

CONDITION
    A  the schema only: the engine's description of its relations, no guidance, no examples
    B  A and the engine's guidance (``guidance.txt``, written from development templates)
    C  B and a bank of worked examples from development templates (``bank.json``)
SPLIT   dev | test
RUN     a number, to repeat a condition

The agent is the repository's ``ReactAgent`` (Sonnet through the Inria key, Datalog,
at most 16 steps, no verification step). The final query is run again here; its
answer is stored for the judge (``judge_runs.py``) and scored with the strict scorer.
"""

import json
import signal
import sys
import time
from pathlib import Path

from common import HERE, canon, gold_table, load_biomedsql
from scorer import strict

PRICE = {
    "input": 2.5,
    "cached": 0.10,
    "output": 10.0,
}  # USD per million tokens (Sonnet 5.5)
CAP_USD = 25.0
MAX_STEPS = 16
QUERY_TIMEOUT = 300


class Alarm(Exception):
    pass


def usd(usage):
    return (
        (usage.get("input", 0) - usage.get("cached", 0)) * PRICE["input"]
        + usage.get("cached", 0) * PRICE["cached"]
        + usage.get("output", 0) * PRICE["output"]
    ) / 1e6


def answer_relations(result, keep=300):
    """[(columns, rows)] of a result, rows canonical; the stored rows are cut at *keep*."""
    if isinstance(
        result, bool
    ):  # True is a non-empty answer, False an empty one
        rel = [(["answer"], [("true",)])] if result else []
        return rel, rel
    if isinstance(result, dict):
        rels = [answer_relations(v, keep) for v in result.values()]
        return [r for full, _ in rels for r in full], [
            r for _, cut in rels for r in cut
        ]
    rows = list(result)
    if not rows:
        return [([], [])], [([], [])]
    fields = list(
        getattr(rows[0], "_fields", None)
        or [f"c{i}" for i in range(len(rows[0]))]
    )
    canonical = [tuple(canon(x) for x in row) for row in rows]
    return [(fields, canonical)], [(fields, canonical[:keep])]


def main(csv_path, data_dir, condition, split, run, out_path):
    if split == "test":
        from freeze import check

        changed = check()
        if changed:
            sys.exit(
                f"the test run depends on files changed since FROZEN.json: {changed}"
            )
    from neurolang_apps.utils import engine_registry
    from neurolang_apps.utils.llm.agent import ReactAgent
    from neurolang_apps.utils.llm.config import load_config
    from neurolang_apps.utils.llm.examples import Example

    nl = engine_registry.build_engine(
        "biomedsql", data_dir=Path(data_dir), use_cache=False
    )
    config = load_config(str(HERE / "llm.yaml"))
    a = config.assistant
    a.strategy, a.max_steps, a.verify_rounds, a.comment_rules = (
        "react",
        MAX_STEPS,
        0,
        False,
    )
    a.query_timeout, a.query_memory_mb = QUERY_TIMEOUT, 30000.0
    provider = config.provider("anthropic-inria")
    guidance = "" if condition == "A" else (HERE / "guidance.txt").read_text()
    examples = []
    if condition == "C":
        examples = [
            Example(b["question"], {"datalog": b["datalog"]})
            for b in json.load(open(HERE / "bank.json"))
        ]
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    items = [
        i
        for i in json.load(open(HERE / "items.json"))["items"]
        if i["split"] == split
    ]
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Alarm()))
    done = (
        {json.loads(l)["id"] for l in open(out_path)}
        if Path(out_path).exists()
        else set()
    )
    spent = (
        sum(json.loads(l).get("usd", 0) for l in open(out_path))
        if Path(out_path).exists()
        else 0.0
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
                "split": split,
                "condition": condition,
                "run": int(run),
                "question": item["question"],
            }
            t0 = time.time()
            try:
                agent = ReactAgent(
                    nl,
                    provider,
                    a,
                    "datalog",
                    engine_name="biomedsql",
                    examples=examples,
                    guidance=guidance,
                )
                res = agent.run(item["question"])
                usage = dict(res.usage)
                rec.update(
                    query=res.query,
                    valid=res.valid,
                    out_of_steps=res.out_of_steps,
                    calls=res.calls,
                    usage=usage,
                    usd=usd(usage),
                    error=res.error,
                )
                relations = []
                if res.valid:
                    signal.alarm(QUERY_TIMEOUT)
                    try:
                        with nl.scope:
                            result = nl.execute_datalog_program(res.query)
                    finally:
                        signal.alarm(0)
                    rec["boolean"] = (
                        result if isinstance(result, bool) else None
                    )
                    full, cut = answer_relations(result)
                    relations = full
                    rec["answer"] = [
                        {"columns": c, "n": len(r), "rows": r} for c, r in cut
                    ]
                rec["strict"] = strict(columns, rows, truncated, relations)
            except Alarm:
                rec.update(strict=False, error="final query timed out")
            except Exception as e:  # noqa: BLE001
                rec.update(
                    strict=False, error=f"{type(e).__name__}: {str(e)[:200]}"
                )
            rec["seconds"] = round(time.time() - t0, 1)
            spent += rec.get("usd", 0)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                condition,
                split,
                run,
                i["id"],
                "strict" if rec["strict"] else "no",
                rec.get("error", "")[:60],
                f"{rec['seconds']}s ${spent:.2f}",
                flush=True,
            )
            if spent > CAP_USD:
                print("SPEND CAP REACHED")
                break


if __name__ == "__main__":
    main(*sys.argv[1:])
