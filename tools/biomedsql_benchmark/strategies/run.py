"""Run the agent on the benchmark's 160 items for a strategy (PROTOCOL.md).

    python run.py BiomedSQL.csv DATA_DIR TAG RUN OUT.jsonl [options]

options: --guidance FILE   rules appended to the agent's prompt (S1, S4)
         --verify          verify_rounds = 1 (S2)
         --templates Q1,Q2 only these templates (S4: the templates of a fold)

A condition without --guidance is the baseline A: the schema only.
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import gold_table, load_biomedsql  # noqa: E402
from run_agent import answer_relations, usd  # noqa: E402
from scorer import strict  # noqa: E402

CAP_USD = 30.0


class Alarm(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("data_dir")
    ap.add_argument("tag")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--guidance")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--templates")
    args = ap.parse_args()

    from neurolang_apps.utils import engine_registry
    from neurolang_apps.utils.llm.agent import ReactAgent
    from neurolang_apps.utils.llm.config import load_config

    nl = engine_registry.build_engine(
        "biomedsql", data_dir=Path(args.data_dir), use_cache=False
    )
    config = load_config(str(HERE.parent / "llm.yaml"))
    a = config.assistant
    a.strategy, a.max_steps, a.comment_rules = "react", 16, False
    a.verify_rounds = 1 if args.verify else 0
    a.query_timeout, a.query_memory_mb = 300, 30000.0
    provider = config.provider("anthropic-inria")
    guidance = Path(args.guidance).read_text() if args.guidance else ""
    by_id = {r["uuid"]: r for r in load_biomedsql(args.csv)}
    items = json.load(open(HERE.parent / "items.json"))["items"]
    if args.templates:
        wanted = set(args.templates.split(","))
        items = [i for i in items if i["template"] in wanted]
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    spent = (
        sum(json.loads(l).get("usd", 0) for l in open(out))
        if out.exists()
        else 0.0
    )
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Alarm()))
    with open(out, "a") as f:
        for i in items:
            if i["id"] in done:
                continue
            item = by_id[i["id"]]
            columns, rows, truncated = gold_table(item)
            rec = {
                "id": i["id"],
                "template": i["template"],
                "split": i["split"],
                "condition": args.tag,
                "run": int(args.run),
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
                    examples=[],
                    guidance=guidance,
                )
                res = agent.run(item["question"])
                u = dict(res.usage)
                rec.update(
                    query=res.query,
                    valid=res.valid,
                    calls=res.calls,
                    usage=u,
                    usd=usd(u),
                    error=res.error,
                    verified=res.verified,
                )
                relations = []
                if res.valid:
                    signal.alarm(300)
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
                    strict=False, error=f"{type(e).__name__}: {e}"[:200]
                )
            rec["seconds"] = round(time.time() - t0, 1)
            spent += rec.get("usd", 0)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                args.tag,
                args.run,
                i["id"],
                rec["strict"],
                f"{rec['seconds']}s ${spent:.2f}",
                flush=True,
            )
            if spent > CAP_USD:
                print("SPEND CAP REACHED")
                break


if __name__ == "__main__":
    main()
