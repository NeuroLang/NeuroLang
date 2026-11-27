"""Judge the stored answers of agent runs.

    python judge_runs.py BiomedSQL.csv RUNS.jsonl [RUNS2.jsonl ...]

Writes ``RUNS.judged.jsonl`` with the judge's verdict added to each record. The
judge sees the question, BiomedSQL's reference answer and result table, and the
answer table (or a boolean, or nothing): never the query or the condition.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import gold_table, load_biomedsql, table_text
from scorer import call_judge, judge_prompt


def answer_text(rec):
    if not rec.get("valid") and "answer" not in rec:
        return "(no result: the system returned no valid answer)"
    parts = []
    if rec.get("boolean") is not None and not rec.get("answer"):
        parts.append(f"The system answered: {rec['boolean']}")
    for rel in rec.get("answer", []):
        text = table_text(rel["columns"], [tuple(r) for r in rel["rows"]])
        if rel["n"] > len(rel["rows"]):
            text += f"\n(the result has {rel['n']} rows in all)"
        parts.append(text)
    return "\n\n".join(parts) if parts else "(no result)"


def main(csv_path, *paths):
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    for path in paths:
        recs = [json.loads(l) for l in open(path)]
        prompts = []
        for rec in recs:
            item = by_id[rec["id"]]
            columns, rows, truncated = gold_table(item)
            text = (
                rec["answer_text"]
                if "answer_text" in rec
                else answer_text(rec)
            )
            prompts.append(judge_prompt(item, columns, rows, truncated, text))
        with ThreadPoolExecutor(8) as pool:
            verdicts = list(pool.map(call_judge, prompts))
        out = Path(path).with_suffix(".judged.jsonl")
        with open(out, "w") as f:
            for rec, v in zip(recs, verdicts):
                rec = {k: val for k, val in rec.items() if k != "answer"} | {
                    "judge": v["verdict"],
                    "judge_note": v["reason"][:300],
                }
                f.write(json.dumps(rec) + "\n")
        n = sum(v["verdict"] == "correct" for v in verdicts)
        print(
            f"{path}: judge {n}/{len(recs)}, unparsed {sum(v['verdict'] == 'unparsed' for v in verdicts)}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
