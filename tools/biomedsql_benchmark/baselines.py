"""Baselines that show what the benchmark does not reward.

    python baselines.py BiomedSQL.csv OUT_PREFIX [llm]

* always empty: an empty answer to every question (strict).
* other item: BiomedSQL's result of another item of the same template, as the
  answer (strict): how often a plausible but wrong table passes.
* llm (optional): Sonnet answers each question from its own knowledge, with no
  database, as a text table; written to ``OUT_PREFIX.nodb.jsonl`` for judge_runs.py.
"""

import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import HERE, gold_table, load_biomedsql
from scorer import MODEL, strict

NODB_PROMPT = """Answer this question about a biomedical database (GWAS summary statistics of Parkinson's and Alzheimer's disease, SMR results for genes and disease, drug annotations and drug indications) from your own knowledge. You cannot query the database. Reply with the result table the database would return: a first line with the column names separated by ' | ', then one row per line. If you do not know, reply with a table of one row saying unknown.

Question: {question}"""


def ask(question):
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 800,
                "messages": [
                    {
                        "role": "user",
                        "content": NODB_PROMPT.format(question=question),
                    }
                ],
            }
        ).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_INRIA_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    reply = json.load(urllib.request.urlopen(request, timeout=120))
    return (
        "".join(
            b.get("text", "") for b in reply["content"] if b["type"] == "text"
        ),
        reply["usage"],
    )


def main(csv_path, prefix, llm=None):
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    items = json.load(open(HERE / "items.json"))["items"]
    rng = random.Random(0)
    by_template = {}
    for i in items:
        by_template.setdefault(i["template"], []).append(i["id"])
    for split in ("dev", "test"):
        sel = [i for i in items if i["split"] == split]
        empty = other = 0
        for i in sel:
            columns, rows, truncated = gold_table(by_id[i["id"]])
            empty += strict(columns, rows, truncated, [])
            pool = [x for x in by_template[i["template"]] if x != i["id"]]
            ocols, orows, _ = gold_table(by_id[rng.choice(pool)])
            other += strict(columns, rows, truncated, [(ocols, orows)])
        n_empty_gold = sum(not gold_table(by_id[i["id"]])[1] for i in sel)
        print(
            f"{split}: {len(sel)} items | gold empty {n_empty_gold} | always-empty {empty} "
            f"({100 * empty / len(sel):.1f}%) | another item of the same template {other} ({100 * other / len(sel):.1f}%)"
        )
    if llm:
        sel = [i for i in items if i["split"] in ("dev", "test")]
        with ThreadPoolExecutor(8) as pool:
            replies = list(
                pool.map(lambda i: ask(by_id[i["id"]]["question"]), sel)
            )
        with open(f"{prefix}.nodb.jsonl", "w") as f:
            for i, (text, usage) in zip(sel, replies):
                f.write(
                    json.dumps(
                        {
                            "id": i["id"],
                            "template": i["template"],
                            "split": i["split"],
                            "condition": "nodb",
                            "run": 1,
                            "valid": True,
                            "answer_text": text,
                            "usage": usage,
                        }
                    )
                    + "\n"
                )
        print("no-database answers:", len(sel))


if __name__ == "__main__":
    main(*sys.argv[1:])
