"""S4 of PROTOCOL.md: learned guidance by template cross-validation.

    python induce_guidance.py BiomedSQL.csv BASELINE_RUN.judged.jsonl OUT_DIR

Folds: the 40 templates, stratified by table, shuffled (seed 0), dealt in 4
folds. For fold f, Sonnet reads the judged failures of the baseline run on the
templates of the other folds and writes at most 12 general rules; the rules
for fold f never see a template of fold f. Writes ``folds.json`` and
``guidance_f.txt`` (f = 0..3) in OUT_DIR.
"""

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import gold_table, load_biomedsql  # noqa: E402
from scorer import MODEL  # noqa: E402

PROMPT = """An agent answers natural-language questions about a biomedical database by writing Datalog queries (NeuroLang). Below are questions it answered wrongly, with the gold result's size, its query, and a reviewer's note on what was wrong. The database holds GWAS summary statistics (Parkinson's, Alzheimer's), SMR results linking genes to diseases across QTL datasets, drug annotations and drug indications.

Write at most 12 short general rules (one or two sentences each) that would have prevented these mistakes on other questions of the same kinds: conventions of the database, when to apply a threshold, what to return, how to match names. Rules must be general. Do not name any gene, drug, SNP, disease or identifier from the examples. Do not mention the examples. Reply with the rules only, numbered.

{cases}
"""


def folds_of(items):
    by_table = {}
    for i in items:
        by_table.setdefault(i["table"], set()).add(i["template"])
    rng = random.Random(0)
    folds = [[] for _ in range(4)]
    k = 0
    for table in sorted(by_table):
        templates = sorted(by_table[table], key=lambda t: int(t[1:]))
        rng.shuffle(templates)
        for t in templates:
            folds[k % 4].append(t)
            k += 1
    return folds


def ask(prompt):
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 2500,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_INRIA_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    reply = json.load(urllib.request.urlopen(request, timeout=300))
    return "".join(
        b.get("text", "") for b in reply["content"] if b["type"] == "text"
    )


def main(csv_path, baseline, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = json.load(open(HERE.parent / "items.json"))["items"]
    folds = folds_of(items)
    (out / "folds.json").write_text(json.dumps(folds))
    by_id = {r["uuid"]: r for r in load_biomedsql(csv_path)}
    recs = [json.loads(l) for l in open(baseline)]
    for f, held in enumerate(folds):
        cases = []
        for r in recs:
            if r["template"] in held or r["judge"] == "correct":
                continue
            columns, rows, truncated = gold_table(by_id[r["id"]])
            gold = (
                "the gold result is empty"
                if not rows
                else f"the gold result has {len(rows)} row(s)"
                f"{' (a list cut by its limit)' if truncated else ''} and "
                f"columns {columns}"
            )
            cases.append(
                f"Question: {r['question']}\n{gold}\n"
                f"Agent's query: {(r.get('query') or '')[:700]}\n"
                f"Reviewer: {r['judge_note'][:300]}\n"
            )
        text = ask(PROMPT.format(cases="\n".join(cases)))
        (out / f"guidance_{f}.txt").write_text(text.strip() + "\n")
        (out / f"cases_{f}.txt").write_text("\n".join(cases))
        print(
            f"fold {f}: {len(held)} templates held out, "
            f"{len(cases)} failures read"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
