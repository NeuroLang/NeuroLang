"""Systems L1 (no data) and L2 (with the evidence table) of PROTOCOL.md.

python run_llm.py OUT.jsonl
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "claude-sonnet-5-5"

QUESTION = (
    "Gene {gene} was tested as a mediator of {disease} in several QTL "
    "datasets (summary-data-based Mendelian randomisation, SMR; a test is "
    "significant when p_SMR_multi < 2.95e-6). Using only the datasets other "
    "than {heldout} (a {tissue} dataset, {kind}), what is the probability "
    "that {heldout} finds {gene} significant (p_SMR_multi < 2.95e-6)?"
)
FORMAT = (
    '\n\nReply with JSON only: {{"probability": <a number between 0 and 1>}}.'
)
TABLE = (
    "\n\nThe tests of {gene} for {disease} in the other datasets "
    "(dataset | tissue | kind | smallest p_SMR_multi | significant):\n{rows}"
)


def question(item):
    return QUESTION.format(**item)


def table(item):
    rows = "\n".join(
        f"{e['dataset']} | {e['tissue']} | {e['kind']} | {e['p']:.3g} | "
        f"{'yes' if e['significant'] else 'no'}"
        for e in item["evidence"]
    )
    return TABLE.format(
        gene=item["gene"], disease=item["disease_name"], rows=rows
    )


def ask(prompt):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    path = HERE / "llm_cache" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_INRIA_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    reply = json.load(urllib.request.urlopen(request, timeout=120))
    text = "".join(
        b.get("text", "") for b in reply["content"] if b["type"] == "text"
    )
    match = re.search(r'"probability"\s*:\s*([0-9.eE+-]+)', text)
    out = {
        "probability": float(match.group(1)) if match else None,
        "text": text[:300],
        "usage": reply["usage"],
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def main(out_path, items_file="items.json", systems="L1,L2"):
    items = json.load(open(HERE / items_file))
    wanted = systems.split(",")
    jobs = []
    for item in items:
        if "L1" in wanted:
            jobs.append(("L1", item, question(item) + FORMAT))
        if "L2" in wanted:
            jobs.append(("L2", item, question(item) + table(item) + FORMAT))
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda j: ask(j[2]), jobs))
    with open(out_path, "w") as f:
        for (system, item, _), reply in zip(jobs, replies):
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "id": item["id"],
                        "probability": reply["probability"],
                        "usage": reply["usage"],
                    }
                )
                + "\n"
            )
    missing = sum(r["probability"] is None for r in replies)
    print(len(jobs), "answers;", missing, "without a probability")


if __name__ == "__main__":
    main(*sys.argv[1:])
