"""Part 2 of PROTOCOL_SAME.md: L2-full and L2-flags through OpenRouter.

    python run_same_llm.py OUT.jsonl        # needs OPENROUTER_API_KEY

L2-full is the L2 prompt of run_llm.py unchanged; L2-flags gives only the
evidence of support_prior (which other datasets tested the pair and which
found it significant), with the gene, the held-out dataset's name, tissue and
kind withheld. Answers are cached in llm_cache_or/ (not run_llm.py's cache,
whose answers came from the direct API).
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_llm import FORMAT, question, table

HERE = Path(__file__).resolve().parent
CACHE = HERE / "llm_cache_or"
MODEL = "anthropic/claude-sonnet-5.5"
MAX_TOKENS = 4000

FLAGS = (
    "A gene (its name is withheld) was tested as a mediator of {disease_name} "
    "in several QTL datasets (summary-data-based Mendelian randomisation, SMR; "
    "a test is significant when p_SMR_multi < 2.95e-6). One further dataset is "
    "held out. The results of the other datasets (dataset | significant):\n"
    "{rows}\n\nWhat is the probability that the held-out dataset finds the "
    "gene significant (p_SMR_multi < 2.95e-6)?"
)


def flags(item):
    rows = "\n".join(
        f"dataset {i} | {'yes' if e['significant'] else 'no'}"
        for i, e in enumerate(item["evidence"], 1)
    )
    return FLAGS.format(disease_name=item["disease_name"], rows=rows)


def ask(prompt):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": MODEL, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"order": ["Anthropic"]},
        }).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "content-type": "application/json"},
    )
    for attempt in range(4):
        try:
            reply = json.load(urllib.request.urlopen(request, timeout=180))
            break
        except Exception:
            if attempt == 3:
                raise
    text = reply["choices"][0]["message"].get("content") or ""
    match = re.search(r'"probability"\s*:\s*([0-9.eE+-]+)', text)
    usage = reply.get("usage", {})
    out = {
        "probability": float(match.group(1)) if match else None,
        "text": text[:300],
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
        "host": reply.get("provider"),
        "finish": reply["choices"][0].get("finish_reason"),
    }
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def main(out_path):
    items = (json.load(open(HERE / "items.json"))
             + json.load(open(HERE / "items_eval.json")))
    jobs = []
    for item in items:
        jobs.append(("L2-full", item, question(item) + table(item) + FORMAT))
        jobs.append(("L2-flags", item, flags(item) + FORMAT))
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda j: ask(j[2]), jobs))
    with open(out_path, "w") as f:
        for (system, item, _), r in zip(jobs, replies):
            f.write(json.dumps({"system": system, "id": item["id"],
                                "probability": r["probability"],
                                "usage": r["usage"], "host": r["host"],
                                "finish": r["finish"]}) + "\n")
    cost = sum(2e-6 * r["usage"]["input_tokens"]
               + 10e-6 * r["usage"]["output_tokens"] for r in replies)
    print(f"{len(jobs)} answers; {sum(r['probability'] is None for r in replies)}"
          f" without a probability; cut at max_tokens "
          f"{sum(r['finish'] == 'length' for r in replies)}; ${cost:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
