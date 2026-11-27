"""The two Sonnet arms of PROTOCOL_LADDER.md.

    python ladder_llm.py dry [ITEMS.json]          # prompts and cost, no call
    python ladder_llm.py run OUT.jsonl [ITEMS.json]   # PAID (OpenRouter)

One prompt per (item, question, arm), one probability each, JSON
{"probability": x} (run_llm.FORMAT), through run_same_llm.ask (OpenRouter,
cache llm_cache_or/, max_tokens 4000).
  flags: the other datasets as "dataset i | organ | significant", the gene
         masked, the held-out datasets named only by their organ;
  full:  the L2 table (dataset | tissue | kind | smallest p | significant),
         the gene named, the held-out datasets named with tissue and kind.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_llm import FORMAT

HERE = Path(__file__).resolve().parent
RUNGS = {"q14": ["Q1", "Q2", "Q3", "Q4"], "q5": ["Q5"], "q6": ["Q6"]}
IN_TOKENS, OUT_TOKENS, IN_PRICE, OUT_PRICE = 600, 500, 2e-6, 10e-6
HEAD = ("{who} was tested as a mediator of {disease} in several QTL "
        "datasets (summary-data-based Mendelian randomisation, SMR; a test "
        "is significant when p_SMR_multi < 2.95e-6). Three datasets are "
        "held out: {held}. ")
ASK = {
    "Q1": "What is the probability that {h1} finds {gene} significant "
          "for {disease} (p_SMR_multi < 2.95e-6)?",
    "Q2": "What is the probability that at least one of the three held-out "
          "datasets finds {gene} significant for {disease}?",
    "Q3": "What is the probability that all three held-out datasets find "
          "{gene} significant for {disease}?",
    "Q4": "What is the probability that at least two of the three held-out "
          "datasets find {gene} significant for {disease}?",
    "Q5": "What is the probability that at least one held-out brain dataset "
          "finds {gene} significant for {disease} while no held-out blood "
          "dataset does?",
    "Q6": "What is the probability that at least one of the three held-out "
          "datasets finds {gene} significant for both {d1} and {d2}?",
}


def yes(e):
    return "yes" if e["significant"] else "no"


def flags_prompt(item, q):
    held = ", ".join(f"held-out dataset {c} ({h['organ']})" for c, h in
                     zip("ABC", item["heldout"]))
    if q == "Q6":
        d1, d2 = item["disease_names"]
        order = {}
        for d in ("PD", "AD"):
            for e in item["evidence"][d]:
                order.setdefault(e["dataset"], {})[d] = e
        rows = []
        for n, (name, by) in enumerate(order.items(), 1):
            e = next(iter(by.values()))
            cells = " | ".join(yes(by[d]) if d in by else "not tested"
                               for d in ("PD", "AD"))
            rows.append(f"dataset {n} | {e['organ']} | {cells}")
        text = (HEAD.format(who="A gene (its name is withheld)",
                            disease=f"{d1} and {d2}", held=held)
                + f"The results of the other datasets (dataset | organ | "
                  f"significant for {d1} | significant for {d2}):\n"
                + "\n".join(rows) + "\n\n"
                + ASK[q].format(gene="the gene", d1=d1, d2=d2))
        return text + FORMAT
    rows = "\n".join(f"dataset {n} | {e['organ']} | {yes(e)}" for n, e in
                     enumerate(item["evidence"], 1))
    text = (HEAD.format(who="A gene (its name is withheld)",
                        disease=item["disease_name"], held=held)
            + "The results of the other datasets (dataset | organ | "
              "significant):\n" + rows + "\n\n"
            + ASK[q].format(gene="the gene", disease=item["disease_name"],
                            h1="held-out dataset A"))
    return text + FORMAT


def full_prompt(item, q):
    held = ", ".join(f"{h['dataset']} (a {h['tissue']} dataset, {h['kind']})"
                     for h in item["heldout"])

    def table(rows):
        return "\n".join(f"{e['dataset']} | {e['tissue']} | {e['kind']} | "
                         f"{e['p']:.3g} | {yes(e)}" for e in rows)

    cols = "(dataset | tissue | kind | smallest p_SMR_multi | significant)"
    gene = item["gene"]
    if q == "Q6":
        d1, d2 = item["disease_names"]
        text = (HEAD.format(who=f"Gene {gene}", disease=f"{d1} and {d2}",
                            held=held)
                + f"The tests of {gene} for {d1} in the other datasets "
                  f"{cols}:\n{table(item['evidence']['PD'])}\n\n"
                  f"The tests of {gene} for {d2} in the other datasets:\n"
                  f"{table(item['evidence']['AD'])}\n\n"
                + ASK[q].format(gene=gene, d1=d1, d2=d2))
        return text + FORMAT
    d = item["disease_name"]
    text = (HEAD.format(who=f"Gene {gene}", disease=d, held=held)
            + f"The tests of {gene} for {d} in the other datasets {cols}:\n"
            + table(item["evidence"]) + "\n\n"
            + ASK[q].format(gene=gene, disease=d, h1=item["heldout"][0][
                "dataset"]))
    return text + FORMAT


def jobs(items_file):
    data = json.load(open(HERE / items_file))
    out = []
    for key, qs in RUNGS.items():
        for item in data[key]:
            for q in qs:
                out.append((q, "flags", item, flags_prompt(item, q)))
                out.append((q, "full", item, full_prompt(item, q)))
    return out


def dry(items_file="results/ladder_items.json"):
    js = jobs(items_file)
    seen = set()
    for q, arm, item, prompt in js:
        if (q, arm) not in seen:
            seen.add((q, arm))
            print(f"===== {q} {arm} ({item['id']}) =====\n{prompt}\n")
    n = len(js)
    per = IN_TOKENS * IN_PRICE + OUT_TOKENS * OUT_PRICE
    print(f"{n} calls ({n // 2} per arm); estimated ${n * per:.2f} "
          f"({IN_TOKENS} in + {OUT_TOKENS} out tokens per call at "
          f"$2/$10 per million). No call was made.")
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        print(f"  {q}: {sum(j[0] == q for j in js)} calls")


def run(out_path, items_file="results/ladder_items.json"):
    from run_same_llm import ask
    js = jobs(items_file)
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda j: ask(j[3]), js))
    with open(out_path, "w") as f:
        for (q, arm, item, _), r in zip(js, replies):
            f.write(json.dumps({
                "id": item["id"], "question": q, "arm": arm,
                "probability": r["probability"], "usage": r["usage"],
                "finish": r["finish"]}) + "\n")
    cost = sum(IN_PRICE * r["usage"]["input_tokens"]
               + OUT_PRICE * r["usage"]["output_tokens"] for r in replies)
    print(f"{len(js)} answers; "
          f"{sum(r['probability'] is None for r in replies)} without a "
          f"probability; ${cost:.2f}")


if __name__ == "__main__":
    {"dry": dry, "run": run}[sys.argv[1]](*sys.argv[2:])
