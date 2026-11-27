"""Draw the items of the replication experiment (see PROTOCOL.md).

    python build_items.py SMR.parquet PREDICTIONS.parquet

PREDICTIONS.parquet is the output of ``tools/biomedsql_calibration.py``.
Writes ``items.json``.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6
SEED = 0
N = 240
DISEASES = {
    "AD": "Alzheimer's disease",
    "PD": "Parkinson's disease",
    "ALS": "amyotrophic lateral sclerosis",
    "FTD": "frontotemporal dementia",
    "LBD": "Lewy body dementia",
    "PSP": "progressive supranuclear palsy",
}


def load_tests(smr_path):
    """One row per (dataset, gene, disease): the smallest p_SMR_multi."""
    cols = [
        "Omic",
        "Gene",
        "Disease",
        "p_SMR_multi",
        "Omic_tissue",
        "Omic_type",
    ]
    smr = pd.read_parquet(smr_path, columns=cols)
    info = smr.drop_duplicates("Omic").set_index("Omic")[
        ["Omic_tissue", "Omic_type"]
    ]
    tests = smr.groupby(
        ["Omic", "Gene", "Disease"], observed=True
    ).p_SMR_multi.min()
    return tests.reset_index(), info


def main(
    smr_path, predictions_path, out="items.json", seed=SEED, exclude=None
):
    """Draw N items; with *exclude* (an items file), no gene of that file."""
    tests, info = load_tests(smr_path)
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    if exclude:
        used = {i["gene"] for i in json.load(open(HERE / exclude))}
        preds = preds[~preds.gene.isin(used)]
    rng = random.Random(int(seed))
    pairs = sorted(set(zip(preds.gene, preds.disease)))
    rng.shuffle(pairs)
    by_pair = {k: g for k, g in preds.groupby(["gene", "disease"])}
    by_gene_disease = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    items = []
    for gene, disease in pairs[:N]:
        candidates = sorted(by_pair[(gene, disease)].heldout)
        h = rng.choice(candidates)
        rows = by_gene_disease[(gene, disease)]
        held = rows[rows.Omic == h].p_SMR_multi.iloc[0]
        evidence = [
            {
                "dataset": r.Omic,
                "tissue": info.loc[r.Omic, "Omic_tissue"],
                "kind": info.loc[r.Omic, "Omic_type"],
                "p": float(r.p_SMR_multi),
                "significant": bool(r.p_SMR_multi < THRESHOLD),
            }
            for r in rows.itertuples()
            if r.Omic != h
        ]
        evidence.sort(key=lambda e: e["p"])
        n_sig = sum(e["significant"] for e in evidence)
        items.append(
            {
                "id": f"{gene}|{disease}|{h}",
                "gene": gene,
                "disease": disease,
                "disease_name": DISEASES[disease],
                "heldout": h,
                "tissue": info.loc[h, "Omic_tissue"],
                "kind": info.loc[h, "Omic_type"],
                "y": int(held < THRESHOLD),
                "support": n_sig / len(evidence),
                "n_other": len(evidence),
                "evidence": evidence,
            }
        )
    (HERE / out).write_text(json.dumps(items, indent=1))
    y = [i["y"] for i in items]
    print(
        len(items),
        "items; base rate",
        sum(y) / len(y),
        "| genes",
        len({i["gene"] for i in items}),
        "| n_other median",
        sorted(i["n_other"] for i in items)[len(items) // 2],
    )
    exact = sum(
        abs(
            i["support"]
            - preds[
                (preds.gene == i["gene"])
                & (preds.disease == i["disease"])
                & (preds.heldout == i["heldout"])
            ].p.iloc[0]
        )
        < 1e-9
        for i in items
    )
    print(
        "support equals the engine's prediction for", exact, "of", len(items)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
