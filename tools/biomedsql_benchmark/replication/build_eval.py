"""Evaluation items of PROTOCOL_PRIOR.md: every held-out prediction with
support > 0 of the genes that are not among those of ``items.json``.

    python build_eval.py SMR.parquet PREDICTIONS.parquet

Writes ``items_eval.json`` (same fields as ``items.json``).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from build_items import DISEASES, THRESHOLD, load_tests

HERE = Path(__file__).resolve().parent


def main(smr_path, predictions_path):
    tests, info = load_tests(smr_path)
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    used = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    preds = preds[~preds.gene.isin(used)]
    by = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    items = []
    for r in preds.itertuples():
        rows = by[(r.gene, r.disease)]
        held = rows[rows.Omic == r.heldout].p_SMR_multi.iloc[0]
        evidence = [
            {
                "dataset": e.Omic,
                "tissue": info.loc[e.Omic, "Omic_tissue"],
                "kind": info.loc[e.Omic, "Omic_type"],
                "p": float(e.p_SMR_multi),
                "significant": bool(e.p_SMR_multi < THRESHOLD),
            }
            for e in rows.itertuples()
            if e.Omic != r.heldout
        ]
        evidence.sort(key=lambda e: e["p"])
        n_sig = sum(e["significant"] for e in evidence)
        items.append(
            {
                "id": f"{r.gene}|{r.disease}|{r.heldout}",
                "gene": r.gene,
                "disease": r.disease,
                "disease_name": DISEASES[r.disease],
                "heldout": r.heldout,
                "tissue": info.loc[r.heldout, "Omic_tissue"],
                "kind": info.loc[r.heldout, "Omic_type"],
                "y": int(held < THRESHOLD),
                "support": n_sig / len(evidence),
                "n_other": len(evidence),
                "evidence": evidence,
            }
        )
    (HERE / "items_eval.json").write_text(json.dumps(items))
    y = [i["y"] for i in items]
    print(
        len(items),
        "items; genes",
        len({i["gene"] for i in items}),
        "; base rate",
        round(sum(y) / len(y), 3),
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
