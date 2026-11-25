"""Post hoc: is the information the share ignores enough to match L2?

    python post_hoc_logit.py SMR.parquet PREDICTIONS.parquet

A logistic regression on features of the other datasets' tests (their
p-values, organ and source), fitted on the held-out predictions with
support > 0 of the 222 genes of ``items.json``, evaluated on the 1,197 of
``items_eval.json`` (the 131 other genes), against ``support_prior`` and L2
(read from ``results/eval_answers.jsonl``). Decided after the result of
PROTOCOL_PRIOR.md, so exploratory.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from neurolang_apps.utils.engines.biomedsql.init import organ_of

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6
CLIP = (0.02, 0.98)
B = 2000
NAMES = [
    "log odds of the smoothed share",
    "log n",
    "strongest -log10 p",
    "mean -log10 p",
    "any same-organ dataset significant",
    "same-organ smoothed share",
    "share of other datasets in the same organ",
    "any same-source dataset significant",
]


def features(others, heldout, organ, source):
    sig = (others.p_SMR_multi < THRESHOLD).values
    n, k = len(others), int(sig.sum())
    strength = -np.log10(np.clip(others.p_SMR_multi.values, 1e-300, 1))
    same_o = np.array([organ[x] == organ[heldout] for x in others.Omic])
    same_s = np.array([source[x] == source[heldout] for x in others.Omic])
    return [
        np.log((k + 0.01) / (n - k + 0.5)),
        np.log(n),
        min(strength.max(), 15) / 15,
        min(strength.mean(), 10) / 10,
        (sig & same_o).sum() > 0,
        ((sig & same_o).sum() + 0.01) / (same_o.sum() + 0.51)
        if same_o.any() else 0.0,
        same_o.mean(),
        (sig & same_s).sum() > 0,
    ]


def fit_logistic(smr_path, predictions_path):
    """The logistic, fitted on the predictions of the genes of items.json;
    its ids, outcomes and probabilities on the other genes' predictions,
    and the model and the number fitted on."""
    smr = pd.read_parquet(
        smr_path,
        columns=["Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue",
                 "Source"],
    )
    info = smr.drop_duplicates("Omic").set_index("Omic")
    organ = {o: organ_of(str(t)) for o, t in info.Omic_tissue.items()}
    source = info.Source.to_dict()
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    by = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    fit_genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}

    X = np.array([
        features(
            (g := by[(r.gene, r.disease)])[g.Omic != r.heldout],
            r.heldout, organ, source,
        )
        for r in preds.itertuples()
    ], float)
    y = preds.y.values
    train = preds.gene.isin(fit_genes).values
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[train], y[train])
    ev = preds[~train].reset_index(drop=True)
    ye = y[~train]
    ids = [f"{r.gene}|{r.disease}|{r.heldout}" for r in ev.itertuples()]
    p = clf.predict_proba(X[~train])[:, 1]
    return ids, ye, p, clf, int(train.sum()), len(fit_genes), ev


def main(smr_path, predictions_path):
    ids, ye, p, clf, n_fit, n_genes, ev = fit_logistic(
        smr_path, predictions_path)
    answers = {}
    for line in open(HERE / "results/eval_answers.jsonl"):
        r = json.loads(line)
        answers[r["id"]] = r
    systems = {
        "support_prior (engine)": [answers[i]["support_prior"] for i in ids],
        "logistic on the table": p,
        "L2, Sonnet + the evidence table": [answers[i]["L2"] for i in ids],
    }
    systems = {k: np.clip(np.asarray(v, float), *CLIP)
               for k, v in systems.items()}

    print(f"fitted on {n_fit} predictions ({n_genes} genes); "
          f"evaluated on {len(ev)} ({ev.gene.nunique()} genes)")
    print("coefficients:")
    for name, c in zip(NAMES, clf.coef_[0]):
        print(f"  {c:+.2f}  {name}")
    for k, v in systems.items():
        print(f"{k:32} Brier {np.mean((v - ye) ** 2):.3f}  "
              f"AUC {roc_auc_score(ye, v):.3f}")

    groups = {}
    for j, g in enumerate(ev.gene.values):
        groups.setdefault(g, []).append(j)
    keys = sorted(groups)
    rng = random.Random(0)
    draws = [
        [j for g in (rng.choice(keys) for _ in keys) for j in groups[g]]
        for _ in range(B)
    ]

    def diff(a, b):
        pa, pb = systems[a], systems[b]
        est = np.mean((pa - ye) ** 2) - np.mean((pb - ye) ** 2)
        bs = np.sort([
            np.mean((pa[d] - ye[d]) ** 2) - np.mean((pb[d] - ye[d]) ** 2)
            for d in draws
        ])
        return f"{est:+.3f} [{bs[int(.025 * B)]:+.3f}, " \
               f"{bs[int(.975 * B) - 1]:+.3f}]"

    print("Brier, logistic - support_prior:",
          diff("logistic on the table", "support_prior (engine)"))
    print("Brier, logistic - L2:           ",
          diff("logistic on the table", "L2, Sonnet + the evidence table"))


if __name__ == "__main__":
    main(*sys.argv[1:])
