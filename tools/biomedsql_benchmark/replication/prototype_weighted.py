"""Choose the form of weighted units of evidence, on the fitting genes only.

    python prototype_weighted.py SMR.parquet PREDICTIONS.parquet

Every candidate is a conditional probability over a choice of unit of
evidence (the 28 QTL datasets and the two imaginary units of
``support_prior``), so the engine can solve it exactly:

* P  ``support_prior``: (k + a) / (n + a + b).
* G  graded evidence: each test is evidence with a probability pi(bin of
  -log10 p), monotone; (sum pi + a) / (n + a + b).
* W  weighted datasets: (sum w_u sig_u + a) / (sum w_u + a + b).
* GW both.
* O  G restricted to the datasets of the held-out dataset's organ (a query
  condition), the imaginary units as back-off.
* GL G with the datasets of another organ than the held-out dataset's
  weighted lambda (a probabilistic fact organ_relevance(u, organ) in the
  query). Added after the first run of this script (P, G, O, W, GW), which
  showed the hard restriction O failing; still on the fitting genes only.

Fitted by log-loss, 5-fold cross-validation grouped by gene, on the held-out
predictions with support > 0 of the 222 genes of ``items.json`` only (the
131 evaluation genes are not read). Rule fixed before running: the lowest CV
log-loss (probabilities clipped to [0.02, 0.98]); within 0.002 of it the
simpler form wins (P < G < O < GL < W < GW).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from neurolang_apps.utils.engines.biomedsql.init import organ_of

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6
CLIP = (0.02, 0.98)
# bins of -log10 p; 5.53 is BiomedSQL's threshold
EDGES = np.array([1.0, 2.0, 3.0, -np.log10(THRESHOLD), 8.0, 12.0])
ORDER = ["P", "G", "O", "GL", "W", "GW"]
FOLDS = 5
ENGINE_PRIOR = (0.01, 0.5)   # engines.yaml, fitted by fit_prior.py


def load(smr_path, predictions_path):
    smr = pd.read_parquet(
        smr_path,
        columns=["Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue"],
    )
    tissue = smr.drop_duplicates("Omic").set_index("Omic").Omic_tissue
    organ = {o: organ_of(str(t)) for o, t in tissue.items()}
    units = sorted(organ)
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    by = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    preds = pd.read_parquet(predictions_path)
    preds = preds[preds.p > 0]
    genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    preds = preds[preds.gene.isin(genes)].reset_index(drop=True)
    rows = []
    for r in preds.itertuples():
        g = by[(r.gene, r.disease)]
        o = g[g.Omic != r.heldout]
        s = -np.log10(np.clip(o.p_SMR_multi.values, 1e-300, 1))
        rows.append(dict(
            unit=np.array([units.index(u) for u in o.Omic]),
            bin=np.searchsorted(EDGES, s, side="right"),
            sig=(o.p_SMR_multi.values < THRESHOLD).astype(float),
            same=np.array([organ[u] == organ[r.heldout] for u in o.Omic],
                          float),
        ))
    return preds, rows, len(units)


def unpack(theta, form, n_units):
    a, b = np.exp(theta[:2])
    i = 2
    pi = w = None
    if form in ("G", "O", "GL", "GW"):
        inc = np.exp(theta[i:i + len(EDGES) + 1])
        pi = 1 / (1 + np.exp(-(np.cumsum(inc) - 6)))   # monotone in (0, 1)
        i += len(EDGES) + 1
    if form in ("W", "GW"):
        w = np.exp(theta[i:i + n_units])
    if form == "GL":
        w = 1 / (1 + np.exp(-theta[i]))     # lambda, in (0, 1)
    return a, b, pi, w


def flat(rows):
    """The tests of the rows, flattened, with the row of each."""
    return dict(
        row=np.concatenate([np.full(len(r["sig"]), j)
                            for j, r in enumerate(rows)]),
        **{k: np.concatenate([r[k] for r in rows])
           for k in ("unit", "bin", "sig", "same")},
        n_rows=len(rows),
    )


def predict(theta, form, t, n_units):
    a, b, pi, w = unpack(theta, form, n_units)
    ev = pi[t["bin"]] if pi is not None else t["sig"]
    if form == "O":
        wt = t["same"]
    elif form == "GL":
        wt = t["same"] + w * (1 - t["same"])
    else:
        wt = w[t["unit"]] if w is not None else np.ones(len(ev))
    num = np.bincount(t["row"], wt * ev, minlength=t["n_rows"])
    den = np.bincount(t["row"], wt, minlength=t["n_rows"])
    return (num + a) / (den + a + b)


def logloss(p, y):
    p = np.clip(p, *CLIP)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def start(form, n_units):
    theta = [np.log(0.05), np.log(1.0)]
    if form in ("G", "O", "GL", "GW"):
        theta += [np.log(1.0)] * (len(EDGES) + 1)
    if form == "GL":
        theta += [0.0]
    if form in ("W", "GW"):
        theta += [0.0] * n_units
    return np.array(theta)


def fit(form, rows, y, n_units, prior=None):
    """Fit by log-loss; with ``prior`` = (a, b), only the rest is fitted."""
    fixed = None if prior is None else np.log(prior)

    def full(theta):
        return theta if fixed is None else np.concatenate([fixed, theta])

    def loss(theta):
        theta = full(theta)
        p = np.clip(predict(theta, form, rows, n_units), 1e-6, 1 - 1e-6)
        reg = 0.01 * np.sum(theta[2:] ** 2) if form in ("W", "GW") else 0
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + reg
    x0 = start(form, n_units)
    res = minimize(loss, x0 if fixed is None else x0[2:], method="Powell",
                   options=dict(maxiter=20000, xtol=1e-4, ftol=1e-7))
    return full(res.x)


def cross_validate(form, rows, y, fold, n_units, prior=None):
    p = np.empty(len(y))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        theta = fit(form, flat([rows[i] for i in np.where(tr)[0]]), y[tr],
                    n_units, prior)
        p[te] = predict(theta, form, flat([rows[i] for i in np.where(te)[0]]),
                        n_units)
    return p


def main(smr_path, predictions_path):
    preds, rows, n_units = load(smr_path, predictions_path)
    y = preds.y.values.astype(float)
    genes = np.array(sorted(preds.gene.unique()))
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(genes),
                       np.arange(len(genes)) % FOLDS))
    fold = preds.gene.map(fold_of).values
    print(f"{len(preds)} predictions, {len(genes)} genes, base rate "
          f"{y.mean():.3f}")
    cv = {}
    for form in ORDER:
        p = cross_validate(form, rows, y, fold, n_units)
        pc = np.clip(p, *CLIP)
        cv[form] = logloss(p, y)
        print(f"{form:3} CV log-loss {cv[form]:.4f}  Brier "
              f"{np.mean((pc - y) ** 2):.4f}")
    best = min(cv.values())
    chosen = next(f for f in ORDER if cv[f] <= best + 0.002)
    print("chosen:", chosen)
    theta = fit(chosen, flat(rows), y, n_units)
    a, b, pi, w = unpack(theta, chosen, n_units)
    print(f"fit on all fitting genes: a = {a:.4f}, b = {b:.4f}")
    if pi is not None:
        lo = np.concatenate([[0], EDGES])
        for x, v in zip(lo, pi):
            print(f"  -log10 p >= {x:5.2f}: pi = {v:.4f}")
    if chosen == "GL":
        print(f"  lambda (another organ) = {w:.4f}")
    elif w is not None:
        print("  dataset weights:", np.round(w, 2).tolist())
    if chosen in ("G", "O", "GL"):
        # the engine keeps selected_unit, the prior of support_prior
        p = cross_validate(chosen, rows, y, fold, n_units, ENGINE_PRIOR)
        print(f"{chosen} with the engine's prior a, b = {ENGINE_PRIOR}: CV "
              f"log-loss {logloss(p, y):.4f}  Brier "
              f"{np.mean((np.clip(p, *CLIP) - y) ** 2):.4f}")
        theta = fit(chosen, flat(rows), y, n_units, ENGINE_PRIOR)
        pi = unpack(theta, chosen, n_units)[2]
        print("  pi for engines.yaml:", np.round(pi, 4).tolist())


if __name__ == "__main__":
    main(*sys.argv[1:])
