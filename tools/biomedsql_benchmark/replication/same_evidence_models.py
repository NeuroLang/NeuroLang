"""Parts 1 and 3 of PROTOCOL_SAME.md: fitted ceilings (F1-F3) and exact lifted
engine forms (E1-E3, E*), all out of fold (5 outer locus-grouped folds).

    python same_evidence_models.py SMR.parquet PREDICTIONS.parquet
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import prototype_weighted as pw
import post_hoc_logit as pl
import leak_checks as lc
from analyse import auc, CLIP
from neurolang_apps.utils.engines.biomedsql.init import organ_of

HERE = Path(__file__).resolve().parent
M_GRID = [1, 2, 5, 10, 20, 50]
INNER = 4


# ---------- generic helpers ----------
def take(rows, idx):
    return [rows[i] for i in idx]


def folds_of(loci_arr, n_folds, seed=0):
    loci = np.array(sorted(set(loci_arr)))
    rng = np.random.default_rng(seed)
    f_of = dict(zip(rng.permutation(loci), np.arange(len(loci)) % n_folds))
    return np.array([f_of[l] for l in loci_arr])


# ---------- E1 ----------
def fit_e1(k, n, y):
    return lc.fit_prior_grid(k, n, y)


def pred_e1(ab, k, n):
    return (k + ab[0]) / (n + ab[0] + ab[1])


# ---------- E2 ----------
def fit_e2(rows, y, nu):
    return lc.fit_graded(rows, y, nu)


def pred_e2(theta, rows, nu):
    return lc.predict_graded(theta, rows, nu)


# ---------- E3 noisy-OR ----------
def e3_unpack(theta):
    eps, tau = 1 / (1 + np.exp(-theta[:2]))
    inc = np.exp(theta[2:2 + len(pw.EDGES) + 1])
    pi = 1 / (1 + np.exp(-(np.cumsum(inc) - 6)))
    return eps, tau, pi


def e3_predict(theta, t):
    eps, tau, pi = e3_unpack(theta)
    ls = np.bincount(t["row"], np.log1p(-tau * pi[t["bin"]]),
                     minlength=t["n_rows"])
    return 1 - (1 - eps) * np.exp(ls)


def fit_e3(rows, y):
    t = pw.flat(rows)

    def loss(theta):
        p = np.clip(e3_predict(theta, t), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    x0 = np.array([np.log(0.05 / 0.95), 0.0] + [np.log(1.0)] * (len(pw.EDGES) + 1))
    return minimize(loss, x0, method="Powell",
                    options=dict(maxiter=20000, xtol=1e-4, ftol=1e-7)).x


def pred_e3(theta, rows):
    return e3_predict(theta, pw.flat(rows))


# ---------- F1 ----------
def f1_fit(k, n, y, m, ab):
    """Per (k, n) cell, the observed rate shrunk towards the fitted prior's
    value for that cell, (k + a) / (n + a + b), with weight m (PROTOCOL_SAME:
    "beta-binomial shrinkage towards the fitted prior")."""
    cells = {}
    for kk, nn, yy in zip(k, n, y):
        s, t = cells.get((kk, nn), (0.0, 0))
        cells[(kk, nn)] = (s + yy, t + 1)
    out = {}
    for (kk, nn), (s, t) in cells.items():
        prior = float(pred_e1(ab, np.array([kk]), np.array([nn]))[0])
        out[(kk, nn)] = (s + m * prior) / (t + m)
    return out


def f1_predict(table, ab, k, n):
    fb = pred_e1(ab, k, n)
    return np.array([table.get((kk, nn), f) for kk, nn, f in zip(k, n, fb)])


def fit_f1(k, n, y, inner_fold):
    best = None
    for m in M_GRID:
        p = np.empty(len(y))
        for f in range(INNER):
            tr, te = inner_fold != f, inner_fold == f
            ab = fit_e1(k[tr], n[tr], y[tr])
            p[te] = f1_predict(f1_fit(k[tr], n[tr], y[tr], m, ab), ab, k[te], n[te])
        ll = pw.logloss(p, y)
        if best is None or ll < best[0]:
            best = (ll, m)
    m = best[1]
    ab = fit_e1(k, n, y)
    return f1_fit(k, n, y, m, ab), ab, m


# ---------- main ----------
def main(smr_path, predictions_path):
    smr = pd.read_parquet(smr_path, columns=[
        "Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue", "Omic_type",
        "Source", "ProbeChr", "Probe_bp"])
    fit_genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    eval_genes = {i["gene"] for i in json.load(open(HERE / "items_eval.json"))}
    allp = pd.read_parquet(predictions_path)
    allp = allp[allp.p > 0]
    allp = allp[allp.gene.isin(fit_genes | eval_genes)].reset_index(drop=True)
    units = sorted(smr.Omic.unique())
    nu = len(units)
    rows, by = lc.build_rows(smr, allp, units)
    y = allp.y.values.astype(float)
    k, n = lc.kn(rows)
    ids = [f"{r.gene}|{r.disease}|{r.heldout}" for r in allp.itertuples()]

    pos = lc.loci_of(smr, set(allp.gene))
    ploc = allp.gene.map(pos.locus).values
    fold = folds_of(ploc, pw.FOLDS)

    # features
    info = smr.drop_duplicates("Omic").set_index("Omic")
    organ = {o: organ_of(str(t)) for o, t in info.Omic_tissue.items()}
    source = info.Source.to_dict()
    kind = info.Omic_type.astype(str).to_dict()
    X2 = np.array([pl.features((g := by[(r.gene, r.disease)])[g.Omic != r.heldout],
                               r.heldout, organ, source)
                   for r in allp.itertuples()], float)
    uidx = {u: i for i, u in enumerate(units)}
    organs = sorted(set(organ.values()))
    kinds = sorted(set(kind.values()))
    diseases = sorted(allp.disease.unique())
    X3 = np.full((len(allp), nu + 6), np.nan)
    for j, r in enumerate(allp.itertuples()):
        g = by[(r.gene, r.disease)]
        s = -np.log10(np.clip(g.p_SMR_multi.values, 1e-300, 1))
        s = np.minimum(s, 300)
        for u, v in zip(g.Omic, s):
            X3[j, uidx[u]] = v
        X3[j, uidx[r.heldout]] = np.nan
        X3[j, nu:] = [uidx[r.heldout], organs.index(organ[r.heldout]),
                      kinds.index(kind[r.heldout]), k[j], n[j],
                      diseases.index(r.disease)]
    cat_mask = np.array([False] * nu + [True, True, True, False, False, True])

    # ---------- sanity ----------
    print("== Sanity checks ==")
    print(f"items {len(allp)}; genes {allp.gene.nunique()}; loci {pos.locus.nunique()}")
    assert len(allp) == 3960 and allp.gene.nunique() == 353 and pos.locus.nunique() == 65
    assert len(set(ids)) == len(ids)

    names = ["F1", "F2", "F3", "E1", "E2", "E3", "Estar"]
    oof = {s: np.full(len(y), np.nan) for s in names}
    chosen = {}
    for f in range(pw.FOLDS):
        tr_i, te_i = np.where(fold != f)[0], np.where(fold == f)[0]
        ytr = y[tr_i]
        inner = folds_of(ploc[tr_i], INNER)
        # F1
        table, ab_f1, m = fit_f1(k[tr_i], n[tr_i], ytr, inner)
        oof["F1"][te_i] = f1_predict(table, ab_f1, k[te_i], n[te_i])
        # F2
        oof["F2"][te_i] = LogisticRegression(C=1.0, max_iter=2000).fit(
            X2[tr_i], ytr).predict_proba(X2[te_i])[:, 1]
        # F3
        # a column all-missing in training is constant: dropped (sklearn's
        # binning fails on it); it carries no information
        keep = np.isfinite(X3[tr_i]).any(0) | cat_mask
        gb = HistGradientBoostingClassifier(
            max_iter=200, random_state=0, categorical_features=cat_mask[keep])
        oof["F3"][te_i] = gb.fit(X3[tr_i][:, keep], ytr).predict_proba(
            X3[te_i][:, keep])[:, 1]
        if not keep.all():
            print(f"  F3: dropped all-missing column(s) {np.where(~keep)[0].tolist()}")
        # E1-E3
        ab = fit_e1(k[tr_i], n[tr_i], ytr)
        oof["E1"][te_i] = pred_e1(ab, k[te_i], n[te_i])
        th2 = fit_e2(take(rows, tr_i), ytr, nu)
        oof["E2"][te_i] = pred_e2(th2, take(rows, te_i), nu)
        th3 = fit_e3(take(rows, tr_i), ytr)
        oof["E3"][te_i] = pred_e3(th3, take(rows, te_i))
        # E*: inner CV selection
        cv = {}
        for form in ("E1", "E2", "E3"):
            p = np.empty(len(tr_i))
            for g in range(INNER):
                a_, b_ = np.where(inner != g)[0], np.where(inner == g)[0]
                ia, ib = tr_i[a_], tr_i[b_]
                if form == "E1":
                    p[b_] = pred_e1(fit_e1(k[ia], n[ia], y[ia]), k[ib], n[ib])
                elif form == "E2":
                    p[b_] = pred_e2(fit_e2(take(rows, ia), y[ia], nu),
                                    take(rows, ib), nu)
                else:
                    p[b_] = pred_e3(fit_e3(take(rows, ia), y[ia]), take(rows, ib))
            cv[form] = pw.logloss(p, ytr)
        ch = min(cv, key=cv.get)
        chosen[f] = ch
        oof["Estar"][te_i] = oof[ch][te_i]
        print(f"fold {f}: train {len(tr_i)}, test {len(te_i)}, F1 m={m}, "
              f"inner CV log-loss {({a: round(b, 4) for a, b in cv.items()})} -> {ch}",
              flush=True)

    for s in names:
        assert not np.isnan(oof[s]).any()
        assert oof[s].min() >= 0 and oof[s].max() <= 1
    assert np.bincount(fold, minlength=5).sum() == len(y)
    print("every item predicted exactly once (one test fold each); all "
          "probabilities in [0, 1]")

    out = {i: dict(y=float(y[j]), fold=int(fold[j]),
                   **{s: float(oof[s][j]) for s in names})
           for j, i in enumerate(ids)}
    out["_chosen"] = {str(f): c for f, c in chosen.items()}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results/same_models_oof.json", "w"))

    # ---------- report ----------
    lines = []
    lines.append(f"{len(y)} items, 353 genes, 65 loci, out of fold (5 locus-grouped folds)")
    lines.append(f"{'model':7} {'Brier':>7}  95% CI (loci)        {'log-loss':>8} {'AUC':>7}")
    for s in names:
        b = lc.brier(oof[s], y)
        _, lo, hi = lc.cluster_ci(b, ploc)
        lines.append(f"{s:7} {b.mean():.4f}  [{lo:.4f}, {hi:.4f}]  "
                     f"{pw.logloss(oof[s], y):8.4f} {auc(oof[s], y):7.4f}")
    lines.append("")
    lines.append("Paired Brier differences (first minus second), 95% CI resampling loci:")
    for a, b in (("Estar", "F2"), ("Estar", "E1"), ("F3", "F2"), ("E3", "E2")):
        d = lc.brier(oof[a], y) - lc.brier(oof[b], y)
        lines.append(f"  {a} - {b}: {lc.fmt(lc.cluster_ci(d, ploc))}")
    lines.append("")
    lines.append("Chosen form per outer fold: " +
                 ", ".join(f"fold {f}: {c}" for f, c in chosen.items()))
    txt = "\n".join(lines)
    print("\n" + txt)
    open(HERE / "results/same_models.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
