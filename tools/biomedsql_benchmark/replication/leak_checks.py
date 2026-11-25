"""Two leakage checks on the pre-registered evaluation.

    python leak_checks.py SMR.parquet PREDICTIONS.parquet

Check 1: locus-grouped split (genes within 1 Mb, same chromosome, are one
locus; position = median ProbeChr / Probe_bp of the gene's rows).
Check 2: dataset-nested fit (parameters refitted without the held-out
dataset's predictions on the fitting genes).
Predictors are the closed forms: share k/n, support_prior (k+a)/(n+a+b),
support_graded (sum pi(bin) + a)/(n + a + b), fitted with prototype_weighted.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import prototype_weighted as pw
import post_hoc_logit as pl
from fit_prior import GRID_A, GRID_B

HERE = Path(__file__).resolve().parent
CLIP, B, LOCUS_BP = pw.CLIP, 2000, 1_000_000


def brier(p, y):
    return (np.clip(p, *CLIP) - y) ** 2


def build_rows(smr, preds, units):
    """Per-prediction evidence of the other datasets (as pw.load builds it)."""
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    by = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    idx = {u: i for i, u in enumerate(units)}
    rows = []
    for r in preds.itertuples():
        g = by[(r.gene, r.disease)]
        o = g[g.Omic != r.heldout]
        s = -np.log10(np.clip(o.p_SMR_multi.values, 1e-300, 1))
        rows.append(dict(
            unit=np.array([idx[u] for u in o.Omic]),
            bin=np.searchsorted(pw.EDGES, s, side="right"),
            sig=(o.p_SMR_multi.values < pw.THRESHOLD).astype(float),
            same=np.zeros(len(o)),
        ))
    return rows, by


def kn(rows):
    return (np.array([r["sig"].sum() for r in rows]),
            np.array([len(r["sig"]) for r in rows], float))


def fit_prior_grid(k, n, y):
    best = min((pw.logloss((k + a) / (n + a + b), y), a, b)
               for a in GRID_A for b in GRID_B)
    return best[1], best[2]


def fit_graded(rows, y, n_units):
    return pw.fit("G", pw.flat(rows), y, n_units, pw.ENGINE_PRIOR)


def predict_graded(theta, rows, n_units):
    return pw.predict(theta, "G", pw.flat(rows), n_units)


def cluster_ci(diff_terms, clusters):
    """Mean of per-item terms and 95% CI resampling clusters."""
    labels, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, diff_terms, minlength=len(labels))
    c = np.bincount(inv, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(labels), (B, len(labels)))
    bs = s[draws].sum(1) / c[draws].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return diff_terms.mean(), lo, hi


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def report(name, preds_by_system, y, clusters, pairs):
    print(f"  n = {len(y)}, clusters = {len(set(clusters))}")
    for s, p in preds_by_system.items():
        print(f"  {s:16} Brier {brier(p, y).mean():.4f}  CI "
              f"[{cluster_ci(brier(p, y), clusters)[1]:.4f}, "
              f"{cluster_ci(brier(p, y), clusters)[2]:.4f}]  "
              f"log-loss {pw.logloss(p, y):.4f}")
    for a, b in pairs:
        d = brier(preds_by_system[a], y) - brier(preds_by_system[b], y)
        print(f"  Brier {a} - {b}: {fmt(cluster_ci(d, clusters))}")


def loci_of(smr, genes):
    pos = smr[smr.Gene.isin(genes)].groupby("Gene").agg(
        chr=("ProbeChr", lambda x: x.mode().iloc[0]), bp=("Probe_bp", "median"))
    pos = pos.reset_index().sort_values(["chr", "bp"]).reset_index(drop=True)
    new = (pos.chr != pos.chr.shift()) | (pos.bp - pos.bp.shift() > LOCUS_BP)
    pos["locus"] = new.cumsum()
    return pos.set_index("Gene")


def main(smr_path, predictions_path):
    smr = pd.read_parquet(smr_path, columns=[
        "Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue", "Source",
        "ProbeChr", "Probe_bp"])
    fit_genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    eval_genes = {i["gene"] for i in json.load(open(HERE / "items_eval.json"))}
    allp = pd.read_parquet(predictions_path)
    allp = allp[allp.p > 0]
    allp = allp[allp.gene.isin(fit_genes | eval_genes)].reset_index(drop=True)
    units = sorted(smr.Omic.unique())
    nu = len(units)
    rows, by = build_rows(smr, allp, units)
    y = allp.y.values.astype(float)
    is_fit = allp.gene.isin(fit_genes).values
    is_ev = allp.gene.isin(eval_genes).values & ~is_fit
    k, n = kn(rows)
    take = lambda m: [rows[i] for i in np.where(m)[0]]

    # consistency with prototype_weighted.load on the fitting genes
    _, prow, _ = pw.load(smr_path, predictions_path)
    assert len(prow) == is_fit.sum() and all(
        np.array_equal(a["bin"], b["bin"]) for a, b in zip(prow, take(is_fit)))
    print(f"{is_fit.sum()} fitting predictions ({len(fit_genes)} genes), "
          f"{is_ev.sum()} evaluation predictions "
          f"({allp[is_ev].gene.nunique()} genes)")

    # shipped parameters, fitted on the fitting genes
    a0, b0 = fit_prior_grid(k[is_fit], n[is_fit], y[is_fit])
    theta0 = fit_graded(take(is_fit), y[is_fit], nu)
    pi0 = pw.unpack(theta0, "G", nu)[2]
    print(f"shipped prior a={a0}, b={b0}; pi = {np.round(pi0, 4).tolist()}")
    ev = allp[is_ev].reset_index(drop=True)
    ye, ke, ne = y[is_ev], k[is_ev], n[is_ev]
    sysp = {
        "share": ke / ne,
        "support_prior": (ke + a0) / (ne + a0 + b0),
        "support_graded": predict_graded(theta0, take(is_ev), nu),
    }
    ids = [f"{r.gene}|{r.disease}|{r.heldout}" for r in ev.itertuples()]
    ans = {}
    for line in open(HERE / "results/eval_answers.jsonl"):
        r = json.loads(line)
        ans[r["id"]] = r
    sysp["L2"] = np.array([ans[i]["L2"] for i in ids], float)

    print("\n== Sanity: full evaluation, closed forms (published prior 0.147,"
          " graded 0.143) ==")
    for s in ("share", "support_prior", "support_graded", "L2"):
        print(f"  {s:16} Brier {brier(sysp[s], ye).mean():.4f}")
    bp, bg = (brier(sysp[s], ye).mean() for s in
              ("support_prior", "support_graded"))
    ok = round(bp, 3) == 0.147 and round(bg, 3) == 0.143
    print("  reproduces published:", ok)
    if not ok:
        return
    ids2, ye2, plog, *_ = pl.fit_logistic(smr_path, predictions_path)
    assert ids2 == ids and np.array_equal(ye2, ye)
    sysp["logistic"] = plog
    print(f"  logistic         Brier {brier(plog, ye).mean():.4f}")
    pairs = [("support_graded", "support_prior"), ("support_graded", "L2"),
             ("logistic", "L2")]
    genes_ev = ev.gene.values
    print("\n== Full evaluation, CI resampling genes (published: graded-prior"
          " -0.0036 [-0.0058,-0.0016]; graded-L2 +0.014 [+0.005,+0.024]) ==")
    report("full", sysp, ye, genes_ev, pairs)

    # ---------------- Check 1 ----------------
    print("\n== Check 1: locus-grouped ==")
    print("position columns: ProbeChr (mode per gene), Probe_bp (median per "
          f"gene); locus = single linkage within {LOCUS_BP} bp over the "
          f"{len(fit_genes | eval_genes)} genes of items.json + items_eval.json")
    pos = loci_of(smr, fit_genes | eval_genes)
    assert len(pos) == len(fit_genes | eval_genes)
    locus = pos.locus
    fit_loci = set(locus[list(fit_genes)])
    ev_genes = sorted(eval_genes)
    shared = [g for g in ev_genes if locus[g] in fit_loci]
    print(f"{len(pos)} genes in {locus.nunique()} loci; "
          f"{len(shared)} of {len(ev_genes)} evaluation genes share a locus "
          f"with a fitting gene, covering "
          f"{int(ev.gene.isin(shared).sum())} of {len(ev)} evaluation "
          f"predictions")
    ev_locus = ev.gene.map(locus).values
    cont = ev.gene.isin(shared).values
    for label, m in (("clean (locus has no fitting gene)", ~cont),
                     ("contaminated complement", cont)):
        print(f"\n-- {label} --")
        report(label, {s: p[m] for s, p in sysp.items()}, ye[m],
               ev_locus[m], pairs)

    print("\n-- fully locus-grouped re-split, 5 folds over loci (seed 0), "
          "pooled 353 genes --")
    pos_all = loci_of(smr, set(allp.gene))
    ploc = allp.gene.map(pos_all.locus).values
    loci = np.array(sorted(set(ploc)))
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(loci), np.arange(len(loci)) % pw.FOLDS))
    fold = np.array([fold_of[l] for l in ploc])
    info = smr.drop_duplicates("Omic").set_index("Omic")
    organ = {o: pw.organ_of(str(t)) for o, t in info.Omic_tissue.items()}
    source = info.Source.to_dict()
    X = np.array([pl.features((g := by[(r.gene, r.disease)])[g.Omic != r.heldout],
                              r.heldout, organ, source)
                  for r in allp.itertuples()], float)
    from sklearn.linear_model import LogisticRegression
    oof = {s: np.empty(len(y)) for s in
           ("share", "support_prior", "support_graded", "logistic")}
    for f in range(pw.FOLDS):
        tr, te = fold != f, fold == f
        a, b = fit_prior_grid(k[tr], n[tr], y[tr])
        th = fit_graded(take(tr), y[tr], nu)
        oof["share"][te] = k[te] / n[te]
        oof["support_prior"][te] = (k[te] + a) / (n[te] + a + b)
        oof["support_graded"][te] = predict_graded(th, take(te), nu)
        oof["logistic"][te] = LogisticRegression(C=1.0, max_iter=2000).fit(
            X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"  fold {f}: {te.sum()} predictions, prior a={a}, b={b}")
    report("oof", oof, y, ploc, [("support_graded", "support_prior"),
                                 ("logistic", "support_graded")])

    # ---------------- Check 2 ----------------
    print("\n== Check 2: dataset-nested fit ==")
    nested = {"support_prior": np.empty(len(ye)),
              "support_graded": np.empty(len(ye))}
    fitm = is_fit
    for q in sorted(set(ev.heldout)):
        trq = fitm & (allp.heldout.values != q)
        a, b = fit_prior_grid(k[trq], n[trq], y[trq])
        th = fit_graded(take(trq), y[trq], nu)
        m = (ev.heldout.values == q)
        nested["support_prior"][m] = (ke[m] + a) / (ne[m] + a + b)
        nested["support_graded"][m] = predict_graded(
            th, [r for r, mm in zip(take(is_ev), m) if mm], nu)
    print(f"{len(set(ev.heldout))} held-out datasets in the evaluation")
    for s in nested:
        print(f"  {s:16} shipped Brier {brier(sysp[s], ye).mean():.4f} "
              f"log-loss {pw.logloss(sysp[s], ye):.4f} | nested Brier "
              f"{brier(nested[s], ye).mean():.4f} log-loss "
              f"{pw.logloss(nested[s], ye):.4f} | max |change| "
              f"{np.abs(nested[s] - sysp[s]).max():.5f}")
    nsys = {"support_prior": nested["support_prior"],
            "support_graded": nested["support_graded"], "L2": sysp["L2"]}
    report("nested", nsys, ye, genes_ev,
           [("support_graded", "support_prior"), ("support_graded", "L2")])


if __name__ == "__main__":
    main(*sys.argv[1:])
