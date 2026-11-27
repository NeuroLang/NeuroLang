"""Part 1 of PROTOCOL_LATENT.md: fit (rho, alpha_brain, alpha_blood,
alpha_other, beta) per outer fold by log-loss on the training-fold
single-held-out items; out-of-fold single-held-out predictions.

    uv run --no-sync python latent_fit.py

Writes results/latent_params.json and results/latent_oof.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import leak_checks as lc
import prototype_weighted as pw
from same_evidence_models import folds_of
from neurolang_apps.utils.engines.biomedsql.init import organ_of

HERE = Path(__file__).resolve().parent
SMR = (HERE / "../../../neurolang_data/probe/goldq/biomedsql/db/"
       "NeurodegenerativeDiseases_SMR_Genes_Full.parquet")
PRED = (HERE / "../../../scratch/probe/results/biomedsql_calibration/"
        "predictions.parquet")
GROUPS = ["brain", "blood", "other"]
FOLDS = pw.FOLDS


def group(organ):
    return organ if organ in ("brain", "blood") else "other"


def unpack(t):
    """t = (t_rho, t_beta, t_brain, t_blood, t_other) -> rho, beta, alphas."""
    sg = lambda x: 1 / (1 + np.exp(-x))
    rho, beta = sg(t[0]), sg(t[1])
    alpha = {g: beta + (1 - beta) * sg(t[2 + i]) for i, g in enumerate(GROUPS)}
    return rho, beta, alpha


def pack(rho, beta, alpha):
    lg = lambda x: np.log(x / (1 - x))
    return np.array([lg(rho), lg(beta)] + [
        lg((alpha[g] - beta) / (1 - beta)) for g in GROUPS])


def posterior_real(rho, beta, alpha, ev):
    """P(real | evidence); ev = list of (group, significant)."""
    lr, ln = np.log(rho), np.log(1 - rho)
    for g, s in ev:
        lr += np.log(alpha[g] if s else 1 - alpha[g])
        ln += np.log(beta if s else 1 - beta)
    m = max(lr, ln)
    return np.exp(lr - m) / (np.exp(lr - m) + np.exp(ln - m))


def counts(ev_list):
    """(n_items, 3 groups) significant and non-significant counts."""
    s = np.zeros((len(ev_list), 3))
    ns = np.zeros((len(ev_list), 3))
    for i, ev in enumerate(ev_list):
        for g, sg_ in ev:
            (s if sg_ else ns)[i, GROUPS.index(g)] += 1
    return s, ns


def predict_q1(t, s, ns, hgroup):
    """P(held-out significant | e), vectorised."""
    rho, beta, alpha = unpack(t)
    a = np.array([alpha[g] for g in GROUPS])
    lr = np.log(rho) + (s * np.log(a) + ns * np.log(1 - a)).sum(1)
    ln = (np.log(1 - rho) + s.sum(1) * np.log(beta)
          + ns.sum(1) * np.log(1 - beta))
    post = 1 / (1 + np.exp(ln - lr))
    return post * a[hgroup] + (1 - post) * beta


def fit(s, ns, hgroup, y):
    def loss(t):
        p = np.clip(predict_q1(t, s, ns, hgroup), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    best = None
    rng = np.random.default_rng(0)
    starts = [pack(0.2, 0.05, dict(brain=.5, blood=.5, other=.5))]
    starts += [rng.normal(0, 1.5, 5) for _ in range(7)]
    for x0 in starts:
        r = minimize(loss, x0, method="L-BFGS-B", bounds=[(-12, 12)] * 5)
        r = minimize(loss, r.x, method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def load_items():
    smr = pd.read_parquet(SMR, columns=[
        "Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue",
        "ProbeChr", "Probe_bp"])
    fit_genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    eval_genes = {i["gene"] for i in json.load(open(HERE / "items_eval.json"))}
    allp = pd.read_parquet(PRED)
    allp = allp[allp.p > 0]
    allp = allp[allp.gene.isin(fit_genes | eval_genes)].reset_index(drop=True)
    return smr, allp


def gene_folds(smr, allp, extra_genes=()):
    """Fold of every gene: the 65 loci of the 3,960 items with the folds of
    same_evidence_models; a gene outside them gets the fold of the single
    one of the 65 loci within 1 Mb (same chromosome) of it, else None."""
    pos = lc.loci_of(smr, set(allp.gene))
    loci = pos.locus.values
    f_of_locus = {}
    ploc = allp.gene.map(pos.locus).values
    fold = folds_of(ploc, FOLDS)
    for l, f in zip(ploc, fold):
        f_of_locus[l] = int(f)
    out = {g: (int(pos.loc[g, "locus"]), f_of_locus[pos.loc[g, "locus"]])
           for g in pos.index}
    unmapped = {}
    if extra_genes:
        p2 = lc.loci_of(smr, set(extra_genes) - set(pos.index))
        for g in p2.index:
            c, b = p2.loc[g, "chr"], p2.loc[g, "bp"]
            near = pos[(pos.chr == c) & ((pos.bp - b).abs() <= lc.LOCUS_BP)]
            ls = set(near.locus)
            if len(ls) == 1:
                l = ls.pop()
                out[g] = (int(l), f_of_locus[l])
            else:
                unmapped[g] = sorted(int(x) for x in ls)
    return out, unmapped, ploc, fold


def main():
    smr, allp = load_items()
    organ = {o: organ_of(str(t)) for o, t in
             smr.drop_duplicates("Omic").set_index("Omic").Omic_tissue.items()}
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True
                        ).p_SMR_multi.min().reset_index()
    by = {k: g for k, g in tests.groupby(["Gene", "Disease"])}
    ev_list, hg = [], []
    for r in allp.itertuples():
        g = by[(r.gene, r.disease)]
        o = g[g.Omic != r.heldout]
        ev_list.append([(group(organ[u]), bool(p < pw.THRESHOLD))
                        for u, p in zip(o.Omic, o.p_SMR_multi)])
        hg.append(GROUPS.index(group(organ[r.heldout])))
    s, ns = counts(ev_list)
    hg = np.array(hg)
    y = allp.y.values.astype(float)
    assert len(y) == 3960
    _, _, ploc, fold = gene_folds(smr, allp)
    oof = np.full(len(y), np.nan)
    params = {}
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        t, ll = fit(s[tr], ns[tr], hg[tr], y[tr])
        rho, beta, alpha = unpack(t)
        oof[te] = predict_q1(t, s[te], ns[te], hg[te])
        params[str(f)] = dict(rho=rho, beta=beta, alpha=alpha, theta=t.tolist(),
                              train_logloss=ll, n_train=int(tr.sum()))
        print(f"fold {f}: rho {rho:.4f} alpha_brain {alpha['brain']:.4f} "
              f"alpha_blood {alpha['blood']:.4f} alpha_other "
              f"{alpha['other']:.4f} beta {beta:.4f}  train log-loss {ll:.4f}")
    assert not np.isnan(oof).any()
    ids = [f"{r.gene}|{r.disease}|{r.heldout}" for r in allp.itertuples()]
    json.dump(params, open(HERE / "results/latent_params.json", "w"), indent=1)
    json.dump({i: dict(y=float(y[j]), fold=int(fold[j]), locus=int(ploc[j]),
                       latent=float(oof[j])) for j, i in enumerate(ids)},
              open(HERE / "results/latent_oof.json", "w"))
    print("Brier (clipped) OOF:", lc.brier(oof, y).mean())


if __name__ == "__main__":
    main()
