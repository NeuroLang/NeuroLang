import json, re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
rng = np.random.default_rng(1); REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
P = json.load(open("/tmp/exp/paraphrases.json")); terms = list(P); K = len(terms)
ids = json.load(open("/tmp/stage3/ids.json")); texts = {k: tuple(v) for k, v in json.load(open("/tmp/stage3/texts.json")).items()}
row_of = {int(k): i for i, k in enumerate(ids)}; n_all = len(ids)
norm = lambda m: (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)
D = norm(np.load("/tmp/nsdata2/ns_syn/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"])
A = sp.load_npz("/tmp/exp/act.npz").tocsr()
tf = pd.read_parquet("/tmp/exp/tfidf.parquet"); tf["row"] = tf.s.map(row_of)
G = {t: np.unique(tf[(tf.term == t) & (tf.tfidf > 0.001)].row.values) for t in terms}
Q = np.load("/tmp/exp/query_emb.npz", allow_pickle=True); Qemb = norm(Q["emb"]); qtexts = list(Q["texts"]); qidx = {t: i for i, t in enumerate(qtexts)}
corpus = [texts[k][0] + ". " + texts[k][1] for k in ids]
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32); X = vec.fit_transform(corpus)
def umap(rows): p = np.asarray(A[rows].sum(0)).ravel() / len(rows); return p / p.mean()
zs = lambda x: (x - x.mean()) / (x.std() + 1e-12)
def prep(M, mb): M = M - mb; M = M - M.mean(1, keepdims=True); return M / np.linalg.norm(M, axis=1, keepdims=True)
S_sp = {t: [(D @ Qemb[qidx[q]]) for q in P[t]] for t in terms}; S_lx = {t: [(X @ vec.transform([q]).T).toarray().ravel() for q in P[t]] for t in terms}
S_sp_lab = {t: D @ Qemb[qidx[t]] for t in terms}; S_lx_lab = {t: (X @ vec.transform([t]).T).toarray().ravel() for t in terms}
ARMS = ["SPECTER2", "TF-IDF", "fusion (z-score)", "random"]; LAB = ["SPECTER2, term as query", "TF-IDF, term as query"]
acc = {(c, a): [] for c in (0, 1) for a in ARMS + LAB + ["ceiling"]}   # per rep: arrays over paraphrases (or terms)
rk = {(c, a): [] for c in (0, 1) for a in ARMS + LAB + ["ceiling"]}; prec = {a: [] for a in ARMS}; disc = {(c, a): [] for c in (0, 1) for a in ARMS}
true_idx = np.array([ti for ti, t in enumerate(terms) for _ in P[t]])
for rep in range(REPS):
    ref = rng.random(n_all) < 0.5; qrows = np.where(~ref)[0]
    Gr = {t: G[t][ref[G[t]]] for t in terms}; Gq = {t: G[t][~ref[G[t]]] for t in terms}
    Rmat = np.stack([umap(Gr[t]) for t in terms]); mb = Rmat.mean(0); Rn = {c: prep(Rmat, mb * c) for c in (0, 1)}
    def ident(Um, ti, c):
        C = prep(np.atleast_2d(Um), mb * c) @ Rn[c].T; order = np.argsort(-C, axis=1)
        rank = np.array([int(np.where(order[i] == ti[i])[0][0]) + 1 for i in range(len(C))])
        d = np.array([C[i, ti[i]] - np.delete(C[i], ti[i]).mean() for i in range(len(C))]); return rank, d
    maps = {a: [] for a in ARMS}; pr = {a: [] for a in ARMS}; labmaps = {a: [] for a in LAB}; ceilmaps = []
    for ti, t in enumerate(terms):
        N = len(Gq[t]); ceilmaps.append(umap(Gq[t]))
        for pi in range(3):
            ssp, slx = S_sp[t][pi][qrows], S_lx[t][pi][qrows]
            sels = {"SPECTER2": ssp, "TF-IDF": slx, "fusion (z-score)": zs(ssp) + zs(slx)}
            for a, s in sels.items():
                sel = qrows[np.argpartition(-s, N)[:N]]; maps[a].append(umap(sel)); pr[a].append(np.isin(sel, Gq[t]).mean())
            sel = rng.choice(qrows, N, replace=False); maps["random"].append(umap(sel)); pr["random"].append(N / len(qrows))
        for a, s in (("SPECTER2, term as query", S_sp_lab[t]), ("TF-IDF, term as query", S_lx_lab[t])):
            labmaps[a].append(umap(qrows[np.argpartition(-s[qrows], N)[:N]]))
    for c in (0, 1):
        for a in ARMS: r, d = ident(np.stack(maps[a]), true_idx, c); rk[(c, a)].append(r); disc[(c, a)].append(d)
        for a in LAB: r, _ = ident(np.stack(labmaps[a]), np.arange(K), c); rk[(c, a)].append(r)
        r, _ = ident(np.stack(ceilmaps), np.arange(K), c); rk[(c, "ceiling")].append(r)
    for a in ARMS: prec[a].append(pr[a])
    print("split", rep + 1, "/", REPS, "done; acc@1 raw SPECTER2 %.1f TF-IDF %.1f" % (100 * (rk[(0, "SPECTER2")][-1] == 1).mean(), 100 * (rk[(0, "TF-IDF")][-1] == 1).mean()), flush=True)
def boot(vals, groups, n=2000):
    ug = np.unique(groups); out = []
    for _ in range(n):
        s = rng.choice(ug, len(ug)); out.append(np.mean(np.concatenate([vals[groups == g] for g in s])))
    return np.percentile(out, [2.5, 97.5])
cmap = {0: "raw correlation", 1: "centered correlation (common network removed)"}
for c in (0, 1):
    print("\n== identification of the term among %d, %s; chance acc@1 = %.1f%%; %d random splits of the literature (reference maps from one half, selections from the other)" % (K, cmap[c], 100 / K, REPS))
    print("  %-26s %7s %7s %7s | %-17s | %s" % ("selection", "acc@1", "acc@3", "acc@5", "acc@1 95% CI", "precision | discrimination"))
    for a in ARMS:
        R = np.stack(rk[(c, a)]); a1 = (R == 1).mean(0); ci = boot(a1, true_idx)
        print("  %-26s %6.1f%% %6.1f%% %6.1f%% | [%4.1f, %4.1f]%%      | %.2f | %+.3f" % (a, 100 * a1.mean(), 100 * (R <= 3).mean(), 100 * (R <= 5).mean(), 100 * ci[0], 100 * ci[1], np.mean(prec[a]), np.mean(disc[(c, a)])))
    for a in LAB + ["ceiling"]:
        R = np.stack(rk[(c, a)]); print("  %-26s %6.1f%% %6.1f%% %6.1f%% | %s" % (a, 100 * (R == 1).mean(), 100 * (R <= 3).mean(), 100 * (R <= 5).mean(), "ceiling: gold studies of the query half" if a == "ceiling" else "(36 term labels, context)"))
    d = (np.stack(rk[(c, "SPECTER2")]) == 1).mean(0).astype(float) - (np.stack(rk[(c, "TF-IDF")]) == 1).mean(0); ci = boot(d, true_idx)
    print("  SPECTER2 - TF-IDF acc@1: %+.1f points, 95%% CI [%+.1f, %+.1f] (bootstrap over terms)" % (100 * d.mean(), 100 * ci[0], 100 * ci[1]))
    d = (np.stack(rk[(c, "SPECTER2")]) == 1).mean(0).astype(float) - (np.stack(rk[(c, "fusion (z-score)")]) == 1).mean(0); ci = boot(d, true_idx)
    print("  SPECTER2 - fusion acc@1: %+.1f points, 95%% CI [%+.1f, %+.1f]" % (100 * d.mean(), 100 * ci[0], 100 * ci[1]))
rows = []
for ti, t in enumerate(terms):
    for pi, q in enumerate(P[t]):
        i = ti * 3 + pi
        rows.append((t, q, len(G[t]), *[float(np.mean([r[i] for r in rk[(0, a)]])) for a in ("SPECTER2", "TF-IDF", "fusion (z-score)")], *[float(np.mean([p[i] for p in prec[a]])) for a in ("SPECTER2", "TF-IDF")]))
df = pd.DataFrame(rows, columns=["term", "paraphrase", "n_studies", "mean_rank_specter2", "mean_rank_tfidf", "mean_rank_fusion", "precision_specter2", "precision_tfidf"]); df.to_csv("/tmp/exp/per_paraphrase_splits.csv", index=False)
pt = df.groupby("term").mean(numeric_only=True).sort_values("mean_rank_specter2"); print("\nper term (mean rank of the right term over 3 paraphrases and %d splits; precision of the selection):" % REPS); print(pt.round(2).to_string())
