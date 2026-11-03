import json, re, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
rng = np.random.default_rng(0)
P = json.load(open("/tmp/exp/paraphrases.json")); terms = list(P)
ids = json.load(open("/tmp/stage3/ids.json")); texts = {k: tuple(v) for k, v in json.load(open("/tmp/stage3/texts.json")).items()}
row_of = {int(k): i for i, k in enumerate(ids)}
norm = lambda m: (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)
D = norm(np.load("/tmp/nsdata2/ns_syn/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"])
A = sp.load_npz("/tmp/exp/act.npz").tocsr(); NV = A.shape[1]
tf = pd.read_parquet("/tmp/exp/tfidf.parquet"); tf["row"] = tf.s.map(row_of)
G = {t: np.unique(tf[(tf.term == t) & (tf.tfidf > 0.001)].row.values) for t in terms}
Q = np.load("/tmp/exp/query_emb.npz", allow_pickle=True); Qemb = norm(Q["emb"]); qtexts = list(Q["texts"])
corpus = [texts[k][0] + ". " + texts[k][1] for k in ids]
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32); X = vec.fit_transform(corpus)
def umap(rows):
    p = np.asarray(A[rows].sum(0)).ravel() / len(rows); return p / p.mean()
def zs(x): return (x - x.mean()) / (x.std() + 1e-12)
def topn(S, n): return np.argpartition(-S, n)[:n]
K = len(terms); Rmat = np.stack([umap(G[t]) for t in terms]); Mbar = Rmat.mean(0)
def prep(M, center): M = M - (Mbar if center else 0); M = M - M.mean(1, keepdims=True); return M / np.linalg.norm(M, axis=1, keepdims=True)
Rn = {c: prep(Rmat, c) for c in (False, True)}
def identify(Um, true_idx, center):
    C = prep(np.atleast_2d(Um), center) @ Rn[center].T; order = np.argsort(-C, axis=1)
    rank = np.array([int(np.where(order[i] == true_idx[i])[0][0]) + 1 for i in range(len(C))])
    disc = np.array([C[i, true_idx[i]] - np.delete(C[i], true_idx[i]).mean() for i in range(len(C))]); return rank, disc
# the arms
rows_meta = []; arms = {"SPECTER2": [], "TF-IDF": [], "fusion (z-score)": [], "random": []}; prec = {a: [] for a in arms}; true_idx = []
for ti, t in enumerate(terms):
    N = len(G[t])
    for q in P[t]:
        qi = qtexts.index(q); s_sp = D @ Qemb[qi]; qv = vec.transform([q]); s_lx = (X @ qv.T).toarray().ravel()
        nz = qv.nnz
        sels = {"SPECTER2": topn(s_sp, N), "TF-IDF": topn(s_lx, N), "fusion (z-score)": topn(zs(s_sp) + zs(s_lx), N)}
        for a, sel in sels.items(): arms[a].append(umap(sel)); prec[a].append(np.isin(sel, G[t]).mean())
        rmap = np.mean([umap(rng.choice(len(ids), N, replace=False)) for _ in range(5)], axis=0); arms["random"].append(rmap); prec["random"].append(N / len(ids))
        rows_meta.append((t, q, N, nz)); true_idx.append(ti)
true_idx = np.array(true_idx)
print("terms %d, paraphrases %d, studies per term median %d (range %d-%d); %d paraphrases have no TF-IDF vocabulary hit" % (K, len(rows_meta), np.median([len(G[t]) for t in terms]), min(map(len, G.values())), max(map(len, G.values())), sum(m[3] == 0 for m in rows_meta)))
# reference confusability and split-half ceiling
Cr = Rn[False] @ Rn[False].T; np.fill_diagonal(Cr, -1); print("most similar reference maps:", ", ".join("%s~%s %.3f" % (terms[i], terms[j], Cr[i, j]) for i, j in sorted(zip(*np.triu_indices(K, 1)), key=lambda ij: -Cr[ij])[:5]))
ceil = {False: [], True: []}
for rep in range(10):
    A_, B_ = {}, {}
    for t in terms:
        p = rng.permutation(G[t]); A_[t], B_[t] = p[:len(p) // 2], p[len(p) // 2:]
    RB = np.stack([umap(B_[t]) for t in terms]); mb = RB.mean(0)
    for c in (False, True):
        Rb = RB - (mb if c else 0); Rb = Rb - Rb.mean(1, keepdims=True); Rb /= np.linalg.norm(Rb, axis=1, keepdims=True)
        Ua = np.stack([umap(A_[t]) for t in terms]) - (mb if c else 0); Ua = Ua - Ua.mean(1, keepdims=True); Ua /= np.linalg.norm(Ua, axis=1, keepdims=True)
        ceil[c].append((np.argmax(Ua @ Rb.T, 1) == np.arange(K)).mean())
# term labels (a keyword query, for context)
lab = {"SPECTER2, term as query": [], "TF-IDF, term as query": []}
for ti, t in enumerate(terms):
    N = len(G[t]); qi = qtexts.index(t); s_sp = D @ Qemb[qi]; s_lx = (X @ vec.transform([t]).T).toarray().ravel()
    lab["SPECTER2, term as query"].append(umap(topn(s_sp, N))); lab["TF-IDF, term as query"].append(umap(topn(s_lx, N)))
# synonym-mediated covariate: another vocabulary term with Jaccard >= 0.2 with the target's studies appears in the paraphrase
vocab = set(tf.term.unique()); words = lambda s: re.findall(r"[a-z0-9]+", s.lower()); Gall = {}
def Gset(v):
    if v not in Gall: Gall[v] = set(tf[(tf.term == v) & (tf.tfidf > 0.001)].row.values)
    return Gall[v]
syn = []
for (t, q, N, nz) in rows_meta:
    toks = words(q); grams = set(toks) | {" ".join(g) for g in zip(toks, toks[1:])}; gt = set(G[t]); best = 0
    for g in grams:
        if g in vocab and not (set(words(g)) & set(words(t))):
            o = Gset(g); best = max(best, len(o & gt) / len(o | gt))
    syn.append(best >= 0.2)
syn = np.array(syn); print("paraphrases containing a vocabulary term that overlaps the target term's studies (Jaccard >= 0.2): %d / %d" % (syn.sum(), len(syn)))
def boot(vals, groups, n=2000):
    ug = np.unique(groups); out = []
    for _ in range(n):
        s = rng.choice(ug, len(ug)); out.append(np.mean(np.concatenate([vals[groups == g] for g in s])))
    return np.percentile(out, [2.5, 97.5])
res = {}
for center in (False, True):
    print("\n== identification of the term among %d (%s correlation%s); chance acc@1 = %.1f%%" % (K, "centered " if center else "raw ", ", common network removed" if center else "", 100 / K))
    print("  split-half ceiling (map of half the term's studies vs the other halves): %.1f%% (sd %.1f)" % (100 * np.mean(ceil[center]), 100 * np.std(ceil[center])))
    print("  %-26s %7s %7s %7s | %-17s | %s" % ("selection from the paraphrase", "acc@1", "acc@3", "acc@5", "acc@1 95% CI", "selection precision | discrimination r"))
    for a, maps in arms.items():
        rank, disc = identify(np.stack(maps), true_idx, center); res[(center, a)] = (rank, disc)
        ci = boot((rank == 1).astype(float), true_idx)
        print("  %-26s %6.1f%% %6.1f%% %6.1f%% | [%4.1f, %4.1f]%%      | %.2f | %+.3f" % (a, 100 * (rank == 1).mean(), 100 * (rank <= 3).mean(), 100 * (rank <= 5).mean(), 100 * ci[0], 100 * ci[1], np.mean(prec[a]), disc.mean()))
    for a, maps in lab.items():
        rank, disc = identify(np.stack(maps), np.arange(K), center)
        print("  %-26s %6.1f%% %6.1f%% %6.1f%% | (36 term labels, for context)" % (a, 100 * (rank == 1).mean(), 100 * (rank <= 3).mean(), 100 * (rank <= 5).mean()))
    d = (res[(center, "SPECTER2")][0] == 1).astype(float) - (res[(center, "TF-IDF")][0] == 1).astype(float); ci = boot(d, true_idx)
    print("  SPECTER2 - TF-IDF acc@1: %+.1f points, 95%% CI [%+.1f, %+.1f] (bootstrap over terms)" % (100 * d.mean(), 100 * ci[0], 100 * ci[1]))
    for name, mask in (("synonym-mediated paraphrases", syn), ("paraphrases without such a term", ~syn)):
        if mask.sum(): print("  %-32s n=%3d  acc@1 SPECTER2 %.1f%%  TF-IDF %.1f%%  fusion %.1f%%" % (name, mask.sum(), *[100 * (res[(center, a)][0][mask] == 1).mean() for a in ("SPECTER2", "TF-IDF", "fusion (z-score)")]))
rank_s, _ = res[(False, "SPECTER2")]; rank_t, _ = res[(False, "TF-IDF")]; rank_f, _ = res[(False, "fusion (z-score)")]
df = pd.DataFrame(rows_meta, columns=["term", "paraphrase", "n_studies", "tfidf_vocab_hits"]); df["syn_mediated"] = syn
for nme, r, a in (("rank_specter2", rank_s, "SPECTER2"), ("rank_tfidf", rank_t, "TF-IDF"), ("rank_fusion", rank_f, "fusion (z-score)")): df[nme] = r
for a, k in (("SPECTER2", "prec_specter2"), ("TF-IDF", "prec_tfidf"), ("fusion (z-score)", "prec_fusion")): df[k] = prec[a]
df.to_csv("/tmp/exp/per_paraphrase.csv", index=False)
pt = df.groupby("term")[["rank_specter2", "rank_tfidf", "rank_fusion", "prec_specter2", "prec_tfidf"]].agg(lambda x: x.median() if x.name.startswith("rank") else x.mean())
pt["n"] = [len(G[t]) for t in pt.index]; print("\nper term (median rank of the right term over its 3 paraphrases; selection precision):")
print(pt.sort_values("rank_specter2").round(2).to_string())
