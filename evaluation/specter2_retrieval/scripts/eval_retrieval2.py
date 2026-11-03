import json, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
rng = np.random.default_rng(0)
exec(open("/tmp/eval_retrieval.py").read().split('print("== E1/E2')[0])   # reuse data loading (E, H, D*, corp_*, ns, nst, meta)
def ap_ndcg(S, rows):
    relv = np.zeros(len(S), bool); relv[rows] = True
    order = np.argsort(-S, kind="stable"); hit = relv[order]; cum = np.cumsum(hit)
    ap = (cum[hit] / (np.nonzero(hit)[0] + 1)).sum() / hit.sum()
    disc = 1 / np.log2(np.arange(2, 12)); ndcg = (hit[:10] * disc).sum() / disc[:min(10, int(hit.sum()))].sum()
    return ap, ndcg, hit[:10].mean(), hit[:100].sum() / hit.sum()
def zs(x): return (x - x.mean()) / (x.std() + 1e-12)
def rrf(*Ss, k=60):
    out = 0
    for S in Ss:
        r = np.empty(len(S)); r[np.argsort(-S, kind="stable")] = np.arange(1, len(S) + 1); out = out + 1 / (k + r)
    return out
tok = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2}
def run(title, names, rel, Dm, corp, vecr, Qk):
    X = vecr.fit_transform(corp); Q = norm(E[Qk]); Xq = vecr.transform(names)
    doc_tok = [tok(t) for t in corp]
    res = {m: [] for m in ("SPECTER2", "TF-IDF", "fusion z-score", "fusion RRF")}; dis_rec, dis_frac, dis_base, lex_rec = [], [], [], []
    for qi, rows in enumerate(rel):
        s_sp = Dm @ Q[qi]; s_lx = (X @ Xq[qi].T).toarray().ravel()
        for m, S in (("SPECTER2", s_sp), ("TF-IDF", s_lx), ("fusion z-score", zs(s_sp) + zs(s_lx)), ("fusion RRF", rrf(s_sp, s_lx))):
            res[m].append(ap_ndcg(S, rows))
        qt = tok(names[qi]); disjoint = np.array([r for r in rows if not (qt & doc_tok[r])])
        dis_frac.append(len(disjoint) / len(rows))
        if len(disjoint) >= 10:
            top = np.argsort(-s_sp, kind="stable")[:100]; dis_rec.append(np.isin(disjoint, top).mean())
            toplx = np.argsort(-s_lx, kind="stable")[:100]; lex_rec.append(np.isin(disjoint, toplx).mean())
            dis_base.append(100 / len(s_sp))
    print(title); print("  %-22s %6s %8s %6s %6s" % ("(%d queries)" % len(rel), "MAP", "nDCG@10", "P@10", "R@100"))
    for m, v in res.items(): print("  %-22s %6.3f %8.3f %6.3f %6.3f" % ((m,) + tuple(np.array(v).mean(0))))
    print("  positives sharing no content word with the query: %.0f%% on average; on the %d queries with >= 10 of them, share of those retrieved in the top 100: SPECTER2 %.1f%%, TF-IDF %.1f%%, chance %.1f%%" % (100 * np.mean(dis_frac), len(dis_rec), 100 * np.mean(dis_rec), 100 * np.mean(lex_rec), 100 * np.mean(dis_base)))
mesh_names = list(meta["mesh"]); mesh_rel = [np.array([int(np.where(ns == r)[0][0]) for r in meta["mesh"][n]]) for n in mesh_names]
mk = lambda: TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32)
print("== E3 Neurosynth / MeSH"); run("fusion and lexical-free positives", mesh_names, mesh_rel, Dns, corp_ns, mk(), "mesh_q")
pos2 = {r: i for i, r in enumerate(nst)}; con_names = list(meta["concept"]); con_rel = [np.array([pos2[r] for r in meta["concept"][n] if r in pos2]) for n in con_names]
print("\n== E4 NeuroStore / LLM concepts"); run("fusion and lexical-free positives", con_names, con_rel, Dnst, corp_nst, mk(), "concept_q")
# E5: related-article retrieval, validated by shared MeSH headings
mesh = json.load(open("/tmp/stage3/mesh.json")); CHECK = {"Humans","Male","Female","Adult","Adolescent","Aged","Middle Aged","Young Adult","Child","Animals","Aged, 80 and over","Infant","Child, Preschool","Rats","Mice"}
ms = [set(n for n, u, m in mesh.get(ids[i], []) if n not in CHECK) for i in ns]
eligible = [j for j, s in enumerate(ms) if len(s) >= 3]; sample = rng.choice(eligible, 3000, replace=False)
vec = mk(); Xn = vec.fit_transform(corp_ns); 
def neigh(S_row, j): S_row = S_row.copy(); S_row[j] = -1e9; return np.argpartition(-S_row, 10)[:10]
def score_nb(nb, j):
    g = np.array([len(ms[j] & ms[k]) for k in nb]); jac = np.array([len(ms[j] & ms[k]) / max(1, len(ms[j] | ms[k])) for k in nb]); return jac.mean(), g
res = {"SPECTER2 (title+abstract docs)": [], "SPECTER2 abstract-only docs": [], "TF-IDF": [], "random": []}
ideal = []
for j in sample:
    for name, S in (("SPECTER2 (title+abstract docs)", Dns @ Dns[j]), ("SPECTER2 abstract-only docs", Dns_abs @ Dns_abs[j]), ("TF-IDF", (Xn @ Xn[j].T).toarray().ravel()), ("random", rng.random(len(ns)))):
        nb = np.argsort(-np.where(np.arange(len(S)) == j, -1e9, S), kind="stable")[:10]
        jac, g = score_nb(nb, j); disc = 1 / np.log2(np.arange(2, 12))
        # nDCG with graded relevance = number of shared MeSH headings; ideal from a sample of 3000 candidates
        res[name].append((jac, g.mean(), (g[:10] * disc[:10]).sum()))
print("\n== E5 related studies: the 10 nearest studies of 3000 Neurosynth studies (each with >= 3 MeSH headings), judged by shared MeSH headings")
print("  %-34s %14s %20s" % ("neighbour definition", "MeSH Jaccard", "shared headings (mean)"))
for n, v in res.items(): v = np.array(v); print("  %-34s %14.3f %20.2f" % (n, v[:, 0].mean(), v[:, 1].mean()))
