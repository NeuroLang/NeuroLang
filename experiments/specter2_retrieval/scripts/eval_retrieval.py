import json, sys, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
rng = np.random.default_rng(0)
meta = json.load(open("/tmp/stage4/meta.json"))
E = np.load("/tmp/stage4/emb_light.npz"); H = np.load("/tmp/stage4/emb_heavy.npz")
norm = lambda m: (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)
ids = json.load(open("/tmp/stage3/ids.json")); texts = {k: tuple(v) for k, v in json.load(open("/tmp/stage3/texts.json")).items()}
ids2 = json.load(open("/tmp/stage2/ids.json")); t2 = {k: tuple(v) for k, v in json.load(open("/tmp/stage2/texts.json")).items()}
D1 = norm(np.load("/tmp/nsdata2/ns_syn/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"])
D2 = norm(np.load("/tmp/nsdata/ns_local/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"])
ns, nst = np.array(meta["ns_docs"]), np.array(meta["nst_docs"])
Dns, Dnst = D1[ns], D2[nst]                       # title + abstract, document adapter
Dns_abs = norm(H["ns_abstract_only_d"])            # abstract only, document adapter
corp_ns = [texts[ids[i]][0] + ". " + texts[ids[i]][1] for i in ns]
corp_nst = [t2[ids2[i]][0] + ". " + t2[ids2[i]][1] for i in nst]
def title_self(Q, D, name):
    Q = norm(Q); n = len(D); ranks = np.empty(n, int)
    for s in range(0, n, 1500):
        S = Q[s:s + 1500] @ D.T; true = S[np.arange(len(S)), np.arange(s, s + len(S))]
        ranks[s:s + len(S)] = 1 + (S > true[:, None]).sum(1)
    r = lambda k: 100 * (ranks <= k).mean()
    print("  %-52s R@1 %5.1f  R@5 %5.1f  R@10 %5.1f  R@50 %5.1f  MRR %.3f  median rank %d" % (name, r(1), r(5), r(10), r(50), (1 / ranks).mean(), np.median(ranks)))
    return ranks
print("== E1/E2: title as query, the study as target (%d Neurosynth, %d NeuroStore studies)" % (len(ns), len(nst)))
print("E1  target = title [SEP] abstract embedding (the one query_similarity uses; the title is part of the target text)")
title_self(E["ns_title_q"], Dns, "Neurosynth, query adapter")
title_self(E["ns_title_d"], Dns, "Neurosynth, document adapter used as the query")
title_self(E["nst_title_q"], Dnst, "NeuroStore, query adapter")
print("E2  target = abstract-only embedding (the title is not in the target text)")
title_self(E["ns_title_q"], Dns_abs, "Neurosynth, query adapter")
title_self(E["ns_title_d"], Dns_abs, "Neurosynth, document adapter used as the query")
# lexical baselines for the same task
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32)
X = vec.fit_transform(corp_ns); Xabs = vec.transform([texts[ids[i]][1] for i in ns]); Xt = vec.transform([texts[ids[i]][0] for i in ns])
def lex_title_self(Xq, Xd, name):
    n = Xd.shape[0]; ranks = np.empty(n, int)
    for s in range(0, n, 1500):
        S = (Xq[s:s + 1500] @ Xd.T).toarray(); true = S[np.arange(len(S)), np.arange(s, s + len(S))]
        ranks[s:s + len(S)] = 1 + (S > true[:, None]).sum(1)
    r = lambda k: 100 * (ranks <= k).mean()
    print("  %-52s R@1 %5.1f  R@5 %5.1f  R@10 %5.1f  R@50 %5.1f  MRR %.3f  median rank %d" % (name, r(1), r(5), r(10), r(50), (1 / ranks).mean(), np.median(ranks)))
print("lexical baseline (TF-IDF cosine, unigrams+bigrams)")
lex_title_self(Xt, X, "Neurosynth, target = title + abstract")
lex_title_self(Xt, Xabs, "Neurosynth, target = abstract only")

def topical(Qs, D, rel, X=None, Xq=None):
    """AP, nDCG@10, P@10, R@100 per query; Qs normalised query embeddings (or None for lexical)."""
    out = []
    for qi, rows in enumerate(rel):
        S = (D @ Qs[qi]) if Qs is not None else (X @ Xq[qi].T).toarray().ravel()
        relv = np.zeros(len(S), bool); relv[rows] = True
        order = np.argsort(-S, kind="stable"); hit = relv[order]
        cum = np.cumsum(hit); ap = (cum[hit] / (np.nonzero(hit)[0] + 1)).sum() / hit.sum()
        disc = 1 / np.log2(np.arange(2, 12)); dcg = (hit[:10] * disc).sum(); idcg = disc[:min(10, int(hit.sum()))].sum()
        out.append((ap, dcg / idcg, hit[:10].mean(), hit[:100].sum() / hit.sum(), relv.mean()))
    return np.array(out)
def report(title, key, qkeys, rel, Dm, corp, Xd):
    print(title)
    print("  %-48s %6s %8s %6s %6s %6s" % ("model (%d queries)" % len(rel), "MAP", "nDCG@10", "P@10", "R@100", "base"))
    names = list(qkeys)
    res = {}
    for name, qk in qkeys.items():
        res[name] = topical(norm(E[qk]), Dm, rel)
    qlex = vec_q.transform(key)
    res["TF-IDF lexical (title+abstract)"] = topical(None, None, rel, Xd, qlex)
    rnd = [np.arange(Dm.shape[0])]; 
    for name, m in res.items():
        print("  %-48s %6.3f %8.3f %6.3f %6.3f %6.3f" % (name, *m[:, :4].mean(0), m[:, 4].mean()))
    base = res["TF-IDF lexical (title+abstract)"][:, 0]
    for name, m in res.items():
        if name.startswith("TF-IDF"): continue
        d = m[:, 0] - base; print("    %-46s vs TF-IDF: mean AP diff %+.3f (se %.3f), better on %d%% of queries" % (name, d.mean(), d.std() / np.sqrt(len(d)), 100 * (d > 0).mean()))
    return res
vec_q = vec
mesh_names = list(meta["mesh"]); mesh_rel = [np.array([int(np.where(ns == r)[0][0]) for r in meta["mesh"][n]]) for n in mesh_names]
print("\n== E3: Neurosynth studies, ground truth = human MeSH headings (query = the descriptor name)")
report("targets = title [SEP] abstract embeddings", mesh_names, {"SPECTER2 ad hoc query adapter": "mesh_q", "SPECTER2 document adapter as query": "mesh_d"}, mesh_rel, Dns, corp_ns, X)
con_names = list(meta["concept"]); pos2 = {r: i for i, r in enumerate(nst)}
con_rel = [np.array([pos2[r] for r in meta["concept"][n] if r in pos2]) for n in con_names]
vec2 = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32)
X2 = vec2.fit_transform(corp_nst); vec_q = vec2
print("\n== E4: NeuroStore studies, ground truth = concepts extracted by an LLM from the full text (query = the concept label)")
report("targets = title [SEP] abstract embeddings", con_names, {"SPECTER2 ad hoc query adapter": "concept_q", "SPECTER2 document adapter as query": "concept_d"}, con_rel, Dnst, corp_nst, X2)
dom_names = list(meta["domain"]); dom_rel = [np.array([pos2[r] for r in meta["domain"][n] if r in pos2]) for n in dom_names]
print("\n== E4b: NeuroStore, ground truth = the 13 cognitive domains of the same extraction (query = the domain name)")
report("targets = title [SEP] abstract embeddings", dom_names, {"SPECTER2 ad hoc query adapter": "domain_q", "SPECTER2 document adapter as query": "domain_d"}, dom_rel, Dnst, corp_nst, X2)
