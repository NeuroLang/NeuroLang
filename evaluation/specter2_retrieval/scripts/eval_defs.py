import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
exec(open("/tmp/eval_retrieval2.py").read().split("mesh_names = list(meta")[0])   # data loading + helpers
Ed = np.load("/tmp/stage4/emb_defs.npz"); defs = json.load(open("/tmp/stage4/concept_defs.json"))
names = list(defs)
# merge NeuroStore concepts that differ only by case
pos2 = {r: i for i, r in enumerate(nst)}; groups = {}
for n in names:
    groups.setdefault(n.strip().lower(), set()).update(pos2[r] for r in meta["concept"][n] if r in pos2)
keys = list(groups); idx = {n.strip().lower(): i for i, n in enumerate(names)}
rel = [np.array(sorted(groups[k])) for k in keys]
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32); X = vec.fit_transform(corp_nst)
first = {k: next(n for n in names if n.strip().lower() == k) for k in keys}
lab_txt = list(first); def_txt = [defs[first[k]] for k in keys]
Ql = norm(np.stack([E["concept_q"][list(meta["concept"]).index(first[k])] for k in keys]))
Qd = norm(np.stack([Ed["def_q"][idx[k]] for k in keys])); Qdd = norm(np.stack([Ed["def_d"][idx[k]] for k in keys]))
Xl, Xd = vec.transform(lab_txt), vec.transform(def_txt)
def evalq(name, getS):
    v = np.array([ap_ndcg(getS(i), r) for i, r in enumerate(rel)]); print("  %-52s %6.3f %8.3f %6.3f %6.3f" % ((name,) + tuple(v.mean(0)))); return v
print("== E6 NeuroStore concepts with a Cognitive Atlas definition: %d concepts, median %d studies" % (len(keys), int(np.median([len(r) for r in rel]))))
print("  %-52s %6s %8s %6s %6s" % ("query", "MAP", "nDCG@10", "P@10", "R@100"))
a = evalq("SPECTER2 ad hoc adapter, label", lambda i: Dnst @ Ql[i])
b = evalq("SPECTER2 ad hoc adapter, Cognitive Atlas definition", lambda i: Dnst @ Qd[i])
c = evalq("SPECTER2 document adapter, definition", lambda i: Dnst @ Qdd[i])
d = evalq("TF-IDF lexical, label", lambda i: (X @ Xl[i].T).toarray().ravel())
e = evalq("TF-IDF lexical, definition", lambda i: (X @ Xd[i].T).toarray().ravel())
f = evalq("fusion RRF, SPECTER2 definition + TF-IDF label", lambda i: rrf(Dnst @ Qd[i], (X @ Xl[i].T).toarray().ravel()))
g = evalq("fusion RRF, SPECTER2 label + TF-IDF label", lambda i: rrf(Dnst @ Ql[i], (X @ Xl[i].T).toarray().ravel()))
print("  definition vs label with SPECTER2: mean AP diff %+.3f (se %.3f), definition better on %d%% of concepts" % ((b[:, 0] - a[:, 0]).mean(), (b[:, 0] - a[:, 0]).std() / np.sqrt(len(a)), 100 * (b[:, 0] > a[:, 0]).mean()))

print("\n== 'working memory' in detail (NeuroStore studies annotated with the concept by the LLM extraction)")
k = "working memory"; i = keys.index(k); r = rel[i]; N = len(Dnst)
print("  annotated studies:", len(r), "of", N)
for name, S in (("SPECTER2 definition (the example's query)", Dnst @ Qd[i]), ("SPECTER2 label", Dnst @ Ql[i]), ("TF-IDF label", (X @ Xl[i].T).toarray().ravel()), ("TF-IDF definition", (X @ Xd[i].T).toarray().ravel())):
    ap, nd, p10, r100 = ap_ndcg(S, r); top = np.argsort(-S, kind="stable")[:1794]; hit = np.isin(top, r).sum()
    print("  %-44s AP %.3f  P@10 %.2f  | top 1,794 (5%%): %4d annotated = precision %.0f%%, recall %.0f%%" % (name, ap, p10, hit, 100 * hit / 1794, 100 * hit / len(r)))
