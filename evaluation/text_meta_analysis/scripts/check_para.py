import json, re, pandas as pd
try:
    from nltk.stem import PorterStemmer; st = PorterStemmer().stem
except Exception:
    def st(w):
        for suf in ("ations","ation","ings","ing","ions","ion","ies","ied","ed","es","ly","s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3: return w[:-len(suf)]
        return w
P = json.load(open("/tmp/exp/paraphrases.json")); tf = pd.read_parquet("/tmp/exp/tfidf.parquet")
vocab = set(tf.term.unique()); words = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bad = []; cov = 0; n = 0
STOP = {"the","and","of","in","to","a","an","with","for","on","at","by","from","as","or","is","are","one","two","that","when","what","who","how","than","over","into","while","their","its","it","be","this"}
for t, qs in P.items():
    ts = {st(w) for w in words(t)}
    for q in qs:
        qs_ = {st(w) for w in words(q)}; n += 1
        if ts & qs_: bad.append((t, q, ts & qs_))
        toks = [w for w in words(q) if w not in STOP]; grams = set(toks) | {" ".join(g) for g in zip(toks, toks[1:])}
        other = {g for g in grams if g in vocab and g != t and not (set(words(g)) & set(words(t)))}
        cov += bool(other)
print(len(P), "terms,", n, "paraphrases;", "violations:", len(bad))
for b in bad: print("  ", b)
print("paraphrases containing another vocabulary term (synonym-mediated candidates): %d / %d" % (cov, n))
