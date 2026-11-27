"""Preliminary, brute force (numpy), NOT the benchmark: does ancestor matching change ranking?"""
import collections, numpy as np
from hpo_lib import load as load_facts
from hpo_onto import load as load_onto, closure
from hpo_cases import load_cases
facts, _ = load_facts(); parents, obsolete, replaced = load_onto(); A = closure(parents)
diseases = sorted({d for d, _ in facts}); di = {d: i for i, d in enumerate(diseases)}
by_pheno = collections.defaultdict(list)
for (d, s), p in facts.items(): by_pheno[s].append((di[d], p))
desc = collections.defaultdict(dict)                       # t -> {descendant: dist}
for t, anc in A.items():
    for a, dist in anc.items(): desc[a][t] = dist
GENERIC = {"HP:0000001", "HP:0000118"}
EPS, GAMMA = 0.02, 0.5
def vec(t, k_up, k_down):
    v = np.zeros(len(diseases))
    for i, p in by_pheno.get(t, ()): v[i] = max(v[i], p)
    for a, dist in A.get(t, {}).items():
        if dist <= k_up and a not in GENERIC:
            for i, p in by_pheno.get(a, ()): v[i] = max(v[i], GAMMA * p)
    if t not in GENERIC:
        for s, dist in desc.get(t, {}).items():
            if dist <= k_down:
                for i, p in by_pheno.get(s, ()): v[i] = max(v[i], GAMMA * p)
    return v
cases = load_cases(replaced)
def run(k_up, k_down, leak):
    cache, ranks, empty, nocand = {}, [], 0, []
    for d, obs in cases:
        ll = np.zeros(len(diseases))
        for t in obs:
            if t not in cache: cache[t] = vec(t, k_up, k_down)
            p = cache[t]
            if leak: p = 1 - (1 - p) * (1 - EPS)
            with np.errstate(divide="ignore"): ll += np.log(p)
        if d not in di: nocand.append(d); ranks.append(10**9); continue
        if np.all(np.isinf(ll)): empty += 1; ranks.append(10**9); continue
        ranks.append(1 + int((ll > ll[di[d]]).sum()))
    r = np.array(ranks, float)
    return f"top1 {100*(r==1).mean():5.1f}%  top10 {100*(r<=10).mean():5.1f}%  top100 {100*(r<=100).mean():5.1f}%  MRR {np.mean(1/r):.3f}  empty {100*empty/len(r):4.1f}%  true-disease-not-a-candidate {len(nocand)}"
print("cases", len(cases), "| chance top-1 = %.3f%%, top-10 = %.2f%%" % (100/len(diseases), 1000/len(diseases)))
for label, ku, kd, lk in [("exact, plain", 0, 0, False), ("exact, leak", 0, 0, True),
                          ("ancestors up<=2/down<=2, leak", 2, 2, True), ("ancestors up<=99/down<=99, leak", 99, 99, True)]:
    print(f"{label:34} {run(ku, kd, lk)}")
