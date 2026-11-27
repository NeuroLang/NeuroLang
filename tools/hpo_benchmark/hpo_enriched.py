"""Does more evidence help the symbolic model? HPOA + phenopacket counts, leave-publication-out.

p(d,t) = (n_dt + a * p_hpoa(d,t)) / (N_d + a) for diseases with phenopackets (N_d cases, n_dt
of them showing t). For the held-out case's own disease, every case from the same PMID is
removed from N_d and n_dt (fallback to HPOA if none remain). Parameters a, eps chosen on dev.
"""
import collections, glob, json, random, re, sys
import numpy as np
from hpo_bench import *
from hpo_cases import PPS, documented_pmids
from hpo_names import disease_names

W = World()
cases = load_cases(W.replaced)
sp = json.load(open("results/hpo_split.json")); dev, test = set(sp["dev"]), set(sp["test"])
refs = documented_pmids()

# all phenopackets: (disease, pmid, observed terms)
allpp = []
for fn in sorted(glob.glob(str(PPS / "**/*.json"), recursive=True)):
    d = json.load(open(fn)); dis = d["diseases"][0]["term"]["id"]
    pm = "PMID:" + re.search(r"PMID_(\d+)", fn).group(1)
    obs = sorted({W.replaced.get(p["type"]["id"], p["type"]["id"]) for p in (d.get("phenotypicFeatures") or []) if not p.get("excluded")})
    if obs and dis in W.di:
        allpp.append((dis, pm, obs))
bydis = collections.defaultdict(list)
for dis, pm, obs in allpp: bydis[dis].append((pm, obs))
print("phenopackets usable", len(allpp), "| diseases with phenopackets among candidates", len(bydis), flush=True)

# case -> pmid(s): non-leaked pp with the same disease and terms
pmids_of = collections.defaultdict(set)
for dis, pm, obs in allpp:
    if pm not in refs[dis]: pmids_of[(dis, tuple(obs))].add(pm)

Nd = np.zeros(W.n); cnt = collections.defaultdict(lambda: np.zeros(W.n))
for dis, lst in bydis.items():
    i = W.di[dis]; Nd[i] = len(lst)
    for pm, obs in lst:
        for t in obs: cnt[t][i] += 1

class Enriched:
    def __init__(self, a): self.a, self.cache = a, {}
    def vec(self, t):
        v = self.cache.get(t)
        if v is None:
            h = W.vec(t, 0, 0, 1.0)
            c = cnt.get(t)
            c = c if c is not None else 0.0
            v = np.where(Nd > 0, (c + self.a * h) / (Nd + self.a), h)
            if len(self.cache) > 50000: self.cache.clear()
            self.cache[t] = v
        return v
    def true_override(self, dis, t, drop_pm):
        """Leave-publication-out p(dis, t)."""
        lst = [o for pm, o in bydis[dis] if pm not in drop_pm]
        h = W.by_dis[dis].get(t, 0.0)
        if not lst: return h
        return (sum(t in o for o in lst) + self.a * h) / (len(lst) + self.a)

def posterior_enriched(E, obs, eps, dis, drop_pm):
    M = 1 - (1 - np.stack([E.vec(t) for t in obs])) * (1 - eps)
    i = W.di[dis]
    M[:, i] = [1 - (1 - E.true_override(dis, t, drop_pm)) * (1 - eps) for t in obs]
    with np.errstate(divide="ignore"): ll = np.log(M).sum(axis=0)
    if np.all(np.isinf(ll)): return np.zeros(W.n)
    w = np.exp(ll - ll.max()); return w / w.sum()

def patients(case_list, pid_offset=0):
    out = []
    for pid, (d, base) in case_list:
        for si in range(len(SPUR_LEVELS)):
            for di_ in range(len(DROP_LEVELS)):
                out.append((pid, d, si, di_, perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(f"real/{pid}/{si}/{di_}")),
                            pmids_of.get((d, tuple(base)), set())))
    return out

real = [c for c in cases if c[0] in test]
indexed_all = list(enumerate(cases))
dev_cases = [(k, c) for k, c in indexed_all if c[0] in dev]
# NB: Phase 1 / Phase 2 patient ids index into `real` (test cases only); dev uses its own seeds (as in hpo_dev.py)
def dev_patients():
    out = []
    for ci, (d, base) in enumerate([c for c in cases if c[0] in dev]):
        for si in range(4):
            for di_ in range(3):
                out.append((ci, d, si, di_, perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(f"0/{ci}/{si}/{di_}")),
                            pmids_of.get((d, tuple(base)), set())))
    return out

def mrr(E, eps, P):
    rr = []
    for _, d, _, _, obs, pm in P:
        if d not in W.di: rr.append(0.0); continue
        r = rank_of(posterior_enriched(E, obs, eps, d, pm), W.di[d]); rr.append(1 / r if r else 0.0)
    return float(np.mean(rr))

if __name__ == "__main__":
    DP = dev_patients()
    res = []
    for a in (0.5, 2.0, 8.0, 32.0):
        for eps in (0.005, 0.02):
            m = mrr(Enriched(a), eps, DP); res.append((m, a, eps)); print(f"dev MRR {m:.4f} a={a} eps={eps}", flush=True)
    best = max(res); print("CHOSEN", best, flush=True)
    json.dump(dict(mrr=best[0], a=best[1], eps=best[2], grid=res), open("results/hpo_enriched_chosen.json", "w"))
