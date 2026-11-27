import json, random, collections
import numpy as np
from hpo_enriched import *
ch = json.load(open("results/hpo_enriched_chosen.json")); E = Enriched(ch["a"]); eps = ch["eps"]
base_cfg = json.load(open("results/hpo_chosen.json"))["exact"]
P = patients(list(enumerate(real)))
rows = []
for pid, d, si, di_, obs, pm in P:
    i = W.di.get(d)
    if i is None: rows.append(dict(pid=pid, disease=d, si=si, di=di_, enr=None, base=None, other=0)); continue
    other = sum(1 for p, _ in bydis[d] if p not in pm)
    rows.append(dict(pid=pid, disease=d, si=si, di=di_, other=other,
                     enr=rank_of(posterior_enriched(E, obs, eps, d, pm), i), base=rank_of(posterior(W, obs, base_cfg), i)))
json.dump(rows, open("results/hpo_enriched_test.json", "w"))
rk = lambda x: np.inf if x is None else float(x)
def met(rs, k): r = np.array([rk(x[k]) for x in rs]); return 100*(r==1).mean(), 100*(r<=10).mean(), np.mean(1/r)
def boot(rs, f, B=1000):
    by = collections.defaultdict(list)
    for x in rs: by[x["disease"]].append(x)
    ks = sorted(by); rng = random.Random(0); v = []
    for _ in range(B): v.append(f([x for k in rng.choices(ks, k=len(ks)) for x in by[k]]))
    v.sort(); return v[int(.025*B)], v[int(.975*B)]
print("params", ch["a"], ch["eps"], "| test patients", len(rows), "| diseases", len({r['disease'] for r in rows}))
for label, sel in [("all cells", lambda r: True), ("clean", lambda r: r["si"] == 0 and r["di"] == 0), ("2 spurious", lambda r: r["si"] == 2 and r["di"] == 0), ("50% dropped", lambda r: r["si"] == 0 and r["di"] == 2)]:
    rs = [r for r in rows if sel(r)]
    for k in ("base", "enr"):
        t1, t10, m = met(rs, k); lo, hi = boot(rs, lambda z, k=k: met(z, k)[2], 300)
        print(f"{label:12} {k:5} top1 {t1:5.1f} top10 {t10:5.1f} MRR {m:.3f} [{lo:.3f},{hi:.3f}] (n={len(rs)})")
d = lambda z: met(z, "enr")[2] - met(z, "base")[2]
lo, hi = boot(rows, d); print(f"paired MRR enriched - HPOA-only, all: {d(rows):+.3f} [{lo:+.3f},{hi:+.3f}]")
# how much training evidence does the true disease have?
oth = collections.Counter(min(r["other"], 10) for r in rows[::12]); print("other-publication cases available for the true disease (base cases):", sorted(oth.items()))
no = [r for r in rows if r["other"] == 0]; ye = [r for r in rows if r["other"] > 0]
for lab, rs in (("no other-PMID case (falls back to HPOA)", no), (">=1 other-PMID case", ye)):
    if rs: print(f"  {lab}: n={len(rs)//12} base cases; base MRR {met(rs,'base')[2]:.3f}, enriched MRR {met(rs,'enr')[2]:.3f}")
# the 60 Phase 2 patients vs Sonnet direct (strict)
p2 = [json.loads(l) for l in open("results/hpo_phase2.jsonl")]
idx = {(r["pid"], r["si"], r["di"]): r for r in rows}
sub = []
for r in p2:
    si, di_ = SPUR_LEVELS.index(r["spur"]), DROP_LEVELS.index(r["drop"]); e = idx[(r["pid"], si, di_)]
    sub.append(dict(disease=r["disease"], enr=e["enr"], base=e["base"], son=r["direct"].get("rank_strict"), son_len=r["direct"].get("rank_lenient")))
print("\n60 Phase-2 patients:")
for k in ("base", "enr", "son"):
    t1, t10, m = met(sub, k); lo, hi = boot(sub, lambda z, k=k: met(z, k)[2], 1000); print(f"  {k:5} top1 {t1:5.1f} top10 {t10:5.1f} MRR {m:.3f} [{lo:.3f},{hi:.3f}]")
for a, b in (("enr", "son"), ("enr", "base")):
    f = lambda z, a=a, b=b: met(z, a)[2] - met(z, b)[2]; lo, hi = boot(sub, f, 2000); print(f"  paired MRR {a} - {b}: {f(sub):+.3f} [{lo:+.3f},{hi:+.3f}]")
