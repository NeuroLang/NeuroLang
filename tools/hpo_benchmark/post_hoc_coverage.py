"""Post hoc: does Sonnet's advantage over the posterior depend on how much of the patient's evidence the annotation can use?

Coverage = share of the patient's observed terms (before perturbation) that the true disease's HPOA annotation lists.
It depends on patient and disease only, not on either system's output.
"""
import json, random
import numpy as np
from hpo_bench import *

W = World(); cases = load_cases(W.replaced)
test = set(json.load(open("results/hpo_split.json"))["test"])
real = [c for c in cases if c[0] in test]
rows = []
for r in json.load(open("results/hybrid_results.json")):
    rows.append(dict(pid=r["pid"], post=r["posterior_rank"], son=r["H0"]))
for l in open("results/hpo_phase2.jsonl"):
    r = json.loads(l)
    rows.append(dict(pid=r["pid"], post=r["phase1"]["exact_leak"], son=r["direct"].get("rank_strict")))
for r in rows:
    truth, base = real[r["pid"]]
    r["cov"] = np.mean([t in W.by_dis[truth] for t in base])
    r["n_annot"] = len(W.by_dis[truth])
rk = lambda x: 0.0 if x is None else 1.0 / x
cov = np.array([r["cov"] for r in rows]); post = np.array([rk(r["post"]) for r in rows]); son = np.array([rk(r["son"]) for r in rows])
print("patients", len(rows), "| mean coverage", cov.mean().round(2), "| overall MRR posterior", post.mean().round(3), "Sonnet direct", son.mean().round(3))
qs = np.quantile(cov, [1/3, 2/3])
rng = np.random.default_rng(0)
for lab, m in (("low coverage", cov <= qs[0]), ("middle", (cov > qs[0]) & (cov <= qs[1])), ("high coverage", cov > qs[1])):
    d = son[m] - post[m]
    bs = np.sort([rng.choice(d, len(d)).mean() for _ in range(5000)])
    print(f"{lab:14} n={m.sum():3d} coverage {cov[m].min():.2f}-{cov[m].max():.2f} | posterior {post[m].mean():.3f}  Sonnet {son[m].mean():.3f} | Sonnet - posterior {d.mean():+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}]")
na = np.array([r["n_annot"] for r in rows]); qa = np.quantile(na, [1/3, 2/3])
print("by the number of annotated phenotypes of the true disease:")
for lab, m in (("few", na <= qa[0]), ("middle", (na > qa[0]) & (na <= qa[1])), ("many", na > qa[1])):
    d = son[m] - post[m]
    bs = np.sort([rng.choice(d, len(d)).mean() for _ in range(5000)])
    print(f"{lab:14} n={m.sum():3d} annotations {na[m].min()}-{na[m].max()} | posterior {post[m].mean():.3f}  Sonnet {son[m].mean():.3f} | Sonnet - posterior {d.mean():+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}]")
print("correlation of coverage with posterior MRR %.2f, with Sonnet MRR %.2f" % (np.corrcoef(cov, post)[0, 1], np.corrcoef(cov, son)[0, 1]))
