import json, collections, random
import numpy as np
rows = [json.loads(l) for l in open("results/hpo_phase1.jsonl")]
SYS = ["exact_plain", "exact_leak", "ancestor_leak", "overlap"]
def metrics(rs, s):
    r = np.array([x[s]["rank"] if x[s]["rank"] else np.inf for x in rs], float)
    return dict(top1=(r == 1).mean(), top10=(r <= 10).mean(), mrr=np.mean(1 / r), empty=np.mean([x[s]["empty"] for x in rs]))
def boot(rs, f, n=1000, seed=0):
    by = collections.defaultdict(list)
    for x in rs: by[x["disease"]].append(x)
    keys = sorted(by); rng = random.Random(seed); v = []
    for _ in range(n):
        samp = [x for k in rng.choices(keys, k=len(keys)) for x in by[k]]
        v.append(f(samp))
    v.sort(); return v[int(.025 * n)], v[int(.975 * n)]
for setname in ("real", "sim"):
    R = [x for x in rows if x["set"] == setname]
    print(f"\n=== {setname}: {len(R)} perturbed patients, {len({x['pid'] for x in R})} base patients, {len({x['disease'] for x in R})} diseases; true disease not a candidate: {sum(not x['candidate'] for x in R)//12} base cases")
    for label, sel in [("all noise cells", lambda x: True), ("clean (0 spurious, 0% dropped)", lambda x: x["spur"] == 0 and x["drop"] == 0),
                       ("1 spurious", lambda x: x["spur"] == 1 and x["drop"] == 0), ("4 spurious", lambda x: x["spur"] == 4 and x["drop"] == 0),
                       ("50% dropped", lambda x: x["spur"] == 0 and x["drop"] == 0.5), ("4 spurious + 50% dropped", lambda x: x["spur"] == 4 and x["drop"] == 0.5)]:
        rs = [x for x in R if sel(x)]
        print(f"-- {label} (n={len(rs)})")
        for s in SYS:
            m = metrics(rs, s); lo, hi = boot(rs, lambda z, s=s: metrics(z, s)["mrr"], 300)
            print(f"   {s:14} top1 {100*m['top1']:5.1f}  top10 {100*m['top10']:5.1f}  MRR {m['mrr']:.3f} [{lo:.3f},{hi:.3f}]  empty {100*m['empty']:5.1f}%")
    print("-- paired MRR differences, all cells (bootstrap over diseases, 95% CI)")
    for a, b in [("exact_leak", "overlap"), ("exact_leak", "exact_plain"), ("ancestor_leak", "exact_leak")]:
        f = lambda z, a=a, b=b: metrics(z, a)["mrr"] - metrics(z, b)["mrr"]
        d = f(R); lo, hi = boot(R, f, 1000); print(f"   {a} - {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
