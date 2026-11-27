"""Choose eps / gamma / k_up / k_down on the development diseases only (30%)."""
import itertools, json, random, time
from hpo_bench import *
W = World()
cases = load_cases(W.replaced)
dev, test = split_diseases(cases)
json.dump(dict(dev=sorted(dev), test=sorted(test)), open("results/hpo_split.json", "w"))
dev_cases = [c for c in cases if c[0] in dev]
print("cases", len(cases), "| dev diseases", len(dev), "cases", len(dev_cases), "| test diseases", len(test), "cases", sum(c[0] in test for c in cases), flush=True)

def patients(seed=0):
    out = []
    for ci, (d, obs) in enumerate(dev_cases):
        for si, spur in enumerate(SPUR_LEVELS):
            for di_, drop in enumerate(DROP_LEVELS):
                rng = random.Random(f"{seed}/{ci}/{si}/{di_}")
                out.append((d, perturb(W, obs, spur, drop, rng)))
    return out
P = patients()
def evaluate(cfg):
    rr = []
    for d, obs in P:
        post = posterior(W, obs, cfg)
        r = rank_of(post, W.di[d]) if d in W.di else None
        rr.append(1 / r if r else 0.0)
    return float(np.mean(rr))
grid = []
for eps in (0.005, 0.02, 0.05):
    grid.append(dict(k_up=0, k_down=0, gamma=1.0, eps=eps))
    for (ku, kd), gamma in itertools.product(((1, 0), (1, 1), (2, 2)), (0.25, 0.5, 1.0)):
        grid.append(dict(k_up=ku, k_down=kd, gamma=gamma, eps=eps))
res = []
for cfg in grid:
    t0 = time.time(); m = evaluate(cfg); res.append((m, cfg))
    print(f"MRR {m:.4f}  {cfg}  ({time.time()-t0:.0f}s)", flush=True)
exact = max((r for r in res if r[1]["k_up"] == 0), key=lambda r: r[0])
anc = max((r for r in res if r[1]["k_up"] > 0), key=lambda r: r[0])
json.dump(dict(exact=exact[1], exact_dev_mrr=exact[0], anc=anc[1], anc_dev_mrr=anc[0],
               grid=[dict(mrr=m, **c) for m, c in res], n_dev_patients=len(P)), open("results/hpo_chosen.json", "w"), indent=1)
print("CHOSEN exact:", exact, "\nCHOSEN ancestor:", anc)
