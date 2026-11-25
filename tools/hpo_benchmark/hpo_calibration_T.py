"""Temperature scaling of the HPOA-only leak posterior: fit T on dev diseases, evaluate on test."""
import json, random
import numpy as np
from hpo_bench import *
W = World(); cases = load_cases(W.replaced)
sp = json.load(open("results/hpo_split.json")); dev, test = set(sp["dev"]), set(sp["test"])
cfg = json.load(open("results/hpo_chosen.json"))["exact"]

def lls(case_list, seedfmt):
    out = []
    for ci, (d, base) in case_list:
        if d not in W.di: continue
        for si in range(4):
            for di_ in range(3):
                obs = perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(seedfmt(ci, si, di_)))
                M = W.probs(obs, cfg)
                with np.errstate(divide="ignore"): ll = np.log(M).sum(axis=0)
                out.append((ll, W.di[d]))
    return out
dev_cases = list(enumerate([c for c in cases if c[0] in dev]))
test_cases = list(enumerate([c for c in cases if c[0] in test]))
DEV = lls(dev_cases, lambda ci, si, di_: f"0/{ci}/{si}/{di_}")
TEST = lls(test_cases, lambda ci, si, di_: f"real/{ci}/{si}/{di_}")
def post(ll, T):
    z = ll / T
    if np.all(np.isinf(z)): return np.zeros_like(ll)
    w = np.exp(z - z.max()); return w / w.sum()
def stats(data, T):
    conf, hit, pt = [], [], []
    for ll, i in data:
        p = post(ll, T); top = int(p.argmax()); conf.append(p[top]); hit.append(top == i); pt.append(p[i])
    conf, hit, pt = map(np.array, (conf, hit, pt))
    bins = [0, .1, .25, .5, .75, .9, 1.0001]; ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum(): ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return conf.mean(), hit.mean(), ece, -np.log(np.clip(pt, 1e-12, 1)).mean(), hit
grid = [1, 2, 4, 8, 16, 32, 64, 128]
dl = {T: stats(DEV, T)[3] for T in grid}
Tbest = min(dl, key=dl.get); print("dev log-loss by T:", {T: round(v, 2) for T, v in dl.items()}, "-> chosen T =", Tbest)
for T in (1, Tbest):
    c, a, e, l, hit = stats(TEST, T)
    print(f"test, T={T:>3}: mean top-1 confidence {c:.3f} | top-1 accuracy {a:.3f} | ECE {e:.3f} | log-loss {l:.2f}")
