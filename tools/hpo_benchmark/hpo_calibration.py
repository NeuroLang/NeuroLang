"""Calibration of the posterior over diseases (top-label reliability, true-disease probability)."""
import json, random, collections
import numpy as np
from hpo_bench import *
import hpo_enriched as H

W = H.W
ch = json.load(open("results/hpo_chosen.json"))["exact"]
en = json.load(open("results/hpo_enriched_chosen.json")); E = H.Enriched(en["a"])
test = set(json.load(open("results/hpo_split.json"))["test"])
real = [c for c in H.cases if c[0] in test]

def collect(model, patients):
    out = []
    for pid, d, si, di_, obs, pm in patients:
        if d not in W.di: continue
        post = posterior(W, obs, ch) if model == "hpoa" else H.posterior_enriched(E, obs, en["eps"], d, pm)
        i = W.di[d]; top = int(post.argmax())
        out.append(dict(conf=float(post[top]), hit=int(top == i), p_true=float(post[i]), si=si, di=di_, disease=d,
                        entropy=float(-(post[post > 0] * np.log(post[post > 0])).sum())))
    return out

def report(rs, label):
    conf = np.array([r["conf"] for r in rs]); hit = np.array([r["hit"] for r in rs]); pt = np.array([r["p_true"] for r in rs])
    bins = [0, .1, .25, .5, .75, .9, 1.0001]
    ece = 0.0; lines = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
            lines.append(f"      conf in [{lo:.2f},{min(hi,1):.2f}): n={m.sum():5d}  mean conf {conf[m].mean():.3f}  top-1 correct {hit[m].mean():.3f}")
    ll = -np.log(np.clip(pt, 1e-12, 1)).mean()
    print(f"  {label:34} n={len(rs):5d} | top-1 acc {hit.mean():.3f} | mean top-1 confidence {conf.mean():.3f} | ECE {ece:.3f} | mean P(true) {pt.mean():.4f} | P(true)<1e-6: {100*(pt<1e-6).mean():.1f}% | log-loss {ll:.2f}")
    return lines

# real patients
for model in ("hpoa", "enriched"):
    P = H.patients(list(enumerate(real)))
    rs = collect(model, P)
    print(f"\n=== real test set, model = {model}")
    lines = report(rs, "all cells")
    print("\n".join(lines))
    for lab, sel in (("clean", lambda r: r["si"] == 0 and r["di"] == 0), ("4 spurious", lambda r: r["si"] == 3 and r["di"] == 0), ("50% dropped", lambda r: r["si"] == 0 and r["di"] == 2)):
        report([r for r in rs if sel(r)], lab)
    json.dump(rs, open(f"results/hpo_calib_real_{model}.json", "w"))

# simulated: the model is correct by construction (leak 0 for clean cell), a sanity check
sim_rng = random.Random(123)
sim = [simulate(W, sim_rng, sim_rng.choice([2, 3, 4])) for _ in range(2000)]
rs = []
plain = dict(k_up=0, k_down=0, gamma=1.0, eps=0.0)
for truth, obs in sim:
    post = posterior(W, obs, plain); i = W.di[truth]; top = int(post.argmax())
    rs.append(dict(conf=float(post[top]), hit=int(top == i), p_true=float(post[i]), si=0, di=0, disease=truth, entropy=0))
print("\n=== simulated clean patients, plain model (model correct by construction, 2000 patients)")
print("\n".join(report(rs, "simulated clean")))
