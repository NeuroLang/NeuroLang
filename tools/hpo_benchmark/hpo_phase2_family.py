"""Score the symbolic systems with the same lenient (name-similarity >= 0.85) rule as Sonnet direct."""
import difflib, json, random
import numpy as np
from hpo_bench import *
from hpo_names import disease_names, norm
W = World(); N = disease_names(); NN = {d: norm(N[d]) for d in W.diseases}
cases = load_cases(W.replaced); test = set(json.load(open("results/hpo_split.json"))["test"])
real = [c for c in cases if c[0] in test]
ch = json.load(open("results/hpo_chosen.json"))["exact"]
rows = [json.loads(l) for l in open("results/hpo_phase2.jsonl")]
def family(truth):
    t = NN[truth]; s = difflib.SequenceMatcher(); s.set_seq2(t)
    out = []
    for d, n in NN.items():
        s.set_seq1(n)
        if s.real_quick_ratio() >= 0.85 and s.quick_ratio() >= 0.85 and s.ratio() >= 0.85: out.append(W.di[d])
    return out
def best_rank(scores, idx):
    r = [rank_of(scores, i) for i in idx]; r = [x for x in r if x]
    return min(r) if r else None
out = []
for r in rows:
    truth = r["disease"]; base = real[r["pid"]][1]
    si, di_ = SPUR_LEVELS.index(r["spur"]), DROP_LEVELS.index(r["drop"])
    obs = perturb(W, base, r["spur"], r["drop"], random.Random(f"real/{r['pid']}/{si}/{di_}"))
    fam = family(truth)
    out.append(dict(n_family=len(fam), posterior=best_rank(posterior(W, obs, ch), fam), overlap=best_rank(overlap_scores(W, obs), fam),
                    direct_lenient=r["direct"].get("rank_lenient")))
json.dump(out, open("results/hpo_phase2_family.json", "w"))
rk = lambda x: np.inf if x is None else float(x)
print("family size median", np.median([o["n_family"] for o in out]), "| patients with family size 1:", sum(o["n_family"] == 1 for o in out), "of", len(out))
for k in ("posterior", "overlap", "direct_lenient"):
    r = np.array([rk(o[k]) for o in out]); print(f"{k:15} top1 {100*(r==1).mean():5.1f}  top10 {100*(r<=10).mean():5.1f}  MRR {np.mean(1/r):.3f}")
rng = random.Random(0)
d = [1 / rk(o["direct_lenient"]) - 1 / rk(o["posterior"]) for o in out]
bs = sorted(np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(2000))
print(f"paired MRR, Sonnet direct minus posterior (family level): {np.mean(d):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]")
