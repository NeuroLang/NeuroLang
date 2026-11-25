"""Phase 1 on the held-out test diseases: real and simulated sets, all noise cells."""
import json, random, resource, time
from hpo_bench import *

W = World()
cases = load_cases(W.replaced)
split = json.load(open("results/hpo_split.json")); test = set(split["test"])
ch = json.load(open("results/hpo_chosen.json"))
SYSTEMS = {
    "exact_plain": dict(k_up=0, k_down=0, gamma=1.0, eps=0.0),
    "exact_leak": ch["exact"],
    "ancestor_leak": ch["anc"],
}
print("systems", SYSTEMS, flush=True)
real = [c for c in cases if c[0] in test]
sim_rng = random.Random(123)
sim = [simulate(W, sim_rng, sim_rng.choice([2, 3, 4])) for _ in range(200)]
out = open("results/hpo_phase1.jsonl", "w")

def score(setname, pid, truth, base_obs, si, di_):
    rng = random.Random(f"{setname}/{pid}/{si}/{di_}")
    obs = perturb(W, base_obs, SPUR_LEVELS[si], DROP_LEVELS[di_], rng)
    rec = dict(set=setname, pid=pid, disease=truth, spur=SPUR_LEVELS[si], drop=DROP_LEVELS[di_],
               n_obs=len(obs), candidate=truth in W.di)
    i = W.di.get(truth)
    for name, cfg in SYSTEMS.items():
        post = posterior(W, obs, cfg)
        rec[name] = dict(rank=rank_of(post, i) if i is not None else None, empty=bool(post.sum() == 0))
    sc = overlap_scores(W, obs)
    rec["overlap"] = dict(rank=rank_of(sc, i) if i is not None else None, empty=bool(sc.max() <= 0))
    out.write(json.dumps(rec) + "\n")

t0 = time.time()
for setname, data in (("real", real), ("sim", sim)):
    for pid, (truth, obs) in enumerate(data):
        for si in range(len(SPUR_LEVELS)):
            for di_ in range(len(DROP_LEVELS)):
                score(setname, pid, truth, obs, si, di_)
        if pid % 50 == 0:
            out.flush(); print(setname, pid, "/", len(data), round(time.time() - t0), "s", flush=True)
out.close()
print("DONE", round(time.time() - t0), "s; maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2**20, flush=True)
