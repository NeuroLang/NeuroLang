"""Lifted-only NeuroLang posterior vs the numpy reference, on sampled test patients."""
import json, random, resource, time
from hpo_bench import *
from probe_common import make_nl, run_item
W = World(); cases = load_cases(W.replaced)
test = set(json.load(open("results/hpo_split.json"))["test"]); ch = json.load(open("results/hpo_chosen.json"))
real = [c for c in cases if c[0] in test]
nl = make_nl()
nl.add_probabilistic_facts_from_tuples([(p, d, s) for (d, s), p in W.facts.items()], name="H")
nl.add_uniform_probabilistic_choice_over_set([(d,) for d in W.diseases], name="D")
rng = random.Random(7); N = 150
for k in range(N):
    truth, base = rng.choice(real); si, di_ = rng.randrange(4), rng.randrange(3)
    obs = perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(f"chk/{k}"))
    for name in ("exact_leak", "ancestor_leak"):
        cfg = ch["exact"] if name == "exact_leak" else ch["anc"]
        def fn():
            M = W.probs(obs, cfg); rel = f"HL_{k}_{name}"
            rows = [(float(M[j, i]), W.diseases[i], t) for j, t in enumerate(obs) for i in range(W.n) if M[j, i] > 0]
            nl.add_probabilistic_facts_from_tuples(rows, name=rel)
            body = " & ".join(f'{rel}(d, "{t}")' for t in obs); cond = " & ".join(f'{rel}(e, "{t}")' for t in obs)
            with nl.scope:
                res = dict(nl.execute_datalog_program(f"post(d, PROB) :- (D(d) & {body}) // (D(e) & {cond})\nans(d, p) :- post(d, p)"))
            ref = posterior(W, obs, cfg)
            err = max(abs(ref[W.di[x]] - p) for x, p in res.items())
            tied = int(sum(1 for i in range(W.n) if ref[i] > 0))
            return dict(max_abs_err=err, n_answers=len(res), n_supported_ref=tied, same_support=len(res) == tied,
                        rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2**20)
        r = run_item("hpo_lifted", f"{k}/{name}", fn, timeout=240, out="hpo_lifted_check.jsonl", spur=SPUR_LEVELS[si], drop=DROP_LEVELS[di_], n_obs=len(obs), model=name)
    if k % 10 == 0: print(k, r["status"], r.get("error_type") or "", r.get("result"), flush=True)
