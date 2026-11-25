import json, collections, random
import numpy as np
rows = [json.loads(l) for l in open("results/hpo_phase2.jsonl")]
n = len(rows); print("patients", n, "| diseases", len({r["disease"] for r in rows}), "| spent $%.2f" % rows[-1]["spent_usd_cumulative"])
def rk(x): return np.inf if x is None else float(x)
SYS = {
  "NeuroLang posterior (numpy ref = Phase 1)": lambda r: rk(r["phase1"]["exact_leak"]),
  "Overlap baseline (Phase 1)": lambda r: rk(r["phase1"]["overlap"]),
  "ReAct agent + NeuroLang": lambda r: rk(r["agent"].get("rank")),
  "Sonnet direct, strict match": lambda r: rk(r["direct"].get("rank_strict")),
  "Sonnet direct, lenient match": lambda r: rk(r["direct"].get("rank_lenient")),
}
def met(rs, f):
    r = np.array([f(x) for x in rs]); return (r == 1).mean(), (r <= 10).mean(), np.mean(1 / r)
def boot(rs, g, B=2000, seed=0):
    rng = random.Random(seed); v = sorted(g([rs[rng.randrange(len(rs))] for _ in rs]) for _ in range(B)); return v[int(.025*B)], v[int(.975*B)]
def table(rs, label):
    print(f"\n-- {label} (n={len(rs)})")
    for name, f in SYS.items():
        t1, t10, mrr = met(rs, f); lo, hi = boot(rs, lambda z: met(z, f)[2])
        print(f"   {name:44} top1 {100*t1:5.1f}  top10 {100*t10:5.1f}  MRR {mrr:.3f} [{lo:.3f},{hi:.3f}]")
table(rows, "all")
for c in [(0, 0.0), (2, 0.0), (0, 0.5), (2, 0.5)]:
    table([r for r in rows if (r["spur"], r["drop"]) == c], f"spurious={c[0]}, dropped={int(c[1]*100)}%")
print("\n-- paired MRR differences (bootstrap over patients)")
for a, b in [("ReAct agent + NeuroLang", "Sonnet direct, strict match"), ("ReAct agent + NeuroLang", "Sonnet direct, lenient match"),
             ("ReAct agent + NeuroLang", "Overlap baseline (Phase 1)"), ("Sonnet direct, lenient match", "Overlap baseline (Phase 1)")]:
    g = lambda z, a=a, b=b: met(z, SYS[a])[2] - met(z, SYS[b])[2]
    d = g(rows); lo, hi = boot(rows, g); print(f"   {a} - {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
ag = [r["agent"] for r in rows]
print("\n-- agent diagnostics")
print("valid answers", sum(a.get("valid", False) for a in ag), "/", n, "| exceptions", sum("exception" in a for a in ag), "| out of steps", sum(a.get("out_of_steps", False) for a in ag), "| no prob column", sum(a.get("no_probability_column", False) for a in ag))
sc = [a for a in ag if a.get("rank") is not None or "ref_rank" in a]
print("ranked by agent:", sum(a.get("rank") is not None for a in ag), "| rank equals reference rank:", sum(a.get("rank_equals_ref", False) for a in ag))
d = [a["max_abs_diff_vs_posterior"] for a in ag if a.get("max_abs_diff_vs_posterior") is not None]
print("agent probabilities equal the exact posterior (max abs diff < 1e-6):", sum(x < 1e-6 for x in d), "/", len(d), "| worst", max(d) if d else None)
print("steps (calls) median", np.median([a.get("calls", 0) for a in ag]), "max", max(a.get("calls", 0) for a in ag), "| median seconds", np.median([a["seconds"] for a in ag]))
dr = [r["direct"] for r in rows]
print("\n-- direct diagnostics: exceptions", sum("exception" in x for x in dr), "| answers per patient median", np.median([x.get("n_answers", 0) for x in dr]),
      "| unmatched names:", sum(x.get("unmatched_names", 0) for x in dr), "of", sum(x.get("n_answers", 0) for x in dr))
tok = lambda k: sum(a.get("usage", {}).get(k, 0) for a in ag)
print("tokens agent in/cached/out:", tok("input"), tok("cached"), tok("output"), "| direct in/out:", sum(x.get("usage", {}).get("input_tokens", 0) for x in dr), sum(x.get("usage", {}).get("output_tokens", 0) for x in dr))
