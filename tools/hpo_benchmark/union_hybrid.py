"""HPO union hybrid (PROTOCOL_UNION.md).

    python union_hybrid.py dev       # free: gate and fusion on used patients
    python union_hybrid.py plan      # the fresh patients, results/union_plan.json
    python union_hybrid.py run       # Sonnet arms (ANTHROPIC_API_INRIA_KEY)
    python union_hybrid.py analyse   # results/union_analysis.txt

Arms on the fresh patients: H0 Sonnet direct; U1 Sonnet re-ranks the union
of the posterior's top 30 (with probabilities) and its own H0 list; U2
reciprocal-rank fusion of H0 and the posterior; U3 a gate (posterior when a
truth-free confidence proxy is high, H0 otherwise), if ``dev`` keeps it.
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import step14_hpo_hybrid as h
from hpo_bench import DROP_LEVELS, SPUR_LEVELS, perturb, posterior, rank_of
from hpo_names import norm

K = 30
RRF_K = 60          # the usual constant of reciprocal-rank fusion, not fitted
DEPTH = 100         # posterior ranks that enter the fusion
BOOT = 10000
NAME_TO_ID = {}
for _d, _n in sorted(h.NAMES.items()):
    NAME_TO_ID.setdefault(norm(_n), _d)


def observed(pid, si, di_):
    truth, base = h.real[pid]
    return perturb(h.W, base, SPUR_LEVELS[si], DROP_LEVELS[di_],
                   random.Random(f"real/{pid}/{si}/{di_}"))


def ranked(post):
    """Disease indices by posterior, ties in index order (stable)."""
    return np.argsort(-post, kind="stable")


def item_id(item):
    """The disease id an answer item names, or None."""
    if not isinstance(item, dict):
        return None
    omim = str(item.get("omim", "")).strip()
    if omim in h.W.di:
        return omim
    return NAME_TO_ID.get(norm(str(item.get("name", ""))))


def fused_rank(h0_items, post, truth):
    """Rank of the truth in the reciprocal-rank fusion of H0 and the
    posterior's top DEPTH (None if in neither)."""
    score = {}
    for r, item in enumerate(h0_items or [], 1):
        d = item_id(item)
        if d is not None and d not in score:
            score[d] = 1 / (RRF_K + r)
    for r, i in enumerate(ranked(post)[:DEPTH], 1):
        if post[i] > 0:
            d = h.W.diseases[i]
            score[d] = score.get(d, 0) + 1 / (RRF_K + r)
    order = sorted(score, key=lambda d: (-score[d], d))
    return order.index(truth) + 1 if truth in order else None


def proxies(obs, post):
    """Truth-free confidence of the posterior: its top probability, and the
    share of the patient's terms its top candidate is annotated with."""
    top = h.W.diseases[int(ranked(post)[0])]
    return {
        "top_probability": float(post.max()),
        "top_coverage": float(np.mean([t in h.W.by_dis[top] for t in obs])),
    }


def rr(rank):
    return 0.0 if rank is None else 1.0 / rank


def used_patients():
    """The 160 patients of phase 2 and of the first hybrid, with the
    posterior's and Sonnet's direct ranks, as post_hoc_coverage.py has them."""
    rows = []
    for r in json.load(open("results/hybrid_results.json")):
        si, di_ = SPUR_LEVELS.index(r["spur"]), DROP_LEVELS.index(r["drop"])
        rows.append(dict(pid=r["pid"], cell=(si, di_),
                         post=r["posterior_rank"], h0=r["H0"]))
    for line in open("results/hpo_phase2.jsonl"):
        r = json.loads(line)
        si, di_ = SPUR_LEVELS.index(r["spur"]), DROP_LEVELS.index(r["drop"])
        rows.append(dict(pid=r["pid"], cell=(si, di_),
                         post=r["phase1"]["exact_leak"],
                         h0=r["direct"].get("rank_strict")))
    return rows


def dev():
    rows = used_patients()
    for r in rows:
        obs = observed(r["pid"], *r["cell"])
        r.update(proxies(obs, posterior(h.W, obs, h.ch)))
    post = np.array([rr(r["post"]) for r in rows])
    son = np.array([rr(r["h0"]) for r in rows])
    gap = post - son
    out = {"patients": len(rows), "posterior_mrr": post.mean(),
           "h0_mrr": son.mean(), "proxies": {}}
    print(f"{len(rows)} used patients: posterior MRR {post.mean():.3f}, "
          f"H0 {son.mean():.3f}")
    rng = np.random.default_rng(0)
    for name in ("top_probability", "top_coverage"):
        x = np.array([r[name] for r in rows])
        rho = lambda idx: np.corrcoef(  # noqa: E731
            np.argsort(np.argsort(x[idx])), np.argsort(np.argsort(gap[idx]))
        )[0, 1]
        bs = np.sort([rho(rng.integers(0, len(x), len(x)))
                      for _ in range(2000)])
        best = max(
            ((np.mean(np.where(x >= t, post, son)), t) for t in np.unique(x)),
        )
        gain = best[0] - max(post.mean(), son.mean())
        out["proxies"][name] = dict(
            spearman=rho(np.arange(len(x))), ci=[bs[50], bs[1949]],
            threshold=best[1], gated_mrr=best[0], gain=gain,
        )
        print(f"  {name:16} Spearman with posterior - H0 "
              f"{rho(np.arange(len(x))):+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]"
              f"; best gate >= {best[1]:.4f}: MRR {best[0]:.3f} "
              f"({gain:+.3f} over the better arm, in sample)")
    keep = {n: p for n, p in out["proxies"].items()
            if p["ci"][0] > 0 and p["gain"] >= 0.02}
    chosen = max(keep, key=lambda n: keep[n]["gated_mrr"]) if keep else None
    out["gate"] = None if chosen is None else dict(
        proxy=chosen, threshold=keep[chosen]["threshold"])
    print("gate kept:", out["gate"])
    # fusion, development only: the 100 hybrid patients, whose H0 answers
    # are cached
    fr, hr, pr = [], [], []
    for r in json.load(open("results/hybrid_results.json")):
        si, di_ = SPUR_LEVELS.index(r["spur"]), DROP_LEVELS.index(r["drop"])
        obs = observed(r["pid"], si, di_)
        p = posterior(h.W, obs, h.ch)
        items = h.parse_json(h.ask(h.prompt_h0(obs))["text"])
        fr.append(rr(fused_rank(items, p, r["disease"])))
        hr.append(rr(r["H0"]))
        pr.append(rr(r["posterior_rank"]))
    out["fusion_dev"] = dict(rrf=np.mean(fr), h0=np.mean(hr),
                             posterior=np.mean(pr))
    print(f"fusion (development, 100 hybrid patients): RRF MRR "
          f"{np.mean(fr):.3f}, H0 {np.mean(hr):.3f}, posterior "
          f"{np.mean(pr):.3f}")
    json.dump(out, open("results/union_dev.json", "w"), indent=1,
              default=float)


def plan():
    used = {r["pid"] for r in used_patients()}
    fresh = sorted(p for p, (d, _) in enumerate(h.real)
                   if p not in used and d in h.W.di)
    random.Random(12).shuffle(fresh)
    out = [dict(pid=p, disease=h.real[p][0], cell=h.CELLS[k % 4])
           for k, p in enumerate(fresh)]
    json.dump(out, open("results/union_plan.json", "w"), indent=1)
    print(f"{len(out)} fresh patients, "
          f"{len({r['disease'] for r in out})} diseases")


def prompt_union(obs, post, h0_items):
    cands = [int(i) for i in ranked(post)[:K]]
    model = "\n".join(
        f"{h.W.diseases[i]} | {h.NAMES[h.W.diseases[i]]} | posterior "
        f"{100 * post[i]:.1f}%" for i in cands
    )
    earlier = "\n".join(
        f"{it.get('name', '')} | {it.get('omim', '')}"
        for it in (h0_items or []) if isinstance(it, dict)
    ) or "(none)"
    return (
        "A patient has these Human Phenotype Ontology terms:\n"
        + h.terms_block(obs)
        + "\n\nTwo sources propose diseases. A statistical model that matches "
        "the terms against the annotations of 12,867 rare diseases, with its "
        f"posterior probabilities ({K} candidates, most probable first):\n"
        + model
        + "\n\nAn earlier answer of yours, from the terms alone (most likely "
        "first):\n" + earlier
        + "\n\nWhich rare diseases are most likely? Use both sources; you may "
        "also name a disease in neither. Reply with a JSON list of at most 10 "
        "objects, most likely first, each {\"name\": \"<disease name as in "
        "OMIM or Orphanet>\", \"omim\": \"<OMIM:number or empty>\"}. No other "
        "text."
    )


def run_patient(p):
    pid, truth = p["pid"], p["disease"]
    obs = observed(pid, *p["cell"])
    post = posterior(h.W, obs, h.ch)
    h0 = h.ask(h.prompt_h0(obs))
    h0_items = h.parse_json(h0["text"])
    u1 = h.ask(prompt_union(obs, post, h0_items))
    u1_items = h.parse_json(u1["text"])
    return dict(
        pid=pid, disease=truth, cell=p["cell"], n_obs=len(obs),
        posterior=rank_of(post, h.W.di[truth]),
        H0=h.rank_direct(h0_items, truth),
        U1=h.rank_direct(u1_items, truth),
        U2=fused_rank(h0_items, post, truth),
        recall_k=h.W.di[truth] in set(int(i) for i in ranked(post)[:K]),
        parsed=dict(H0=h0_items is not None, U1=u1_items is not None),
        usage=[h0["usage"], u1["usage"]],
        **proxies(obs, post),
    )


def run():
    patients = json.load(open("results/union_plan.json"))
    with ThreadPoolExecutor(6) as pool:
        recs = list(pool.map(run_patient, patients))
    json.dump(recs, open("results/union_results.json", "w"), indent=1)
    analyse()


def analyse():
    recs = json.load(open("results/union_results.json"))
    gate = json.load(open("results/union_dev.json"))["gate"]
    for r in recs:
        if gate:
            r["U3"] = (r["posterior"] if r[gate["proxy"]] >= gate["threshold"]
                       else r["H0"])
    arms = ["posterior", "H0", "U1", "U2"] + (["U3"] if gate else [])
    by = {}
    for j, r in enumerate(recs):
        by.setdefault(r["disease"], []).append(j)
    keys = sorted(by)
    rng = random.Random(0)
    draws = [[j for d in (rng.choice(keys) for _ in keys) for j in by[d]]
             for _ in range(BOOT)]
    RR = {a: np.array([rr(r[a]) for r in recs]) for a in arms}
    lines = [f"{len(recs)} patients, {len(keys)} diseases; recall@{K} "
             f"{100 * np.mean([r['recall_k'] for r in recs]):.1f}%; unparsed "
             + str({a: sum(not r["parsed"][a] for r in recs)
                    for a in ("H0", "U1")}),
             f"gate: {gate}",
             f"{'arm':10} {'top1':>6} {'top10':>6} {'MRR':>6} "
             "95% CI (resampling diseases)"]
    for a in arms:
        ranks = np.array([np.inf if r[a] is None else r[a] for r in recs])
        bs = np.sort([RR[a][d].mean() for d in draws])
        lines.append(
            f"{a:10} {100 * (ranks == 1).mean():6.1f} "
            f"{100 * (ranks <= 10).mean():6.1f} {RR[a].mean():6.3f} "
            f"[{bs[int(.025 * BOOT)]:.3f}, {bs[int(.975 * BOOT) - 1]:.3f}]")
    lines.append("paired MRR differences")
    pairs = [("U1", "H0"), ("U2", "H0"), ("U1", "posterior"),
             ("U2", "posterior"), ("U1", "U2")]
    if gate:
        pairs += [("U3", "H0"), ("U3", "posterior")]
    for a, b in pairs:
        d = RR[a] - RR[b]
        bs = np.sort([d[x].mean() for x in draws])
        lines.append(f"  {a} - {b:10} {d.mean():+.3f} "
                     f"[{bs[int(.025 * BOOT)]:+.3f}, "
                     f"{bs[int(.975 * BOOT) - 1]:+.3f}]")
    tokens = [u for r in recs for u in r["usage"]]
    cost = sum(2e-6 * u["input_tokens"] + 10e-6 * u["output_tokens"]
               for u in tokens)
    lines.append(f"cost at $2/$10 per million tokens: ${cost:.2f}")
    text = "\n".join(lines)
    print(text)
    open("results/union_analysis.txt", "w").write(text + "\n")


if __name__ == "__main__":
    dict(dev=dev, plan=plan, run=run, analyse=analyse)[sys.argv[1]]()
