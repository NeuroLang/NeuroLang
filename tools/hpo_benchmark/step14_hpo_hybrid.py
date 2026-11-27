"""HPO hybrid (PROTOCOL_HYBRID.md)."""
import difflib, hashlib, json, os, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from hpo_bench import *
from hpo_names import disease_names, term_labels, norm

K = 30
MODEL = "claude-sonnet-5-5"
CACHE = Path("results/hybrid_cache"); CACHE.mkdir(exist_ok=True, parents=True)
W = World(); NAMES = disease_names(); LABELS = term_labels()
cases = load_cases(W.replaced)
test = set(json.load(open("results/hpo_split.json"))["test"])
ch = json.load(open("results/hpo_chosen.json"))["exact"]
real = [c for c in cases if c[0] in test]
used_pids = {json.loads(l)["pid"] for l in open("results/hpo_phase2.jsonl")}
CELLS = [(0, 0), (2, 0), (0, 2), (2, 2)]          # (spurious idx, dropped idx)


def _request(prompt):
    """One answer from Sonnet 5.5: through OpenRouter when HPO_LLM_ROUTE is
    "openrouter" (OPENROUTER_API_KEY, Anthropic preferred as host), else
    directly from Anthropic (ANTHROPIC_API_INRIA_KEY)."""
    if os.environ.get("HPO_LLM_ROUTE") == "openrouter":
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({"model": "anthropic/claude-sonnet-5.5", "max_tokens": 1200,
                             "messages": [{"role": "user", "content": prompt}],
                             "provider": {"order": ["Anthropic"]}}).encode(),
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "content-type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=180))
        u = d.get("usage", {})
        return {"text": d["choices"][0]["message"]["content"] or "",
                "usage": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)},
                "route": "openrouter", "host": d.get("provider"), "model": d.get("model")}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": MODEL, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"x-api-key": os.environ["ANTHROPIC_API_INRIA_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return {"text": "".join(b.get("text", "") for b in d["content"] if b["type"] == "text"), "usage": d["usage"]}


def ask(prompt):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(4):
        try:
            out = _request(prompt); break
        except Exception:
            if attempt == 3: raise
    f.write_text(json.dumps(out)); return out


def terms_block(obs):
    return "\n".join(f"{t}: {LABELS.get(t, '?')}" for t in obs)


def prompt_h0(obs):
    return ("A patient has these Human Phenotype Ontology terms:\n" + terms_block(obs) +
            "\n\nWhich rare diseases are most likely? Reply with a JSON list of at most 10 objects, most likely first, "
            'each {"name": "<disease name as in OMIM or Orphanet>", "omim": "<OMIM:number or empty>"}. No other text.')


def prompt_hybrid(obs, cands, post, with_prob):
    lines = []
    for i in cands:
        d = W.diseases[i]
        lines.append(f"{d} | {NAMES[d]}" + (f" | posterior {100 * post[i]:.1f}%" if with_prob else ""))
    return ("A patient has these Human Phenotype Ontology terms:\n" + terms_block(obs) +
            f"\n\nA statistical model that matches the terms against the annotations of 12,867 rare diseases proposes these {len(cands)} candidates"
            + (" with its posterior probabilities" if with_prob else "") + ":\n" + "\n".join(lines) +
            "\n\nWhich diseases are most likely? Reply with a JSON list of at most 10 disease ids taken from the candidates, "
            "most likely first, for example [\"OMIM:123456\", \"ORPHA:789\"]. No other text.")


def parse_json(text):
    try:
        return json.loads(text[text.index("["):text.rindex("]") + 1])
    except Exception:
        return None


def rank_direct(items, truth):
    tn = norm(NAMES[truth])
    for k, it in enumerate(items or [], 1):
        if isinstance(it, dict) and (str(it.get("omim", "")).strip() == truth or norm(str(it.get("name", ""))) == tn):
            return k
    return None


def rank_ids(ids, truth, allowed):
    ids = [x for x in (ids or []) if isinstance(x, str) and x in allowed]
    for k, x in enumerate(ids, 1):
        if x == truth:
            return k
    return None


def run_patient(args):
    pid, (si, di_) = args
    truth, base = real[pid]
    obs = perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(f"real/{pid}/{si}/{di_}"))
    post = posterior(W, obs, ch)
    order = np.argsort(-post)
    cands = [int(i) for i in order[:K]]
    allowed = {W.diseases[i] for i in cands}
    shuffled = cands[:]; random.Random(f"shuf/{pid}").shuffle(shuffled)
    rec = {"pid": pid, "disease": truth, "spur": SPUR_LEVELS[si], "drop": DROP_LEVELS[di_], "n_obs": len(obs),
           "recall_k": truth in allowed, "posterior_rank": rank_of(post, W.di[truth])}
    r0 = ask(prompt_h0(obs)); rec["H0"] = rank_direct(parse_json(r0["text"]), truth)
    for arm, order_, prob in (("H1", shuffled, False), ("H2", shuffled, True), ("H3", cands, False)):
        r = ask(prompt_hybrid(obs, order_, post, prob))
        parsed = parse_json(r["text"]); rec[arm] = rank_ids(parsed, truth, allowed)
        rec[arm + "_parsed"] = parsed is not None
    rec["tokens"] = None
    return rec


def main():
    rng = random.Random(11)
    by_dis = {}
    for pid, (d, obs) in enumerate(real):
        if pid not in used_pids and d in W.di:
            by_dis.setdefault(d, []).append(pid)
    dis = sorted(by_dis); rng.shuffle(dis)
    plan = [(rng.choice(by_dis[d]), CELLS[k % 4]) for k, d in enumerate(dis[:100])]
    print("patients", len(plan), "diseases", len({real[p][0] for p, _ in plan}), flush=True)

    with ThreadPoolExecutor(6) as pool:
        recs = list(pool.map(run_patient, plan))
    json.dump(recs, open("results/hybrid_results.json", "w"))
    rk = lambda x: np.inf if x is None else float(x)
    def met(rs, k):
        r = np.array([rk(x[k]) for x in rs]); return 100 * (r == 1).mean(), 100 * (r <= 10).mean(), np.mean(1 / r)
    print("recall@%d: %.1f%%" % (K, 100 * np.mean([r["recall_k"] for r in recs])), "| unparsed", {a: sum(not r[a + "_parsed"] for r in recs) for a in ("H1", "H2", "H3")})
    arms = [("posterior alone", "posterior_rank"), ("H0 direct", "H0"), ("H1 candidates", "H1"), ("H2 cand.+probabilities", "H2"), ("H3 cand. ranked", "H3")]
    rngb = random.Random(0); n = len(recs)
    draws = [[rngb.randrange(n) for _ in range(n)] for _ in range(5000)]
    print(f"{'arm':24} {'top1':>5} {'top10':>6} {'MRR':>6} {'95% CI':>15}")
    for name, key in arms:
        t1, t10, m = met(recs, key)
        bs = sorted(met([recs[i] for i in d], key)[2] for d in draws)
        print(f"{name:24} {t1:5.1f} {t10:6.1f} {m:6.3f} [{bs[125]:.3f},{bs[4874]:.3f}]")
    print("\npaired MRR differences, 95% CI over patients")
    for a, b in (("H1", "H0"), ("H2", "H0"), ("H3", "H0"), ("H2", "H1"), ("H3", "H1"), ("H1", "posterior_rank"), ("H2", "posterior_rank"), ("H3", "posterior_rank")):
        f = lambda rs: met(rs, a)[2] - met(rs, b)[2]
        bs = sorted(f([recs[i] for i in d]) for d in draws)
        print(f"  {a} - {b:15} {f(recs):+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}]")
    for lab, sel in (("true disease among candidates", lambda r: r["recall_k"]), ("not among candidates", lambda r: not r["recall_k"])):
        rs = [r for r in recs if sel(r)]
        print(f"\n{lab} (n={len(rs)}): " + " | ".join(f"{name.split()[0]} {met(rs, key)[2]:.3f}" for name, key in arms))


if __name__ == "__main__":
    main()
