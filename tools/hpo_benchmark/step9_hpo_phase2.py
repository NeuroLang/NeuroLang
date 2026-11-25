"""Phase 2: ReAct agent (Sonnet) writing the NeuroLang query, and Sonnet answering directly."""
import difflib, json, os, random, signal, sys, time, urllib.request
from pathlib import Path
from hpo_bench import *
from hpo_names import disease_names, term_labels, norm
from probe_common import make_nl, RESULTS
from neurolang_apps.utils.llm.config import load_config
from neurolang_apps.utils.llm.agent import ReactAgent

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = RESULTS / (sys.argv[2] if len(sys.argv) > 2 else "hpo_phase2.jsonl")
CAP_USD = 60.0
PRICE = dict(base=2.0, write=2.5, read=0.10, out=10.0)       # USD per MTok, Sonnet 5.5
MODEL = "claude-sonnet-5-5"; KEY = os.environ["ANTHROPIC_API_INRIA_KEY"]
CELLS = [(0, 0), (2, 0), (0, 2), (2, 2)]                      # (spurious idx, dropped idx): 0/2 spurious, 0%/50% dropped
spent = 0.0

def cost(inp, cached, out, cache_write=0):
    return ((inp - cached) * PRICE["write"] + cached * PRICE["read"] + out * PRICE["out"]) / 1e6   # conservative: uncached billed as cache write

W = World(); NAMES = disease_names(); LABELS = term_labels()
cases = load_cases(W.replaced)
split = json.load(open("results/hpo_split.json")); test = set(split["test"])
ch = json.load(open("results/hpo_chosen.json"))["exact"]
real = [c for c in cases if c[0] in test]
rng = random.Random(0)
by = {}
for pid, (d, obs) in enumerate(real):
    if d in W.di:
        by.setdefault(d, []).append(pid)
dis = sorted(by); rng.shuffle(dis)
pids = [rng.choice(by[d]) for d in dis[:60]]
rng.shuffle(pids)
plan = [(pid, CELLS[i % 4]) for i, pid in enumerate(pids)][:N] if N <= 60 else None
json.dump(plan, open(RESULTS / "hpo_phase2_plan.json", "w"))
phase1 = {}
for l in open("results/hpo_phase1.jsonl"):
    r = json.loads(l)
    if r["set"] == "real":
        phase1[(r["pid"], r["spur"], r["drop"])] = r

cfg = load_config(str(Path(__file__).parent / "llm.yaml")); a = cfg.assistant
a.strategy, a.few_shot_k, a.max_steps = "react", 0, 10
a.verify_rounds, a.comment_rules, a.query_timeout, a.temperature = 0, False, 120, 0.0
provider = cfg.provider("anthropic-inria")

DESCRIPTION = ("One patient, described by observed phenotypes, and a probabilistic model of which rare "
               "disease they have among about 12,900 candidates.")
GUIDANCE = """\
The engine holds two relations about ONE patient:
- Prior(disease): a probabilistic choice, exactly one disease is true and all are equally likely a priori.
- Evidence(disease, term): independent probabilistic facts. Each row says that a patient with that disease
  would show the phenotype `term` with a probability (not visible as a column). It lists every disease for
  each of the patient's observed phenotypes; no other terms are present.
Diseases are strings such as "OMIM:619340", terms such as "HP:0001250".

Probabilities: put the PROB marker in a rule head, e.g. `ans(x, PROB) :- R(x) & S(x)` gives, for each x, the
probability that R(x) and S(x) hold. A conditional probability uses `//`:
`ans(x, PROB) :- (R(x) & S(x)) // (T(y))` is the probability of the left part given the right part; variables
of the right part that are not in the head are summed over.
Your final query must end with a rule whose head is ans(...), first column the disease and last the probability.
"""
COLUMNS = {"Prior": ["disease"], "Evidence": ["disease", "term"]}

def question(obs):
    lines = "; ".join(f'{t} ({LABELS.get(t, "?")})' for t in obs)
    return ("A patient shows these phenotypes: " + lines + ". Which diseases are most likely? "
            "Rank all candidate diseases by their posterior probability given these phenotypes.")

class Alarm(Exception): pass
def _h(*_): raise Alarm()
signal.signal(signal.SIGALRM, _h)

def agent_arm(obs, truth):
    rec = {}
    nl = make_nl()
    M = W.probs(obs, ch)
    rows = [(float(M[j, i]), W.diseases[i], t) for j, t in enumerate(obs) for i in range(W.n)]
    nl.add_probabilistic_facts_from_tuples(rows, name="Evidence")
    nl.add_uniform_probabilistic_choice_over_set([(d,) for d in W.diseases], name="Prior")
    t0 = time.time()
    try:
        agent = ReactAgent(nl, provider, a, "datalog", engine_name="", loaded=COLUMNS,
                           description=DESCRIPTION, guidance=GUIDANCE, examples=[])
        res = agent.run(question(obs))
        rec.update(query=res.query, valid=res.valid, out_of_steps=res.out_of_steps, calls=res.calls,
                   usage=dict(res.usage), error=res.error)
        global spent
        u = res.usage; spent += cost(u.get("input", 0), u.get("cached", 0), u.get("output", 0))
        if res.valid:
            signal.alarm(180)
            try:
                with nl.scope:
                    out = list(nl.execute_datalog_program(res.query))
            finally:
                signal.alarm(0)
            rec["n_rows"] = len(out)
            if out and isinstance(out[0][-1], float) and len(out[0]) >= 2:
                sc = np.zeros(W.n); 
                for r in out:
                    i = W.di.get(str(r[0]))
                    if i is not None: sc[i] = max(sc[i], r[-1])
                i = W.di[truth]
                rec["rank"] = rank_of(sc, i)
                ref = posterior(W, obs, ch)
                rec["ref_rank"] = rank_of(ref, i)
                rec["rank_equals_ref"] = rec["rank"] == rec["ref_rank"]
                nz = sc > 0
                rec["max_abs_diff_vs_posterior"] = float(np.abs(sc - ref).max()) if nz.sum() == W.n else None
            else:
                rec["rank"] = None; rec["no_probability_column"] = True
    except Exception as e:
        rec["exception"] = f"{type(e).__name__}: {str(e)[:200]}"; rec["rank"] = None
    rec["seconds"] = round(time.time() - t0, 1)
    return rec

ALLNORM = {norm(n): d for d, n in NAMES.items()}
def direct_arm(obs, truth):
    global spent
    lines = "\n".join(f'{t}: {LABELS.get(t, "?")}' for t in obs)
    prompt = ("A patient has these Human Phenotype Ontology terms:\n" + lines +
              "\n\nWhich rare diseases are most likely? Reply with a JSON list of at most 10 objects, most likely first, "
              'each {"name": "<disease name as in OMIM or Orphanet>", "omim": "<OMIM:number or empty>"}. No other text.')
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    t0 = time.time(); rec = {}
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
        text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
        u = d["usage"]; rec["usage"] = u
        spent += cost(u["input_tokens"] + u.get("cache_read_input_tokens", 0), u.get("cache_read_input_tokens", 0), u["output_tokens"])
        items = json.loads(text[text.index("["):text.rindex("]") + 1])
        rec["n_answers"] = len(items)
        tn = norm(NAMES[truth]); strict = lenient = None
        for k, it in enumerate(items, 1):
            nm, om = norm(str(it.get("name", ""))), str(it.get("omim", "")).strip()
            hit = (om == truth) or (nm == tn)
            if hit and strict is None: strict = k
            if lenient is None and (hit or (nm and difflib.SequenceMatcher(None, nm, tn).ratio() >= 0.85)):
                lenient = k
        rec["rank_strict"], rec["rank_lenient"] = strict, lenient
        rec["unmatched_names"] = sum(1 for it in items if norm(str(it.get("name", ""))) not in ALLNORM and str(it.get("omim", "")).strip() not in W.di)
        rec["answers"] = items[:10]
    except Exception as e:
        rec["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec

out = open(OUT, "a")
for n, (pid, (si, di_)) in enumerate(plan):
    truth, base = real[pid]
    obs = perturb(W, base, SPUR_LEVELS[si], DROP_LEVELS[di_], random.Random(f"real/{pid}/{si}/{di_}"))
    p1 = phase1[(pid, SPUR_LEVELS[si], DROP_LEVELS[di_])]
    rec = dict(pid=pid, disease=truth, spur=SPUR_LEVELS[si], drop=DROP_LEVELS[di_], n_obs=len(obs),
               phase1=dict(exact_leak=p1["exact_leak"]["rank"], overlap=p1["overlap"]["rank"]))
    rec["agent"] = agent_arm(obs, truth)
    rec["direct"] = direct_arm(obs, truth)
    rec["spent_usd_cumulative"] = round(spent, 3)
    out.write(json.dumps(rec, default=str) + "\n"); out.flush()
    print(n + 1, "/", len(plan), "agent rank", rec["agent"].get("rank"), "(ref", rec["agent"].get("ref_rank"), ") direct", rec["direct"].get("rank_strict"),
          "/", rec["direct"].get("rank_lenient"), "| p1 exact_leak", rec["phase1"]["exact_leak"], "| spent $%.2f" % spent, flush=True)
    if spent > CAP_USD:
        print("SPEND CAP REACHED, stopping"); break
