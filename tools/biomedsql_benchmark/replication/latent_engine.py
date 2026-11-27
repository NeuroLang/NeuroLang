"""Part 2 of PROTOCOL_LATENT.md: the latent-effect model on every ladder
item, as a NeuroLang program solved by the lifted solvers only, and in closed
form.

    python latent_engine.py --workers 4   # all rungs
    python latent_engine.py probe          # record the refusals

One program per item, with the parameters of the outer fold containing the
item's locus (results/latent_params.json). The formulations are described at
"NeuroLang programs" below. A rung is labelled "engine" when the lifted
solvers solved every item of it within the per-item time limit and matched
the closed form to 1e-9; otherwise its values are the closed form and the
number solved, the errors and a program are recorded.
"""

import json
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import latent_fit as lf
from ladder_engine import new_engine

HERE = Path(__file__).resolve().parent
TIMEOUT = {"Q4": 60}   # seconds per item; 300 for the other rungs
DEFAULT_TIMEOUT = 300
Q4_MAX_EVIDENCE = 5
RUNGS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


# ---------------- closed forms ----------------
def post_real(par, ev):
    return lf.posterior_real(par["rho"], par["beta"], par["alpha"], ev)


def state_probs(par, g):
    """(P(sig | real), P(sig | null)) of a dataset in organ group g."""
    return par["alpha"][g], par["beta"]


def mix(post, fr, fn):
    return post * fr + (1 - post) * fn


def closed_q14(par, item):
    ev = [(lf.group(e["organ"]), e["significant"]) for e in item["evidence"]]
    post = post_real(par, ev)
    H = [state_probs(par, lf.group(h["organ"])) for h in item["heldout"]]
    out = {}
    for name, f in (
            ("Q1", lambda p: p[0]),
            ("Q2", lambda p: 1 - np.prod([1 - x for x in p])),
            ("Q3", lambda p: np.prod(p)),
            ("Q4", lambda p: p[0] * p[1] + p[0] * p[2] + p[1] * p[2]
             - 2 * p[0] * p[1] * p[2])):
        out[name] = mix(post, f([h[0] for h in H]), f([h[1] for h in H]))
    return out


def closed_q5(par, item):
    ev = [(lf.group(e["organ"]), e["significant"]) for e in item["evidence"]]
    post = post_real(par, ev)
    vals = []
    for k in (0, 1):
        ps = {"brain": [], "blood": []}
        for h in item["heldout"]:
            if h["organ"] in ps:
                ps[h["organ"]].append(state_probs(par, h["organ"])[k])
        vals.append((1 - np.prod([1 - p for p in ps["brain"]]))
                    * np.prod([1 - p for p in ps["blood"]]))
    return {"Q5": mix(post, *vals)}


def closed_q6(par, item):
    posts = {d: post_real(par, [(lf.group(e["organ"]), e["significant"])
                                for e in item["evidence"][d]])
             for d in ("PD", "AD")}
    tot = 0.0
    for kp, wp in ((0, posts["PD"]), (1, 1 - posts["PD"])):
        for ka, wa in ((0, posts["AD"]), (1, 1 - posts["AD"])):
            ps = [state_probs(par, lf.group(h["organ"])) for h in item["heldout"]]
            tot += wp * wa * (1 - np.prod([1 - p[kp] * p[ka] for p in ps]))
    return {"Q6": tot}


# ---------------- NeuroLang programs ----------------
# Two formulations of the same model, chosen per rung (see the report):
# (sig) one relation of datasets, constants for the datasets:
#   state(s)          probabilistic choice {real: rho, null: 1 - rho}
#   out(s, q)         P(q significant | s): alpha_o if real, beta if null
#   nout(s, q)        1 - out(s, q), evidence datasets only
#   sig(q)  :- state(s), out(s, q)     nsig(q) :- state(s), nout(s, q)
#   e() :- sig('e0'), nsig('e1'), ...  the evidence
#   q() :- <formula over sig('h0'), ...>;   ans(PROB) :- q() // e()
#   Q1, Q2 (q() :- inh(q), sig(q)), Q3, Q5, Q6.  Negated evidence (~sig) is
#   refused (NotUnateException / NonLiftable), hence nsig.
# (rel) one relation set per dataset (no self-join), used for Q4 only: the
#   union of three self-joins of sig(.) took 25-50 s on one evidence atom.

def facts(nl, par, state, out, nout, held, ev):
    nl.add_probabilistic_choice_from_tuples(
        [(par["rho"], "real"), (1 - par["rho"], "null")], name=state)
    rows, nrows = [], []
    for q, g in held + ev:
        a, b = state_probs(par, g)
        rows += [(a, "real", q), (b, "null", q)]
    for q, g in ev:
        a, b = state_probs(par, g)
        nrows += [(1 - a, "real", q), (1 - b, "null", q)]
    nl.add_probabilistic_facts_from_tuples(rows, name=out)
    if nrows:
        nl.add_probabilistic_facts_from_tuples(nrows, name=nout)


def evidence_rule(ev, sig, nsig, pre="e"):
    return [f"{sig if s else nsig}('{pre}{i}')" for i, (_, s) in enumerate(ev)]


def latent_programs_sig(par, item, rung):
    if rung == "Q6":
        ev = {d: [(lf.group(e["organ"]), e["significant"])
                  for e in item["evidence"][d]] for d in ("PD", "AD")}
        held = [(f"h{i}", lf.group(h["organ"]))
                for i, h in enumerate(item["heldout"])]
        evq = {d: [(f"e{i}", g) for i, (g, _) in enumerate(ev[d])]
               for d in ev}

        def setup(nl):
            for d in ("PD", "AD"):
                facts(nl, par, f"state{d}", f"out{d}", f"nout{d}", held,
                      evq[d])
            nl.add_tuple_set([(q,) for q, _ in held], name="inh")
        lines = []
        for d in ("PD", "AD"):
            lines += [f"s{d}(q) :- state{d}(s), out{d}(s, q)",
                      f"ns{d}(q) :- state{d}(s), nout{d}(s, q)"]
        body = (evidence_rule(ev["PD"], "sPD", "nsPD")
                + evidence_rule(ev["AD"], "sAD", "nsAD"))
        lines += ["e() :- " + ", ".join(body),
                  "q() :- inh(q), sPD(q), sAD(q)",
                  "ans(PROB) :- q() // e()"]
        return setup, "\n".join(lines)
    ev = [(lf.group(e["organ"]), e["significant"]) for e in item["evidence"]]
    held = [(f"h{i}", lf.group(h["organ"]))
            for i, h in enumerate(item["heldout"])]
    evq = [(f"e{i}", g) for i, (g, _) in enumerate(ev)]
    lines = ["sig(q) :- state(s), out(s, q)",
             "nsig(q) :- state(s), nout(s, q)",
             "e() :- " + ", ".join(evidence_rule(ev, "sig", "nsig"))]

    def setup(nl):
        facts(nl, par, "state", "out", "nout", held, evq)
        nl.add_tuple_set([(q,) for q, _ in held], name="inh")
        nl.add_tuple_set([(q,) for q, g in held if g == "brain"], name="inbrain")
        nl.add_tuple_set([(q,) for q, g in held if g == "blood"], name="inblood")
    if rung == "Q1":
        lines += ["q() :- sig('h0')"]
    elif rung == "Q2":
        lines += ["q() :- inh(q), sig(q)"]
    elif rung == "Q3":
        lines += ["q() :- sig('h0'), sig('h1'), sig('h2')"]
    elif rung == "Q5":
        lines += ["br() :- inbrain(q), sig(q)", "bl() :- inblood(q), sig(q)",
                  "q() :- br(), ~bl()"]
    lines += ["ans(PROB) :- q() // e()"]
    return setup, "\n".join(lines)


def latent_programs(par, item, rung):
    return (latent_programs_rel if rung == "Q4" else latent_programs_sig)(
        par, item, rung)


# ---- the (rel) formulation ----
def add_dataset(nl, par, name, g, evidence_sig):
    """Facts of one dataset; evidence_sig None for a held-out dataset."""
    a, b = state_probs(par, g)
    if evidence_sig is False:
        nl.add_probabilistic_facts_from_tuples(
            [(1 - a, "real"), (1 - b, "null")], name=f"nout_{name}")
    else:
        nl.add_probabilistic_facts_from_tuples(
            [(a, "real"), (b, "null")], name=f"out_{name}")


def latent_programs_rel(par, item, rung):
    """-> (setup, program text); non-significant evidence uses the
    complementary fact, named sig_<d> for uniformity of the rules."""
    states = ["state"] if rung != "Q6" else ["statePD", "stateAD"]
    held = [(f"h{i}", lf.group(h["organ"]))
            for i, h in enumerate(item["heldout"])]
    lines, plan = [], []   # plan: (name, group, sig or None)
    if rung == "Q6":
        for d in ("PD", "AD"):
            for i, e in enumerate(item["evidence"][d]):
                plan.append((f"{d}e{i}", f"state{d}", lf.group(e["organ"]),
                             e["significant"]))
            for i, (q, g) in enumerate(held):
                plan.append((f"{d}{q}", f"state{d}", g, None))
    else:
        for i, e in enumerate(item["evidence"]):
            plan.append((f"e{i}", "state", lf.group(e["organ"]),
                         e["significant"]))
        for q, g in held:
            plan.append((q, "state", g, None))
    ev_atoms = [f"sig_{n}()" for n, st, g, sg in plan if sg is not None]

    def setup(nl):
        for st in states:
            nl.add_probabilistic_choice_from_tuples(
                [(par["rho"], "real"), (1 - par["rho"], "null")], name=st)
        for n, st, g, sg in plan:
            add_dataset(nl, par, n, g, sg)
    for n, st, g, sg in plan:
        rel = "nout" if sg is False else "out"
        lines.append(f"sig_{n}() :- {st}(s), {rel}_{n}(s)")
    lines.append("e() :- " + ", ".join(ev_atoms))
    H = [f"sig_h{i}()" for i in range(len(held))]
    if rung == "Q1":
        lines.append(f"q() :- {H[0]}")
    elif rung == "Q2":
        lines += [f"q() :- {h}" for h in H]
    elif rung == "Q3":
        lines.append("q() :- " + ", ".join(H))
    elif rung == "Q4":
        lines += [f"q() :- {H[a]}, {H[b]}" for a, b in ((0, 1), (0, 2), (1, 2))]
    elif rung == "Q5":
        for i, (q, g) in enumerate(held):
            if g in ("brain", "blood"):
                lines.append(f"{g[:2]}() :- {H[i]}")
        lines.append("q() :- br(), ~bl()")
    elif rung == "Q6":
        lines += [f"q() :- sig_PDh{i}(), sig_ADh{i}()"
                  for i in range(len(held))]
    lines.append("ans(PROB) :- q() // e()")
    return setup, "\n".join(lines)


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout("no answer within the per-item time limit "
                  f"({TIMEOUT}, default {DEFAULT_TIMEOUT} s)")


def run_engine(task):
    rung, item, par = task
    setup, program = latent_programs(par, item, rung)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT.get(rung, DEFAULT_TIMEOUT))
    t = time.time()
    try:
        nl = new_engine()
        setup(nl)
        with nl.environment:
            res = nl.execute_datalog_program(program)
        df = res.as_pandas_dataframe()
        val = float(df.values[0][0]) if len(df) else 0.0
        return item["id"], rung, val, None, program, time.time() - t
    except BaseException as exc:
        return (item["id"], rung, None,
                f"{type(exc).__name__}: {str(exc)[:400]}", program,
                time.time() - t)
    finally:
        signal.alarm(0)


def probe():
    """Record, in results/latent_ladder.json, the refusals of the lifted
    solvers on the two forms the protocol suggests first."""
    P = json.load(open(HERE / "results/latent_params.json"))["0"]
    out = {}
    base = ("sig(q) :- state(s), out(s, q)\n"
            "nsig(q) :- state(s), nout(s, q)\n")
    cases = {
        "negated evidence ~sig (single-rule latent)":
            ("sig(q) :- state(s), out(s, q)\n"
             "e() :- sig('e0'), ~sig('e1')\nq() :- sig('h0')\n"
             "ans(PROB) :- q() // e()"),
        "two-rule latent, sig('h0') // sig('e0')":
            ("sig(q) :- state('real'), det(q)\n"
             "sig(q) :- state('null'), fp(q)\n"
             "ans(PROB) :- sig('h0') // sig('e0')"),
    }

    def setup(nl):
        nl.add_probabilistic_choice_from_tuples(
            [(P["rho"], "real"), (1 - P["rho"], "null")], name="state")
        a = P["alpha"]["brain"]
        rows = [(a, "real", q) for q in ("h0", "e0", "e1")]
        rows += [(P["beta"], "null", q) for q in ("h0", "e0", "e1")]
        nl.add_probabilistic_facts_from_tuples(rows, name="out")
        nl.add_probabilistic_facts_from_tuples(
            [(a, q) for q in ("h0", "e0", "e1")], name="det")
        nl.add_probabilistic_facts_from_tuples(
            [(P["beta"], q) for q in ("h0", "e0", "e1")], name="fp")
    for name, prog in cases.items():
        try:
            nl = new_engine()
            setup(nl)
            with nl.environment:
                res = nl.execute_datalog_program(prog)
            out[name] = {"program": prog, "result": res.as_pandas_dataframe()
                         .values.tolist()}
        except BaseException as exc:
            out[name] = {"program": prog,
                         "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        print(name, "->", out[name].get("error", out[name].get("result"))[:300]
              if isinstance(out[name].get("error", out[name].get("result")), str)
              else out[name])
    f = HERE / "results/latent_ladder.json"
    data = json.load(open(f))
    data["refusal_probes"] = out
    json.dump(data, open(f, "w"), indent=1)


def main(only=None, workers=4, limit=None):
    if only == "probe":
        return probe()
    d = json.load(open(HERE / "results/ladder_items_signal.json"))
    P = json.load(open(HERE / "results/latent_params.json"))
    smr, allp = lf.load_items()
    genes = {i["gene"] for k in d for i in d[k]}
    gf, unmapped, _, _ = lf.gene_folds(smr, allp, genes)
    print("ladder loci not among the 65:", unmapped or "none")
    assert not unmapped

    def par_of(item):
        return P[str(gf[item["gene"]][1])]
    if limit:   # smoke test: the first `limit` items of each sample
        d = {k: v[:limit] for k, v in d.items()}
    items = {"Q1": d["q14"], "Q2": d["q14"], "Q3": d["q14"], "Q4": d["q14"],
             "Q5": d["q5"], "Q6": d["q6"]}
    closed_fn = {"q14": closed_q14, "q5": closed_q5, "q6": closed_q6}
    values = {}
    for key, fn in closed_fn.items():
        for it in d[key]:
            for q, v in fn(par_of(it), it).items():
                values.setdefault(it["id"], {})[q] = {
                    "closed": float(v), "engine": None,
                    "fold": gf[it["gene"]][1]}
    # Q4 is attempted only on items with at most Q4_MAX_EVIDENCE evidence
    # datasets: measured 1.5 s (1 dataset), 3.5 s (2), 19 s (4), over 50 s
    # at 6, roughly doubling per dataset.
    tasks = [(q, it, par_of(it)) for q in RUNGS if only in (None, q)
             for it in items[q]
             if q != "Q4" or len(it["evidence"]) <= Q4_MAX_EVIDENCE]
    cache_f = HERE / ("results/latent_engine_cache.jsonl" if not limit
                      else "results/latent_engine_cache_smoke.jsonl")
    cache = {}
    if cache_f.exists():
        for line in open(cache_f):
            r = json.loads(line)
            cache[(r[0], r[1])] = r
    errors, programs, times = {}, {}, {}

    def record(r):
        i, q, val, err, prog, dt = r
        values[i][q]["engine"] = val
        times.setdefault(q, []).append(dt)
        if err:
            errors.setdefault(q, []).append((i, err))
            programs.setdefault(q, prog)
    todo = []
    for t in tasks:
        r = cache.get((t[1]["id"], t[0]))
        record(r) if r else todo.append(t)
    pool = Pool(workers) if workers > 1 else None
    with open(cache_f, "a") as cf:
        results = (pool.imap_unordered(run_engine, todo, chunksize=1)
                   if pool else map(run_engine, todo))
        for n, r in enumerate(results):
            cf.write(json.dumps(r) + "\n")
            cf.flush()
            record(r)
            if n % 50 == 0:
                print(n, len(todo), flush=True)
    if pool:
        pool.close()
    maxdiff, source = {}, {}
    for q in RUNGS:
        if only not in (None, q):
            continue
        ids = [it["id"] for it in items[q]]
        ok = [i for i in ids if values[i][q]["engine"] is not None]
        maxdiff[q] = (max(abs(values[i][q]["engine"] - values[i][q]["closed"])
                          for i in ok) if ok else None)
        full = len(ok) == len(ids) and maxdiff[q] <= 1e-9
        source[q] = "engine" if full else "closed form"
        print(f"{q}: solved {len(ok)}/{len(ids)}; max |engine - closed| = "
              f"{maxdiff[q]}; source: {source[q]}; mean time "
              f"{np.mean(times.get(q, [0])):.1f}s; errors {len(errors.get(q, []))}")
        for i in ids:
            v = values[i][q]
            v["latent"] = v["engine"] if source[q] == "engine" else v["closed"]
            v["source"] = source[q]
    out = HERE / ("results/latent_ladder.json" if only is None and not limit
                  else "results/latent_ladder_partial.json")
    json.dump({"values": values, "max_diff": maxdiff, "source": source,
               "n_solved": {q: sum(values[it["id"]][q]["engine"] is not None
                                   for it in items[q])
                            for q in RUNGS if only in (None, q)},
               "errors": {q: {"n": len(e), "first": e[:3],
                              "kinds": sorted({x[1].split(":")[0] for x in e})}
                          for q, e in errors.items()},
               "programs": programs}, open(out, "w"), indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", help="a rung (Q1..Q6) or 'probe'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: first N items per sample")
    a = ap.parse_args()
    main(a.only, a.workers, a.limit)
