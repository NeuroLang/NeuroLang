"""The engine's probability for every ladder question (PROTOCOL_LADDER.md).

    python ladder_engine.py [ITEMS.json [OUT.json]]

Held-out outcomes are independent probabilistic facts whose probability is
support_prior (k + 0.01) / (n + 0.01 + 0.5) of the evidence; the rung's
formula is a NeuroLang query over them, solved by the lifted solvers only
(small dichotomy theorem, Dalvi-Suciu); a refusal is recorded, never
replaced by weighted model counting. Each value is checked against its
closed form to 1e-9.

Facts (items are batched; the item is the head variable):
  held_sig(q)          q = "item#dataset", one fact per held-out dataset
  held1(i) held2(i) held3(i)   the i-th held-out dataset of item i
                       (Q1, Q3, Q4: the three named atoms of the formula)
  held_pd(q) held_ad(q)        Q6, two independent facts per dataset
  in_h(i, q), brain(q), blood(q)   deterministic
Queries:
  Q1 ans(i, PROB) :- held1(i)
  Q2 ans(i, PROB) :- in_h(i, q), held_sig(q)
  Q3 ans(i, PROB) :- held1(i), held2(i), held3(i)
  Q4 u(i) :- held_a(i), held_b(i) for the three pairs; ans(i, PROB) :- u(i)
  Q5 br(i) :- in_h(i, q), brain(q), held_sig(q); bl(i) likewise for blood;
     ans(i, PROB) :- br(i), ~bl(i)
  Q6 ans(i, PROB) :- in_h(i, q), held_pd(q), held_ad(q)
Q3 in the form held_sig(a), held_sig(b), held_sig(c) with a, b, c bound by
a table of the held-out triple is refused as non-liftable (self-join with
no separator), hence the named atoms.
"""

import json
import sys
from math import prod
from pathlib import Path

from neurolang.frontend import NeurolangPDL
from neurolang.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver as sdt,
)

HERE = Path(__file__).resolve().parent
A, B = 0.01, 0.5


def prior(k, n):
    return (k + A) / (n + A + B)


def prior_of(rows):
    return prior(sum(r["significant"] for r in rows), len(rows))


def new_engine():
    nl = NeurolangPDL()
    nl.probabilistic_solvers = (
        sdt.solve_succ_query, dalvi_suciu_lift.solve_succ_query)
    nl.probabilistic_marg_solvers = (
        sdt.solve_marg_query, dalvi_suciu_lift.solve_marg_query)
    return nl


def solve(program, setup):
    nl = new_engine()
    setup(nl)
    with nl.environment:
        res = nl.execute_datalog_program(program)
    return {i: p for i, p in res.as_pandas_dataframe().itertuples(
        index=False)}


def positional(nl, p3):
    """held1..held3: probability of the i-th held-out dataset of each item."""
    for k in range(3):
        nl.add_probabilistic_facts_from_tuples(
            [(p[k], i) for i, p in p3.items()], name=f"held{k + 1}")


def q14(items):
    p1 = {i["id"]: prior_of(i["evidence"]) for i in items}
    p3 = {i: [p] * 3 for i, p in p1.items()}
    closed = {}
    for i, p in p1.items():
        closed[i] = {"Q1": p, "Q2": 1 - (1 - p) ** 3, "Q3": p ** 3,
                     "Q4": 3 * p * p - 2 * p ** 3}
    rows = [(p, f"{i['id']}#{h['dataset']}") for i in items
            for h in i["heldout"] for p in [p1[i["id"]]]]
    in_h = [(i["id"], f"{i['id']}#{h['dataset']}") for i in items
            for h in i["heldout"]]

    def setup(nl):
        positional(nl, p3)
        nl.add_probabilistic_facts_from_tuples(rows, name="held_sig")
        nl.add_tuple_set(in_h, name="in_h")

    programs = {
        "Q1": "ans(i, PROB) :- held1(i)",
        "Q2": "ans(i, PROB) :- in_h(i, q), held_sig(q)",
        "Q3": "ans(i, PROB) :- held1(i), held2(i), held3(i)",
        "Q4": ("u(i) :- held1(i), held2(i)\nu(i) :- held1(i), held3(i)\n"
               "u(i) :- held2(i), held3(i)\nans(i, PROB) :- u(i)"),
    }
    return programs, setup, closed


def q5(items):
    rows, in_h, brain, blood, closed = [], [], [], [], {}
    for it in items:
        by_organ = {}
        for e in it["evidence"]:
            by_organ.setdefault(e["organ"], []).append(e)
        ps = {"brain": [], "blood": []}
        for h in it["heldout"]:
            q = f"{it['id']}#{h['dataset']}"
            if h["organ"] not in ps:
                continue            # neither brain nor blood: no atom
            p = prior_of(by_organ.get(h["organ"], []))
            rows.append((p, q))
            in_h.append((it["id"], q))
            (brain if h["organ"] == "brain" else blood).append((q,))
            ps[h["organ"]].append(p)
        closed[it["id"]] = {"Q5": (1 - prod(1 - p for p in ps["brain"]))
                            * prod(1 - p for p in ps["blood"])}

    def setup(nl):
        nl.add_probabilistic_facts_from_tuples(rows, name="held_sig")
        nl.add_tuple_set(in_h, name="in_h")
        nl.add_tuple_set(brain, name="brain")
        nl.add_tuple_set(blood, name="blood")

    program = ("br(i) :- in_h(i, q), brain(q), held_sig(q)\n"
               "bl(i) :- in_h(i, q), blood(q), held_sig(q)\n"
               "ans(i, PROB) :- br(i), ~bl(i)")
    return {"Q5": program}, setup, closed


def q6(items):
    rows_pd, rows_ad, in_h, closed = [], [], [], {}
    for it in items:
        ppd, pad = prior_of(it["evidence"]["PD"]), prior_of(
            it["evidence"]["AD"])
        for h in it["heldout"]:
            q = f"{it['id']}#{h['dataset']}"
            rows_pd.append((ppd, q))
            rows_ad.append((pad, q))
            in_h.append((it["id"], q))
        closed[it["id"]] = {"Q6": 1 - prod(1 - ppd * pad for _ in it["heldout"])}

    def setup(nl):
        nl.add_probabilistic_facts_from_tuples(rows_pd, name="held_pd")
        nl.add_probabilistic_facts_from_tuples(rows_ad, name="held_ad")
        nl.add_tuple_set(in_h, name="in_h")

    return {"Q6": "ans(i, PROB) :- in_h(i, q), held_pd(q), held_ad(q)"}, \
        setup, closed


def main(items_file="results/ladder_items.json",
         out_file="results/ladder_engine.json"):
    data = json.load(open(HERE / items_file))
    values, refused, maxdiff = {}, {}, {}
    for key, build in (("q14", q14), ("q5", q5), ("q6", q6)):
        programs, setup, closed = build(data[key])
        for q, program in programs.items():
            try:
                got = solve(program, setup)
            except Exception as exc:
                refused[q] = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(q, "REFUSED by the lifted solvers:", refused[q])
                continue
            # an item missing from the answer has probability 0
            diffs = [abs(got.get(i, 0.0) - c[q]) for i, c in closed.items()]
            maxdiff[q] = max(diffs)
            for i, c in closed.items():
                values.setdefault(i, {})[q] = {
                    "engine": float(got.get(i, 0.0)), "closed": c[q]}
            print(f"{q}: {len(closed)} items, max |engine - closed form| = "
                  f"{maxdiff[q]:.3g}; missing from the answer: "
                  f"{sum(i not in got for i in closed)}")
    (HERE / out_file).write_text(json.dumps(
        {"values": values, "max_diff": maxdiff, "refused": refused},
        indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
