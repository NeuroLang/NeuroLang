"""Build check, not part of any arm: a hand-written Datalog query per template,
run on the S2 engine (lifted solvers only), against gold.json.

    uv run --no-sync python check_lifted.py [--small]

It checks that every template is solved by the lifted solvers (a refusal would
make its questions unanswerable for S2), that both arms' tables agree, and the
gold predicates. ``--small`` leaves out the five BiomedSQL tables (P7a, which
reads drug_indication, is then skipped). These queries are never shown to an
agent.

The templates the lifted solvers refuse (``engine_refused``: P4d, P5d, P6d)
cannot be checked and are skipped. Its run on the first 168 questions: 0
mismatches or refusals; P4e (added later) was checked on P4e-0 only.
"""

import json
import sys
import time

from engine import build
from gold import HERE

DI_COLUMNS = 23     # drug_indication: drugName is column 2, mesh_heading 15


def listed(m):
    args = [f"v{i}" for i in range(DI_COLUMNS)]
    args[2], args[15] = "x", f"'{m}'"
    return f"listed(x) :- drug_indication({', '.join(args)})\n"


SUM = "ans(s) :- AGGREGATE[](p(u, q) @ sum(q)) = s"
REFERENCE = {
    "P1a": "ans(PROB) :- test_evidence('{d}', '{g}', '{dis}')",
    "P1b": "ans(PROB) :- indication_success('{x}', '{m}')",
    "P1c": "ans(PROB) :- snp_causal('{s}', '{g}', '{dis}')",
    "P2a": "ans(PROB) :- test_evidence(d, '{g}', '{dis}') & dataset_organ(d, 'brain')",
    "P2b": "ans(PROB) :- indication_success('{x}', m)",
    "P2c": "ans(PROB) :- snp_causal(s, '{g}', '{dis}')",
    "P3a": "ans(PROB) :- drug_target('{x}', g) & test_evidence(d, g, '{dis}')",
    "P3b": "ans(PROB) :- drug_target('{x}', g) & snp_causal(s, g, '{dis}')",
    "P3c": "ans(PROB) :- test_evidence(d, '{g}', '{dis}') & dataset_reliable(d)",
    "P4a": "e(g) :- test_evidence(d, g, '{dis}')\n"
           "c(g) :- snp_causal(s, g, '{dis}')\n"
           "ans(PROB) :- drug_target('{x}', g) & e(g) & c(g)",
    "P4b": "ans(PROB) :- dataset_reliable(d) & drug_target('{x}', g) & "
           "test_evidence(d, g, '{dis}')",
    "P4c": "ans(PROB) :- dataset_reliable(d) & test_evidence(d, '{g1}', '{dis}')"
           " & test_evidence(d, '{g2}', '{dis}')",
    "P5a": "e() :- test_evidence(d, '{g}', '{dis}') & dataset_organ(d, 'brain')\n"
           "e() :- test_evidence(d, '{g}', '{dis}') & dataset_reliable(d)\n"
           "ans(PROB) :- e()",
    "P5b": "e() :- test_evidence(d, '{g1}', '{dis}') & dataset_reliable(d)\n"
           "e() :- test_evidence(d, '{g2}', '{dis}') & dataset_reliable(d)\n"
           "ans(PROB) :- e()",
    "P5c": "e() :- drug_target('{x}', g) & test_evidence(d, g, '{dis}') & "
           "dataset_reliable(d)\n"
           "e() :- drug_target('{x}', g) & test_evidence(d, g, '{dis}') & "
           "dataset_organ(d, 'blood')\n"
           "ans(PROB) :- e()",
    "P6a": "ans(PROB) :- (drug_target('{x}', g) & test_evidence(d, g, '{dis}') "
           "& dataset_reliable(d)) // (test_evidence(d2, '{g0}', '{dis}') & "
           "dataset_organ(d2, 'blood'))",
    "P6b": "b() :- drug_target('{x}', g) & snp_causal(s, g, '{dis}')\n"
           "ans(PROB) :- drug_target('{x}', g) & test_evidence(d, g, '{dis}') "
           "& ~b()",
    "P6c": "ans(PROB) :- (test_evidence(d, '{g}', '{dis}') & dataset_reliable(d))"
           " // (test_evidence(d2, '{g}', '{dis}') & dataset_organ(d2, 'brain'))",
    "P4e": "p(g) :- test_evidence(d, g, 'PD')\n"
           "a(g) :- test_evidence(d, g, 'AD')\n"
           "ans(PROB) :- drug_target('{x}', g) & p(g) & a(g)",
    "P7a": "{listed}p(u, PROB) :- listed(u) & drug_approved(u) & "
           "drug_target(u, g) & test_evidence(d, g, '{dis}')\n" + SUM,
    "P7b": "p(u, PROB) :- drug_target('{x}', u) & test_evidence(d, u, '{dis}') "
           "& dataset_reliable(d)\n" + SUM,
    "P7c": "p(u, PROB) :- drug_target('{x}', g) & test_evidence(u, g, '{dis}') "
           "& dataset_reliable(u)\n" + SUM,
}


def main(small=False):
    t = time.time()
    nl = build(with_biomedsql=not small)
    print(f"engine built in {time.time() - t:.0f}s", flush=True)
    questions = json.load(open(HERE / "questions.json"))
    gold = json.load(open(HERE / "gold.json"))
    bad = 0
    for q in questions:
        if (small and q["template"] == "P7a") or q.get("engine_refused"):
            continue
        p = q["params"]
        program = REFERENCE[q["template"]].format(
            listed=listed(p["m"]) if "m" in p else "", **p
        )
        t = time.time()
        try:
            with nl.scope:
                rows = [tuple(r) for r in nl.execute_datalog_program(program)]
            value = float(rows[0][-1]) if rows else 0.0
            ok = abs(value - gold[q["id"]]["gold"]) <= 1e-9 * max(1, value)
            msg = f"{value:.9f}"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {str(e)[:120]}"
        bad += not ok
        print(f"{q['id']}: {'ok' if ok else 'MISMATCH'} {msg} gold "
              f"{gold[q['id']]['gold']:.9f} {time.time() - t:.1f}s", flush=True)
    print(f"{bad} mismatches or refusals")


if __name__ == "__main__":
    main("--small" in sys.argv)
