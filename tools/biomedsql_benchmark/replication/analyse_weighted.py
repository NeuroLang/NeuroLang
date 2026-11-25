"""The evaluation of PROTOCOL_WEIGHTED.md.

    python analyse_weighted.py SMR.parquet PREDICTIONS.parquet DATA_DIR

Asks the engine (lifted solvers only) for ``support_graded`` and
``support_prior`` of every evaluation prediction, the held-out dataset left
out of the units, checks them against the closed forms, and writes
``results/graded_engine_values.json``; then the metrics of the protocol.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import CLIP, auc, metrics
from post_hoc_logit import fit_logistic

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.95e-6
B = 2000
RULES = {
    "support_graded": "unit_evidence",
    "support_prior": "unit_significant",
}


def engine_values(data_dir, items):
    """(graded, prior) of each item (heldout, gene, disease), by the engine."""
    from neurolang.probabilistic import (
        dalvi_suciu_lift,
        small_dichotomy_theorem_based_solver as sdt,
    )
    from neurolang_apps.utils import engine_registry

    nl = engine_registry.build_engine(
        "biomedsql", data_dir=Path(data_dir), use_cache=False
    )
    nl.probabilistic_solvers = (
        sdt.solve_succ_query, dalvi_suciu_lift.solve_succ_query
    )
    nl.probabilistic_marg_solvers = (
        sdt.solve_marg_query, dalvi_suciu_lift.solve_marg_query
    )
    out = {}
    for name, relation in RULES.items():
        start = time.time()
        with nl.scope:
            nl.add_tuple_set(items, name="item")
            res = nl.execute_datalog_program(
                "other(u, h, g, d) :- item(h, g, d), evidence_unit(u), "
                "(u != h)\n"
                f"sp(h, g, d, PROB) :- ({relation}(u, g, d) & selected_unit(u)"
                " & other(u, h, g, d)) // (unit_tested(u, g, d) & "
                "selected_unit(u) & other(u, h, g, d))\n"
                "ans(h, g, d, p) :- sp(h, g, d, p)"
            )
            df = res.as_pandas_dataframe()
        out[name] = {
            f"{r.g}|{r.d}|{r.h}": r.p for r in df.itertuples(index=False)
        }
        print(f"engine {name}: {len(df)} values in "
              f"{time.time() - start:.1f} s")
    return out


def closed_forms(tests, rows, strength):
    from neurolang_apps.utils.engines.biomedsql.init import (
        strength_probability,
    )

    tests = tests.assign(
        sig=tests.p_SMR_multi < THRESHOLD,
        pi=strength_probability(tests.p_SMR_multi, strength),
    )
    tot = tests.groupby(["Gene", "Disease"])[["sig", "pi"]].agg("sum")
    tot["n"] = tests.groupby(["Gene", "Disease"]).size()
    held = tests.set_index(["Omic", "Gene", "Disease"])
    r = rows.join(tot, on=["gene", "disease"])
    h = held.loc[list(zip(r.heldout, r.gene, r.disease))]
    n = r.n.values - 1
    k = r.sig.values - h.sig.values
    s = r.pi.values - h.pi.values
    return (s + 0.01) / (n + 0.51), (k + 0.01) / (n + 0.51), n


def main(smr_path, predictions_path, data_dir):
    import yaml

    conf = yaml.safe_load(open(
        HERE.parents[2] / "packages/neurolang_apps/neurolang_apps/utils/"
        "engines/engines.yaml"
    ))["engines"]["biomedsql"]["python_init_options"]
    assert (conf["prior_significant"], conf["prior_none"]) == (0.01, 0.5)
    strength = conf["evidence_strength"]

    smr = pd.read_parquet(
        smr_path, columns=["Omic", "Gene", "Disease", "p_SMR_multi"]
    )
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True)
    tests = tests.p_SMR_multi.min().reset_index()
    preds = pd.read_parquet(predictions_path)
    fit_genes = {i["gene"] for i in json.load(open(HERE / "items.json"))}
    eval_genes = {i["gene"] for i in json.load(open(HERE / "items_eval.json"))}
    sets = {
        "primary": preds[preds.gene.isin(eval_genes) & (preds.p > 0)],
        "secondary": preds[preds.gene.isin(eval_genes) & (preds.p == 0)],
        "false alarms": preds[~preds.gene.isin(fit_genes | eval_genes)],
    }
    sets = {k: v.reset_index(drop=True) for k, v in sets.items()}

    asked = pd.concat([sets["primary"], sets["secondary"]])
    values_path = HERE / "results/graded_engine_values.json"
    if values_path.exists():
        engine = json.load(open(values_path))
    else:
        engine = engine_values(data_dir, list(
            asked[["heldout", "gene", "disease"]].itertuples(
                index=False, name=None)
        ))
        json.dump(engine, open(values_path, "w"))

    answers = {}
    for line in open(HERE / "results/eval_answers.jsonl"):
        r = json.loads(line)
        answers[r["id"]] = r
    logit_ids, _, logit_p, *_ = fit_logistic(smr_path, predictions_path)
    logit = dict(zip(logit_ids, logit_p))

    for label, rows in sets.items():
        y = rows.y.values.astype(float)
        graded, prior, n = closed_forms(tests, rows, strength)
        print(f"\n== {label}: {len(rows)} predictions, "
              f"{rows.gene.nunique()} genes, base rate {y.mean():.4f}")
        if label == "false alarms":
            for name, v in (("support_graded", graded),
                            ("support_prior", prior)):
                print(f"  {name:16} (closed form) mean {v.mean():.4f} "
                      f"max {v.max():.4f} Brier {np.mean((v - y) ** 2):.5f}")
            continue
        ids = [f"{r.gene}|{r.disease}|{r.heldout}"
               for r in rows.itertuples()]
        eg = np.array([engine["support_graded"][i] for i in ids])
        ep = np.array([engine["support_prior"][i] for i in ids])
        print(f"  engine - closed form, max |diff|: graded "
              f"{np.abs(eg - graded).max():.2e}, prior "
              f"{np.abs(ep - prior).max():.2e}")
        systems = {"support_graded": eg, "support_prior": ep}
        if label == "primary":
            assert np.allclose(
                ep, [answers[i]["support_prior"] for i in ids], atol=1e-9)
            systems["L2"] = np.array([answers[i]["L2"] for i in ids])
            systems["logistic"] = np.array([logit[i] for i in ids])
        groups = {}
        for j, g in enumerate(rows.gene.values):
            groups.setdefault(g, []).append(j)
        keys = sorted(groups)
        rng = random.Random(0)
        draws = [
            [j for g in (rng.choice(keys) for _ in keys) for j in groups[g]]
            for _ in range(B)
        ]
        print(f"  {'system':16} {'Brier':>6} {'95% CI':>15} {'logloss':>7} "
              f"{'ECE':>6} {'AUC':>6}")
        for name, v in systems.items():
            m = metrics(v, y)
            bs = [metrics(v[d], y[d])["brier"] for d in draws]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {name:16} {m['brier']:6.3f} [{lo:5.3f}, {hi:5.3f}] "
                  f"{m['logloss']:7.3f} {m['ece']:6.3f} {m['auc']:6.3f}")
        # metrics() clips to [0.02, 0.98], which flattens probabilities below
        # 0.02: the AUC without the clip ranks them
        print("  AUC unclipped: " + ", ".join(
            f"{name} {auc(np.asarray(v, float), y):.3f}"
            for name, v in systems.items()))
        print("  paired, support_graded minus (negative = graded better)")
        for other in [s for s in systems if s != "support_graded"]:
            for metric in ("brier", "logloss"):
                a = np.clip(systems["support_graded"], *CLIP)
                b = np.clip(systems[other], *CLIP)

                def loss(p, yy):
                    if metric == "brier":
                        return np.mean((p - yy) ** 2)
                    return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))

                est = loss(a, y) - loss(b, y)
                bs = np.sort([loss(a[d], y[d]) - loss(b[d], y[d])
                              for d in draws])
                print(f"    - {other:14} {metric:8} {est:+.4f} "
                      f"[{bs[int(.025 * B)]:+.4f}, "
                      f"{bs[int(.975 * B) - 1]:+.4f}]")
        if label == "primary":
            print("  Brier by the number of other datasets")
            for lab, m in (("1", n == 1), ("2-3", (n >= 2) & (n <= 3)),
                           ("4-9", (n >= 4) & (n <= 9)), (">=10", n >= 10)):
                parts = " ".join(
                    f"{k} {np.mean((np.clip(v[m], *CLIP) - y[m]) ** 2):.3f}"
                    for k, v in systems.items())
                print(f"    n {lab:5} ({m.sum():4d}): {parts}")


if __name__ == "__main__":
    main(*sys.argv[1:])
