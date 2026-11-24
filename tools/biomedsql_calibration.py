"""
Calibration of a gene's support by the QTL datasets of the biomedsql engine.

Leave one QTL dataset out: for each of the 28 datasets h, the engine computes,
over the other 27,

    other(q) :- qtl_dataset(q, t, k, o), (q != h)
    support(g, d, PROB) :-
        (qtl_significant(q, g, d) & selected_qtl(q) & other(q))
        // (qtl_tested(q, g, d) & selected_qtl(q) & other(q))

and every (gene, disease) pair that h tested, and that another dataset
tested, is scored against whether h found it significant. The deterministic
answer to BiomedSQL's yes/no question ("is there a significant signal?",
some other dataset significant) is scored the same way, as a contrast.

    uv run --package neurolang-apps python tools/biomedsql_calibration.py \
        --data-dir neurolang_data --out biomedsql_calibration

prints the tables of doc/biomedsql_calibration.rst and writes the
predictions (parquet) and a reliability diagram (png) under --out.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurolang.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
)
from neurolang_apps.utils import engine_registry

BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def engine(data_dir):
    nl = engine_registry.build_engine("biomedsql", data_dir=data_dir)
    # Lifted inference only: a query lifting cannot solve raises.
    nl.probabilistic_solvers = (
        small_dichotomy_theorem_based_solver.solve_succ_query,
        dalvi_suciu_lift.solve_succ_query,
    )
    nl.probabilistic_marg_solvers = (
        small_dichotomy_theorem_based_solver.solve_marg_query,
        dalvi_suciu_lift.solve_marg_query,
    )
    return nl


def run(nl, program):
    with nl.scope:
        return [tuple(r) for r in nl.execute_datalog_program(program)]


def held_out_predictions(nl):
    datasets = [q for (q,) in run(nl, "ans(q) :- qtl_dataset(q, t, k, o)")]
    frames = []
    for h in sorted(datasets):
        other = f"other(q) :- qtl_dataset(q, t, k, o), (q != '{h}')\n"
        support = dict(
            ((g, d), p)
            for g, d, p in run(
                nl,
                other
                + (
                    "support(g, d, PROB) :- "
                    "(qtl_significant(q, g, d) & selected_qtl(q) & other(q))"
                    " // "
                    "(qtl_tested(q, g, d) & selected_qtl(q) & other(q))\n"
                    "ans(g, d, p) :- support(g, d, p)"
                ),
            )
        )
        tested_other = set(
            run(nl, other + "ans(g, d) :- qtl_tested(q, g, d), other(q)")
        )
        held = run(nl, f"ans(g, d) :- qtl_tested('{h}', g, d)")
        significant = set(
            run(nl, f"ans(g, d) :- qtl_significant('{h}', g, d)")
        )
        rows = [
            (h, g, d, support.get((g, d), 0.0), (g, d) in significant)
            for g, d in held
            if (g, d) in tested_other
        ]
        frames.append(
            pd.DataFrame(
                rows, columns=["heldout", "gene", "disease", "p", "y"]
            )
        )
        print(f"{h}: {len(rows)} predictions", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df.y.astype(int)
    df["any_signal"] = (df.p > 0).astype(float)
    return df


def reliability(p, y):
    """Rows (bin, n, mean predicted, observed) and the calibration error."""
    p, y = np.asarray(p), np.asarray(y)
    rows, ece = [], 0.0
    masks = [("p = 0", p == 0)] + [
        (f"({lo:g}, {hi:g}]", (p > lo) & (p <= hi))
        for lo, hi in zip(BINS[:-1], BINS[1:])
    ]
    for name, m in masks:
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((name, int(m.sum()), p[m].mean(), y[m].mean()))
    return rows, ece


def bootstrap_ece(df, column, n=1000, seed=0):
    """95% interval of the ECE, resampling the (correlated) held-out folds."""
    folds = {h: f for h, f in df.groupby("heldout")}
    keys, rng, values = sorted(folds), random.Random(seed), []
    for _ in range(n):
        sample = pd.concat([folds[k] for k in rng.choices(keys, k=len(keys))])
        values.append(reliability(sample[column], sample.y)[1])
    values.sort()
    return values[int(0.025 * n)], values[int(0.975 * n)]


def report(df, column, label):
    rows, ece = reliability(df[column], df.y)
    brier = ((df[column] - df.y) ** 2).mean()
    brier0 = ((df.y.mean() - df.y) ** 2).mean()
    lo, hi = bootstrap_ece(df, column)
    print(f"\n{label}: n={len(df)}, base rate {df.y.mean():.4f}")
    print(
        f"  {'predicted':12} {'pairs':>7} {'mean predicted':>15}"
        f" {'observed':>9}"
    )
    for name, n, mp, ob in rows:
        print(f"  {name:12} {n:7d} {mp:15.3f} {ob:9.3f}")
    print(
        f"  ECE {ece:.4f} [95% CI {lo:.4f}, {hi:.4f}] | Brier {brier:.5f} "
        f"(base rate {brier0:.5f}, skill {1 - brier / brier0:+.3f}) | "
        f"AUC {roc_auc_score(df.y, df[column]):.3f}"
    )
    return rows


def plot(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for column, label, marker in (
        ("p", "NeuroLang support", "o"),
        ("any_signal", "any signal (yes/no)", "s"),
    ):
        sub = df[df.p > 0]
        rows, _ = reliability(sub[column], sub.y)
        ax.plot(
            [r[2] for r in rows],
            [r[3] for r in rows],
            marker=marker,
            label=label,
        )
        for r in rows:
            ax.annotate(
                str(r[1]),
                (r[2], r[3]),
                fontsize=6,
                textcoords="offset points",
                xytext=(4, -8),
            )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("significant in the held-out QTL dataset")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("neurolang_data")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("biomedsql_calibration")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = held_out_predictions(engine(args.data_dir))
    df.to_parquet(args.out / "predictions.parquet")
    supported = df[df.p > 0]
    report(df, "p", "All held-out predictions")
    report(
        supported,
        "p",
        "Pairs some other dataset found significant (support > 0)",
    )
    report(
        supported,
        "any_signal",
        "Same pairs, deterministic yes/no answer (any signal = 1)",
    )
    wrong0 = int(((df.p == 0) & (df.y == 1)).sum())
    wrong1 = int(((df.p == 1) & (df.y == 0)).sum())
    print(
        f"\nHard values: {wrong0} of {int(df.y.sum())} held-out positives had "
        f"support 0; {wrong1} of {int((df.p == 1).sum())} supports of 1 were "
        "not significant in the held-out dataset"
    )
    plot(df, args.out / "reliability.png")


if __name__ == "__main__":
    main()
