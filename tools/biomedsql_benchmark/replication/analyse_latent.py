"""Readings 1-5 of PROTOCOL_LATENT.md.

    uv run --no-sync python analyse_latent.py

Reads results/latent_ladder.json, latent_oof.json, same_models_oof.json,
ladder_engine_signal.json, ladder_llm.jsonl; writes results/latent_analysis.txt.
Brier clipped to [0.02, 0.98]; 95% intervals, 2,000 draws, resampling loci.
"""

import io
import json
from contextlib import redirect_stdout

import numpy as np

import analyse_ladder as al
import leak_checks as lc

HERE = al.HERE
ITEMS = "results/ladder_items_signal.json"
ENGINE = "results/ladder_engine_signal.json"
LLM = HERE / "results/ladder_llm.jsonl"


def pooled(table, qs, a, b):
    ds, ls = [], []
    for q in qs:
        t = table[q]
        y = t["y"].astype(float)
        ok = ~(np.isnan(t[a]) | np.isnan(t[b]))
        ds.append(al.brier(t[a][ok], y[ok]) - al.brier(t[b][ok], y[ok]))
        ls.append(t["locus"][ok])
    return np.concatenate(ds), np.concatenate(ls)


def main():
    lat = json.load(open(HERE / "results/latent_ladder.json"))
    table = al.load(ITEMS, ENGINE, LLM)
    for q in al.RUNGS:
        t = table[q]
        t["latent"] = np.array([lat["values"][i][q]["latent"] for i in t["id"]])
        ref = np.array([lat["values"][i][q]["closed"] for i in t["id"]])
        assert np.abs(t["latent"] - ref).max() < 1e-8
    print("Latent model (PROTOCOL_LATENT.md). Parameters of the outer fold "
          "containing each item's locus.")
    print("Source of the latent value per rung: " + ", ".join(
        f"{q} {lat['source'][q]}" for q in al.RUNGS))
    print("max |engine - closed form| per rung: " + ", ".join(
        f"{q} {lat['max_diff'][q]:.3g}" if lat["max_diff"][q] is not None
        else f"{q} n/a" for q in al.RUNGS))
    print("engine items solved per rung (of the rung's items): " + ", ".join(
        f"{q} {lat['n_solved'][q]}/{len(table[q]['id'])}" for q in al.RUNGS))
    print("\nLegend: latent = latent-effect model; engine = independent "
          "engine of the ladder (ladder_engine_signal.json); flags, full = "
          "Sonnet.\n")
    al.per_rung(table, ["latent", "engine", "flags", "full"])
    print("\nPaired differences, latent minus X, per rung "
          "(below 0: latent better)")
    for q in al.RUNGS:
        for x in ("engine", "flags", "full"):
            d, l = pooled(table, [q], "latent", x)
            print(f"  {q} latent - {x:6} {al.fmt(lc.cluster_ci(d, l))} "
                  f"(n {len(d)})")

    def rd(title, qs, x):
        d, l = pooled(table, qs, "latent", x)
        print(f"  {title:34} latent - {x:6} {al.fmt(lc.cluster_ci(d, l))} "
              f"(n {len(d)})")
    print("\n1. H4, the dependence helps: latent minus independent engine")
    rd("pooled Q2, Q4, Q5", ["Q2", "Q4", "Q5"], "engine")
    rd("all rungs pooled", al.RUNGS, "engine")
    print("\n2. H5, same evidence: latent minus Sonnet-flags")
    rd("pooled Q4-Q6", al.POOLED, "flags")
    rd("all rungs pooled", al.RUNGS, "flags")
    print("\n3. H6, full table: latent minus Sonnet-full")
    rd("pooled Q4-Q6", al.POOLED, "full")
    rd("all rungs pooled", al.RUNGS, "full")

    print("\n4. Single question, 3,960 single-held-out items, out of fold")
    o = json.load(open(HERE / "results/latent_oof.json"))
    s = json.load(open(HERE / "results/same_models_oof.json"))
    ids = sorted(o)
    assert all(i in s for i in ids) and len(ids) == 3960
    y = np.array([o[i]["y"] for i in ids])
    assert (y == np.array([s[i]["y"] for i in ids])).all()
    assert all(o[i]["fold"] == s[i]["fold"] for i in ids)
    loc = np.array([o[i]["locus"] for i in ids])
    P = {"latent": np.array([o[i]["latent"] for i in ids]),
         "E1": np.array([s[i]["E1"] for i in ids]),
         "F1": np.array([s[i]["F1"] for i in ids])}
    for k, p in P.items():
        b = lc.brier(p, y)
        print(f"  {k:7} Brier "
              f"{b.mean():.4f} [{lc.cluster_ci(b, loc)[1]:.4f}, "
              f"{lc.cluster_ci(b, loc)[2]:.4f}]")
    for x in ("E1", "F1"):
        d = lc.brier(P["latent"], y) - lc.brier(P[x], y)
        print(f"  Brier latent - {x}: {lc.fmt(lc.cluster_ci(d, loc))}")

    print("\n5. Coherence on Q1-Q4 (tolerance 0.01)")
    al.coherence(table, ["latent", "engine", "flags", "full"])


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "results/latent_analysis.txt").write_text(buf.getvalue())
    print(buf.getvalue())
