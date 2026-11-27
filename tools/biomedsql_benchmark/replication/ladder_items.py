"""Draw the items of PROTOCOL_LADDER.md (seed 3).

    python ladder_items.py [SMR.parquet]

Writes results/ladder_items.json: {"q14": [...], "q5": [...], "q6": [...]}.
Gold answers are what the held-out datasets actually show.
"""

import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd

from build_items import DISEASES, THRESHOLD
from leak_checks import loci_of
from neurolang_apps.utils.engines.biomedsql.init import organ_of

HERE = Path(__file__).resolve().parent
SMR = (HERE / "../../../neurolang_data/probe/goldq/biomedsql/db/"
       "NeurodegenerativeDiseases_SMR_Genes_Full.parquet")
SEED = 3
N_Q14, N_Q5, N_Q6 = 300, 150, 150
MIN_DATASETS = 4
H_SIZE = 3


def load(path):
    cols = ["Omic", "Gene", "Disease", "p_SMR_multi", "Omic_tissue",
            "Omic_type", "ProbeChr", "Probe_bp"]
    smr = pd.read_parquet(path, columns=cols)
    info = smr.drop_duplicates("Omic").set_index("Omic")[
        ["Omic_tissue", "Omic_type"]]
    tests = smr.groupby(["Omic", "Gene", "Disease"], observed=True
                        ).p_SMR_multi.min().reset_index()
    return smr, info, tests


def row(info, omic, p):
    tissue = info.loc[omic, "Omic_tissue"]
    return {"dataset": omic, "tissue": tissue,
            "kind": info.loc[omic, "Omic_type"], "organ": organ_of(tissue),
            "p": float(p), "significant": bool(p < THRESHOLD)}


def nsig(ps, held):
    """Significant tests among the datasets of ps outside held."""
    return sum(sig(p) for o, p in ps.items() if o not in held)


def sig(p):
    return bool(p < THRESHOLD)


def pick_h(rng, ps, evidence_sig, min_sig, ok=None):
    """H of 3 datasets of ps; with min_sig, its evidence must hold at least
    that many significant tests. Returns None when no H qualifies."""
    ok = ok or NO_CONSTRAINT
    if min_sig and evidence_sig(()) < min_sig:
        return None
    combos = [c for c in itertools.combinations(sorted(ps), H_SIZE)
              if ok(c)]
    if min_sig:
        combos = [c for c in combos if evidence_sig(c) >= min_sig]
    if not combos:
        return None
    if not min_sig and ok is NO_CONSTRAINT:
        return rng.sample(sorted(ps), H_SIZE)
    return list(rng.choice(combos))


def NO_CONSTRAINT(c):
    return True


def main(path=SMR, min_sig=0, out_name="ladder_items.json"):
    """min_sig = 0 is PROTOCOL_LADDER.md as written; min_sig >= 1 is the
    variant keeping only items whose evidence holds that many significant
    datasets (see the report)."""
    min_sig = int(min_sig)
    smr, info, tests = load(path)
    # by[(gene, disease)] = {dataset: smallest p}
    by = {}
    for r in tests.itertuples(index=False):
        by.setdefault((r.Gene, r.Disease), {})[r.Omic] = r.p_SMR_multi
    rng = random.Random(SEED)
    pairs = sorted(k for k, v in by.items() if len(v) >= MIN_DATASETS)

    # Q1-Q4
    order = pairs[:]
    rng.shuffle(order)
    q14 = []
    for gene, disease in order:
        if len(q14) == N_Q14:
            break
        ps = by[(gene, disease)]
        H = pick_h(rng, ps, lambda c: nsig(ps, c), min_sig)
        if H is None:
            continue
        ev = sorted((row(info, o, p) for o, p in ps.items() if o not in H),
                    key=lambda e: e["p"])
        held = [row(info, o, ps[o]) for o in H]
        s = [h["significant"] for h in held]
        q14.append({
            "id": f"q14|{gene}|{disease}", "gene": gene, "disease": disease,
            "disease_name": DISEASES[disease], "heldout": held,
            "evidence": ev,
            "gold": {"Q1": int(s[0]), "Q2": int(any(s)), "Q3": int(all(s)),
                     "Q4": int(sum(s) >= 2)},
        })

    # Q5: H holds at least one brain and one blood dataset
    order = pairs[:]
    rng.shuffle(order)
    q5 = []
    for gene, disease in order:
        if len(q5) == N_Q5:
            break
        ps = by[(gene, disease)]
        organ = {o: organ_of(info.loc[o, "Omic_tissue"]) for o in ps}
        H = pick_h(rng, ps, lambda c: nsig(ps, c), min_sig, ok=lambda c: any(
            organ[o] == "brain" for o in c) and any(
            organ[o] == "blood" for o in c))
        if H is None:
            continue
        ev = sorted((row(info, o, p) for o, p in ps.items() if o not in H),
                    key=lambda e: e["p"])
        held = [row(info, o, ps[o]) for o in H]
        brain = any(h["significant"] for h in held if h["organ"] == "brain")
        blood = any(h["significant"] for h in held if h["organ"] == "blood")
        q5.append({
            "id": f"q5|{gene}|{disease}", "gene": gene, "disease": disease,
            "disease_name": DISEASES[disease], "heldout": held,
            "evidence": ev, "gold": {"Q5": int(brain and not blood)},
        })

    # Q6: PD and AD, H tested for both
    genes = sorted({g for g, d in by if d == "PD"
                    and (g, "AD") in by})
    rng.shuffle(genes)
    q6 = []
    for gene in genes:
        if len(q6) == N_Q6:
            break
        pd_, ad = by[(gene, "PD")], by[(gene, "AD")]
        both = sorted(set(pd_) & set(ad))
        if len(pd_) < MIN_DATASETS or len(ad) < MIN_DATASETS \
                or len(both) < H_SIZE:
            continue
        H = pick_h(rng, {o: 0 for o in both}, lambda c: nsig(pd_, c)
                   + nsig(ad, c), min_sig)
        if H is None:
            continue
        evidence = {}
        for d, ps in (("PD", pd_), ("AD", ad)):
            evidence[d] = sorted(
                (row(info, o, p) for o, p in ps.items() if o not in H),
                key=lambda e: e["p"])
        held = []
        for o in H:
            h = row(info, o, pd_[o])
            h.pop("p"), h.pop("significant")
            h["PD"] = {"p": float(pd_[o]), "significant": sig(pd_[o])}
            h["AD"] = {"p": float(ad[o]), "significant": sig(ad[o])}
            held.append(h)
        gold = any(h["PD"]["significant"] and h["AD"]["significant"]
                   for h in held)
        q6.append({
            "id": f"q6|{gene}", "gene": gene, "disease": "PD+AD",
            "disease_names": [DISEASES["PD"], DISEASES["AD"]],
            "heldout": held, "evidence": evidence, "gold": {"Q6": int(gold)},
        })

    genes_all = {i["gene"] for items in (q14, q5, q6) for i in items}
    loci = loci_of(smr, genes_all).locus.to_dict()
    for items in (q14, q5, q6):
        for i in items:
            i["locus"] = int(loci[i["gene"]])

    out = {"q14": q14, "q5": q5, "q6": q6}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / out_name).write_text(
        json.dumps(out, indent=1))
    for items, qs, planned in ((q14, ["Q1", "Q2", "Q3", "Q4"], N_Q14),
                               (q5, ["Q5"], N_Q5), (q6, ["Q6"], N_Q6)):
        for q in qs:
            y = [i["gold"][q] for i in items]
            print(f"{q}: {len(items)} items (planned {planned}); base rate "
                  f"{sum(y) / len(y):.3f}; loci "
                  f"{len({i['locus'] for i in items})}")


if __name__ == "__main__":
    main(*sys.argv[1:])
