"""The 168 questions of PROTOCOL.md: 7 levels x 3 templates x (6 scored + 2 dev).

    uv run --no-sync python questions.py     # -> questions.json

For each template, in the order of ``TEXT``, the candidate instances are listed
from the tables of ``data/`` and sorted, then visited in the order of one
random permutation (numpy, ``default_rng(5)`` shared by all templates, drawn
template after template); a candidate is kept when its lineage holds 2 to 20
uncertain rows (P1: its one row; P7: at most 20 rows per drug, gene or dataset
summed over, and at least 2 rows in all) and its answer is not trivial
(P1-P6: strictly between 0 and 1; P7: some unit strictly between 0 and 1),
until 8 are kept: the first 6 are scored, the last 2 are dev.

The last four templates (P4d, P4e, P5d, P6d, ``restored``) are the protocol's
example shapes left out of the first build: the lifted solvers refuse P4d, P5d
and P6d (``engine_refused``); P4e has only 2 valid instances. Drawn after
the 21 others (so the draws of those are unchanged), by the same rules; one
that has fewer than 8 valid instances keeps them all, scored first.

Each question names the uncertainty model in one sentence (``UNCERTAINTY``).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from gold import DB, REFUSED, RESTORED, TEMPLATES, exact, lineage_sizes

HERE = Path(__file__).resolve().parent
SEED = 5
SCORED, DEV = 6, 2

DISEASES = {
    "AD": "Alzheimer's disease (AD)",
    "PD": "Parkinson's disease (PD)",
    "ALS": "amyotrophic lateral sclerosis (ALS)",
    "FTD": "frontotemporal dementia (FTD)",
    "LBD": "Lewy body dementia (LBD)",
    "PSP": "progressive supranuclear palsy (PSP)",
}
GWAS_DISEASES = ("AD", "PD")    # snp_causal holds only these

UNCERTAINTY = (
    "Each row of test_evidence, dataset_reliable, indication_success and "
    "snp_causal is an independent event that holds with probability prob."
)

TEXT = {
    "P1a": "What is the probability that the SMR test of gene {g} in the QTL "
           "dataset {d} is evidence that {g} mediates {D}?",
    "P1b": "What is the probability that the drug {x} succeeds for the "
           "indication '{m}'?",
    "P1c": "What is the probability that the GWAS variant {s} is a causal SNP "
           "of gene {g} for {D}?",
    "P2a": "What is the probability that gene {g} has evidence for {D} in at "
           "least one brain QTL dataset?",
    "P2b": "What is the probability that the drug {x} succeeds for at least "
           "one of its indications?",
    "P2c": "What is the probability that gene {g} has at least one causal SNP "
           "for {D}?",
    "P3a": "What is the probability that the drug {x} targets at least one "
           "gene that has evidence for {D} in some QTL dataset?",
    "P3b": "What is the probability that the drug {x} targets at least one "
           "gene that has a causal SNP for {D}?",
    "P3c": "What is the probability that gene {g} has evidence for {D} in at "
           "least one reliable QTL dataset?",
    "P4a": "What is the probability that some target gene of the drug {x} has "
           "both evidence for {D} in some QTL dataset and a causal SNP for "
           "{D}?",
    "P4b": "What is the probability that there is a reliable QTL dataset in "
           "which at least one target gene of the drug {x} has evidence for "
           "{D}?",
    "P4c": "What is the probability that there is a reliable QTL dataset in "
           "which both gene {g1} and gene {g2} have evidence for {D}?",
    "P5a": "What is the probability that gene {g} has evidence for {D} in a "
           "brain QTL dataset, or has evidence for {D} in a reliable QTL "
           "dataset (or both)?",
    "P5b": "What is the probability that gene {g1} or gene {g2} (or both) has "
           "evidence for {D} in a reliable QTL dataset?",
    "P5c": "What is the probability that the drug {x} targets a gene with "
           "evidence for {D} in a reliable QTL dataset, or targets a gene "
           "with evidence for {D} in a blood QTL dataset (or both)?",
    "P6a": "Given that gene {g0}, a target of the drug {x}, has evidence for "
           "{D} in some blood QTL dataset, what is the probability that {x} "
           "targets a gene with evidence for {D} in a reliable QTL dataset?",
    "P6b": "What is the probability that the drug {x} targets at least one "
           "gene with evidence for {D} in some QTL dataset, and that none of "
           "its target genes has a causal SNP for {D}?",
    "P6c": "Given that gene {g} has evidence for {D} in some brain QTL "
           "dataset, what is the probability that {g} has evidence for {D} in "
           "a reliable QTL dataset?",
    "P7a": "Consider the approved drugs that have '{m}' among their "
           "indications (whether or not they succeed for it). What is the "
           "expected number of these drugs that target at least one gene with "
           "evidence for {D} in some QTL dataset?",
    "P7b": "What is the expected number of target genes of the drug {x} that "
           "have evidence for {D} in a reliable QTL dataset?",
    "P7c": "What is the expected number of QTL datasets that are reliable and "
           "in which at least one target gene of the drug {x} has evidence "
           "for {D}?",
    # The protocol's example shapes that the lifted solvers refuse.
    "P4d": "What is the probability that some target gene of the drug {x} has "
           "evidence for {D} both in a brain QTL dataset and in a blood QTL "
           "dataset?",
    "P4e": "What is the probability that some target gene of the drug {x} has "
           "both evidence for Parkinson's disease (PD) and evidence for "
           "Alzheimer's disease (AD), each in some QTL dataset?",
    "P5d": "What is the probability that gene {g} has evidence for {D} in a "
           "brain QTL dataset, or has evidence for {D} in at least two QTL "
           "datasets (or both)?",
    "P6d": "What is the probability that the drug {x} targets at least one "
           "gene with evidence for Parkinson's disease (PD) in some QTL "
           "dataset, and that none of its target genes has evidence for "
           "Alzheimer's disease (AD)?",
}


def candidates(db, template):
    """All the instances of *template*, as parameter dicts, sorted."""
    E = sorted(k for k in db.prob if k[0] == "E")
    genes = sorted({(g, dis) for _, _, g, dis in E})
    drugs = sorted(db.T)
    snp_genes = sorted(k for k in db.C_by if k[1] in GWAS_DISEASES)
    target_pairs = sorted({
        (g1, g2) for x in drugs for g1, g2 in combinations(sorted(set(db.T[x])), 2)
    })
    if template == "P1a":
        return [dict(d=d, g=g, dis=dis) for _, d, g, dis in E
                if db.prob[("E", d, g, dis)] < 1]
    if template == "P1b":
        return [dict(x=x, m=m) for k, x, m in
                sorted(k for k in db.prob if k[0] == "I") if db.prob[(k, x, m)] < 1]
    if template == "P1c":
        return [dict(s=s, g=g, dis=dis) for _, s, g, dis in
                sorted(k for k in db.prob if k[0] == "C")]
    if template in ("P2a", "P3c", "P5a", "P5d", "P6c"):
        return [dict(g=g, dis=dis) for g, dis in genes]
    if template == "P2b":
        return [dict(x=x) for x in sorted(db.I_by)]
    if template == "P2c":
        return [dict(g=g, dis=dis) for g, dis in snp_genes]
    if template in ("P4e", "P6d"):
        return [dict(x=x) for x in drugs]
    if template in ("P3a", "P4b", "P4d", "P5c", "P7b", "P7c"):
        return [dict(x=x, dis=dis) for x in drugs for dis in DISEASES]
    if template in ("P3b", "P4a", "P6b"):
        return [dict(x=x, dis=dis) for x in drugs for dis in GWAS_DISEASES]
    if template in ("P4c", "P5b"):
        return [dict(g1=g1, g2=g2, dis=dis) for g1, g2 in target_pairs
                for dis in DISEASES
                if db.E_by[(g1, dis)] and db.E_by[(g2, dis)]]
    if template == "P6a":
        return [dict(x=x, g0=g0, dis=dis) for x in drugs
                for g0 in sorted(set(db.T[x])) for dis in DISEASES
                if any(db.organ[d] == "blood" for d in db.E_by[(g0, dis)])]
    if template == "P7a":
        return [dict(m=m, dis=dis) for m in sorted(db.drugs_of)
                for dis in DISEASES]
    raise KeyError(template)


def acceptable(db, template, params):
    """Whether the instance's lineage is in range and its answer not trivial."""
    whole, largest = lineage_sizes(db, template, params)
    level = template[:2]
    if level == "P1":
        if whole != 1:
            return False
    elif level == "P7":
        if whole < 2 or largest > 20:
            return False
    elif not 2 <= whole <= 20:
        return False
    values = [exact(u, db) for u in TEMPLATES[template](db, **params)]
    inside = [1e-9 < v < 1 - 1e-9 for v in values]
    return any(inside) if level == "P7" else all(inside)


def text(template, params):
    names = {
        "g": params.get("g"), "d": params.get("d"), "x": params.get("x"),
        "m": params.get("m"), "s": params.get("s"), "g0": params.get("g0"),
        "g1": params.get("g1"), "g2": params.get("g2"),
        "D": DISEASES.get(params.get("dis")),
    }
    return f"{TEXT[template].format(**names)} {UNCERTAINTY}"


def main():
    db = DB()
    rng = np.random.default_rng(SEED)
    questions = []
    for template in TEXT:
        pool = candidates(db, template)
        kept = []
        for i in rng.permutation(len(pool)):
            if acceptable(db, template, pool[i]):
                kept.append(pool[i])
                if len(kept) == SCORED + DEV:
                    break
        print(f"{template}: {len(pool)} candidates, kept {len(kept)}")
        # A refused template keeps as many as exist (scored first).
        assert len(kept) == SCORED + DEV or template in RESTORED, template
        for n, params in enumerate(kept):
            questions.append({
                "id": f"{template}-{n}",
                "level": template[:2],
                "template": template,
                "params": params,
                "split": "scored" if n < SCORED else "dev",
                "restored": template in RESTORED,
                "engine_refused": template in REFUSED,
                "question": text(template, params),
            })
    json.dump(questions, open(HERE / "questions.json", "w"), indent=1)
    print(f"{len(questions)} questions")


if __name__ == "__main__":
    main()
