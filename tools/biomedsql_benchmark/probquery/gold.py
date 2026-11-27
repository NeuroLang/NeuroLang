"""Gold answers of PROTOCOL.md, independent of NeuroLang.

    uv run --no-sync python gold.py       # questions.json -> gold.json

For each question, its lineage (the rows of the uncertain tables that can
change its answer) is collected from the tables of ``data/`` with pandas, all
2^m possible worlds of those rows are enumerated with numpy, the question is
evaluated as a predicate on every world, and the weights of the worlds where it
holds are summed. A conditional question is P(A and B) / P(B); an expected
number (P7) is the sum, over its units (drugs, genes or datasets), of the
probability of each unit's event, each with its own lineage.

10% of the questions (seed 5) are also estimated by Monte Carlo with 10^6
sampled worlds and must lie within 3 standard errors of the exact value.

The templates are defined here (``TEMPLATES``), as functions from the
database and the instance parameters to units; ``questions.py`` uses them to
draw the instances. Nothing here imports NeuroLang.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_LINEAGE = 20
MC_SAMPLES = 10**6


class DB:
    """The uncertain rows, by key, and the deterministic tables, indexed."""

    def __init__(self, data=DATA):
        read = lambda n: pd.read_parquet(Path(data) / f"{n}.parquet")
        self.prob = {}
        self.E_by = defaultdict(list)       # (gene, disease) -> datasets
        for d, g, dis, p in read("test_evidence").itertuples(index=False):
            self.prob[("E", d, g, dis)] = p
            self.E_by[(g, dis)].append(d)
        for d, p in read("dataset_reliable").itertuples(index=False):
            self.prob[("R", d)] = p
        self.I_by = defaultdict(list)       # drug -> mesh headings
        for x, m, p in read("indication_success").itertuples(index=False):
            self.prob[("I", x, m)] = p
            self.I_by[x].append(m)
        self.C_by = defaultdict(list)       # (gene, disease) -> snps
        for s, g, dis, p in read("snp_causal").itertuples(index=False):
            self.prob[("C", s, g, dis)] = p
            self.C_by[(g, dis)].append(s)
        self.T = defaultdict(list)          # drug -> target genes
        for x, g in read("drug_target").itertuples(index=False):
            self.T[x].append(g)
        self.approved = set(read("drug_approved").drug)
        self.organ = dict(read("dataset_organ").itertuples(index=False))
        self.drugs_of = defaultdict(list)   # mesh heading -> drugs
        for x, ms in self.I_by.items():
            for m in ms:
                self.drugs_of[m].append(x)


# ---------------------------------------------------------------- events
# An event is a small expression tree over row keys, evaluated on a matrix
# of worlds (one column per lineage row) by ``evaluate``.

def row(key):
    return ("row", key)


def any_(*events):
    return ("or", [e for e in events if e is not None])


def at_least_two(*events):
    return ("at_least_two", [e for e in events if e is not None])


def all_(*events):
    return ("and", list(events))


def not_(event):
    return ("not", event)


def keys(event, out=None):
    out = [] if out is None else out
    if event[0] == "row":
        if event[1] not in out:
            out.append(event[1])
    elif event[0] == "not":
        keys(event[1], out)
    else:
        for e in event[1]:
            keys(e, out)
    return out


def evaluate(event, column, worlds):
    kind = event[0]
    if kind == "row":
        return worlds[:, column[event[1]]]
    if kind == "not":
        return ~evaluate(event[1], column, worlds)
    parts = [evaluate(e, column, worlds) for e in event[1]]
    if kind == "at_least_two":
        return sum(p.astype(np.int64) for p in parts) >= 2 if parts else (
            np.zeros(len(worlds), bool)
        )
    if kind == "or":
        out = np.zeros(len(worlds), bool)
        for p in parts:
            out |= p
        return out
    out = np.ones(len(worlds), bool)
    for p in parts:
        out &= p
    return out


@dataclass
class Unit:
    """P(event | given), given None for an unconditional event."""

    event: tuple
    given: Optional[tuple] = None

    def lineage(self):
        out = keys(self.event)
        if self.given is not None:
            keys(self.given, out)
        return out


def exact(unit, db):
    lineage = unit.lineage()
    m = len(lineage)
    if m == 0:
        return 0.0
    assert m <= MAX_LINEAGE, m
    worlds = (
        (np.arange(2**m, dtype=np.int64)[:, None] >> np.arange(m)) & 1
    ).astype(bool)
    weight = np.ones(2**m)
    for j, k in enumerate(lineage):
        weight *= np.where(worlds[:, j], db.prob[k], 1 - db.prob[k])
    column = {k: j for j, k in enumerate(lineage)}
    holds = evaluate(unit.event, column, worlds)
    if unit.given is None:
        return float(weight[holds].sum())
    given = evaluate(unit.given, column, worlds)
    if not given.any():
        return float("nan")     # P(given) = 0: never drawn
    return float(weight[holds & given].sum() / weight[given].sum())


def monte_carlo(unit, db, rng, n=MC_SAMPLES):
    """(estimate, standard error) of P(unit) from n sampled worlds."""
    lineage = unit.lineage()
    if not lineage:
        return 0.0, 0.0
    p = np.array([db.prob[k] for k in lineage])
    worlds = rng.random((n, len(lineage))) < p
    column = {k: j for j, k in enumerate(lineage)}
    holds = evaluate(unit.event, column, worlds)
    if unit.given is None:
        est = holds.mean()
        return float(est), float(np.sqrt(est * (1 - est) / n))
    given = evaluate(unit.given, column, worlds)
    k = given.sum()
    est = (holds & given).sum() / k
    return float(est), float(np.sqrt(est * (1 - est) / k))


# ---------------------------------------------------------------- templates
def E(db, g, dis, organ=None):
    """The rows of test_evidence of gene g for dis (in datasets of organ)."""
    return [
        ("E", d, g, dis) for d in db.E_by[(g, dis)]
        if organ is None or db.organ[d] == organ
    ]


def any_rows(ks):
    return any_(*[row(k) for k in ks])


def reliable_evidence(db, g, dis):
    """g has evidence for dis in a reliable dataset."""
    return any_(*[
        all_(row(("E", d, g, dis)), row(("R", d))) for d in db.E_by[(g, dis)]
    ])


def evidence(db, g, dis, organ=None):
    return any_rows(E(db, g, dis, organ))


def causal(db, g, dis):
    return any_rows([("C", s, g, dis) for s in db.C_by[(g, dis)]])


def targets(db, x, event_of_gene):
    return any_(*[event_of_gene(g) for g in db.T[x]])


def p1a(db, d, g, dis):
    return [Unit(row(("E", d, g, dis)))]


def p1b(db, x, m):
    return [Unit(row(("I", x, m)))]


def p1c(db, s, g, dis):
    return [Unit(row(("C", s, g, dis)))]


def p2a(db, g, dis):
    return [Unit(evidence(db, g, dis, "brain"))]


def p2b(db, x):
    return [Unit(any_rows([("I", x, m) for m in db.I_by[x]]))]


def p2c(db, g, dis):
    return [Unit(causal(db, g, dis))]


def p3a(db, x, dis):
    return [Unit(targets(db, x, lambda g: evidence(db, g, dis)))]


def p3b(db, x, dis):
    return [Unit(targets(db, x, lambda g: causal(db, g, dis)))]


def p3c(db, g, dis):
    return [Unit(reliable_evidence(db, g, dis))]


def p4a(db, x, dis):
    return [Unit(targets(db, x, lambda g: all_(
        evidence(db, g, dis), causal(db, g, dis)
    )))]


def reliable_dataset_for_targets(db, x, d, dis):
    """d is reliable and some target of x has evidence for dis in d."""
    return all_(row(("R", d)), any_rows(
        [("E", d, g, dis) for g in db.T[x] if ("E", d, g, dis) in db.prob]
    ))


def target_datasets(db, x, dis):
    return sorted({d for g in db.T[x] for d in db.E_by[(g, dis)]})


def p4b(db, x, dis):
    return [Unit(any_(*[
        reliable_dataset_for_targets(db, x, d, dis)
        for d in target_datasets(db, x, dis)
    ]))]


def p4c(db, g1, g2, dis):
    common = sorted(set(db.E_by[(g1, dis)]) & set(db.E_by[(g2, dis)]))
    return [Unit(any_(*[
        all_(row(("R", d)), row(("E", d, g1, dis)), row(("E", d, g2, dis)))
        for d in common
    ]))]


def p5a(db, g, dis):
    return [Unit(any_(
        evidence(db, g, dis, "brain"), reliable_evidence(db, g, dis)
    ))]


def p5b(db, g1, g2, dis):
    return [Unit(any_(
        reliable_evidence(db, g1, dis), reliable_evidence(db, g2, dis)
    ))]


def p5c(db, x, dis):
    return [Unit(any_(
        targets(db, x, lambda g: reliable_evidence(db, g, dis)),
        targets(db, x, lambda g: evidence(db, g, dis, "blood")),
    ))]


def p6a(db, x, g0, dis):
    return [Unit(
        targets(db, x, lambda g: reliable_evidence(db, g, dis)),
        given=evidence(db, g0, dis, "blood"),
    )]


def p6b(db, x, dis):
    return [Unit(all_(
        targets(db, x, lambda g: evidence(db, g, dis)),
        not_(targets(db, x, lambda g: causal(db, g, dis))),
    ))]


def p6c(db, g, dis):
    return [Unit(
        reliable_evidence(db, g, dis), given=evidence(db, g, dis, "brain")
    )]


def p7a(db, m, dis):
    drugs = sorted(set(db.drugs_of[m]) & db.approved)
    return [Unit(targets(db, x, lambda g: evidence(db, g, dis))) for x in drugs]


def p7b(db, x, dis):
    return [Unit(reliable_evidence(db, g, dis)) for g in sorted(set(db.T[x]))]


def p7c(db, x, dis):
    return [
        Unit(reliable_dataset_for_targets(db, x, d, dis))
        for d in target_datasets(db, x, dis)
    ]


# The protocol's example shapes left out of the first build (restored), three
# of which the lifted solvers refuse (engine_refused), as scored templates
# worded as the protocol's examples.
def p4d(db, x, dis):
    return [Unit(targets(db, x, lambda g: all_(
        evidence(db, g, dis, "brain"), evidence(db, g, dis, "blood")
    )))]


def p4e(db, x):
    return [Unit(targets(db, x, lambda g: all_(
        evidence(db, g, "PD"), evidence(db, g, "AD")
    )))]


def p5d(db, g, dis):
    return [Unit(any_(
        evidence(db, g, dis, "brain"),
        at_least_two(*[row(k) for k in E(db, g, dis)]),
    ))]


def p6d(db, x):
    return [Unit(all_(
        targets(db, x, lambda g: evidence(db, g, "PD")),
        not_(targets(db, x, lambda g: evidence(db, g, "AD"))),
    ))]


# P4e was set aside for its few valid instances, not refused: the lifted
# solvers answer it (2 instances exist).
RESTORED = ("P4d", "P4e", "P5d", "P6d")
REFUSED = ("P4d", "P5d", "P6d")

TEMPLATES = {
    "P1a": p1a, "P1b": p1b, "P1c": p1c,
    "P2a": p2a, "P2b": p2b, "P2c": p2c,
    "P3a": p3a, "P3b": p3b, "P3c": p3c,
    "P4a": p4a, "P4b": p4b, "P4c": p4c,
    "P5a": p5a, "P5b": p5b, "P5c": p5c,
    "P6a": p6a, "P6b": p6b, "P6c": p6c,
    "P7a": p7a, "P7b": p7b, "P7c": p7c,
    "P4d": p4d, "P4e": p4e, "P5d": p5d, "P6d": p6d,
}


def units(db, question):
    return TEMPLATES[question["template"]](db, **question["params"])


def gold(db, question):
    return sum(exact(u, db) for u in units(db, question))


def lineage_sizes(db, template, params):
    """(rows of the whole lineage, rows of the largest unit's lineage)."""
    us = TEMPLATES[template](db, **params)
    whole = set()
    for u in us:
        whole |= set(u.lineage())
    return len(whole), max((len(u.lineage()) for u in us), default=0)


def main():
    db = DB()
    questions = json.load(open(HERE / "questions.json"))
    out = {}
    for q in questions:
        whole, largest = lineage_sizes(db, q["template"], q["params"])
        out[q["id"]] = {
            "gold": gold(db, q), "lineage": whole, "max_unit_lineage": largest,
        }
    rng = np.random.default_rng(5)
    checked = sorted(
        rng.choice(len(questions), size=round(0.1 * len(questions)),
                   replace=False)
    )
    failures = 0
    for i in checked:
        q = questions[i]
        est = se2 = 0.0
        for u in units(db, q):
            e, s = monte_carlo(u, db, rng)
            est, se2 = est + e, se2 + s * s
        se = np.sqrt(se2)
        g = out[q["id"]]["gold"]
        ok = abs(est - g) <= 3 * max(se, 1 / MC_SAMPLES)
        failures += not ok
        out[q["id"]]["mc"] = {"estimate": est, "se": float(se), "ok": bool(ok)}
        print(f"MC {q['id']}: gold {g:.6f} mc {est:.6f} se {se:.2e} "
              f"{'ok' if ok else 'FAIL'}")
    print(f"MC check: {len(checked)} questions, {failures} failures")
    json.dump(out, open(HERE / "gold.json", "w"), indent=1)
    lin = [v["lineage"] for v in out.values()]
    print(f"{len(out)} golds; lineage {min(lin)}-{max(lin)} rows")


if __name__ == "__main__":
    main()
