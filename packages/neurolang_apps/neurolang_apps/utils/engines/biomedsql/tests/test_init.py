"""The units of evidence of the BiomedSQL engine, on toy tables: the
predicates of ``engines.yaml`` equal their closed forms, by lifted
inference only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from neurolang.frontend import NeurolangPDL
from neurolang.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver as sdt,
)

from .. import init
from ..init import init_engine, strength_probability

ENGINES = Path(init.__file__).resolve().parents[1] / "engines.yaml"
STRENGTH = [[0.0, 0.01], [3.0, 0.1], [5.53, 0.8], [8.0, 1.0]]
A, B = 0.01, 0.5
# (QTL dataset, tissue, gene, disease, p_SMR_multi): G1 is tested by three
# datasets, one test with two probes; G2 by one
SMR = [
    ("q1", "Cortex", "G1", "PD", 1e-9),
    ("q2", "Whole Blood", "G1", "PD", 1e-4),
    ("q2", "Whole Blood", "G1", "PD", 1e-6),
    ("q3", "Liver", "G1", "PD", 0.5),
    ("q1", "Cortex", "G2", "AD", 1e-2),
]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    db = tmp_path_factory.mktemp("data") / "biomedsql" / "db"
    db.mkdir(parents=True)
    smr = pd.DataFrame(
        SMR, columns=["Omic", "Omic_tissue", "Gene", "Disease", "p_SMR_multi"]
    ).assign(Omic_type="eQTL")
    for name, source in init.TABLES.items():
        df = smr if name == "smr" else pd.DataFrame({"x": ["a"]})
        df.to_parquet(db / f"{source}.parquet")
    nl = NeurolangPDL()
    init_engine(nl, None, db.parents[1], A, B, STRENGTH)
    nl.probabilistic_solvers = (
        sdt.solve_succ_query, dalvi_suciu_lift.solve_succ_query
    )
    nl.probabilistic_marg_solvers = (
        sdt.solve_marg_query, dalvi_suciu_lift.solve_marg_query
    )
    return nl


def test_strength_probability_by_bin():
    p = strength_probability([1.0, 1e-3, 1e-6, 1e-9, 1e-300], STRENGTH)
    assert p.tolist() == [0.01, 0.1, 0.8, 1.0, 1.0]
    assert strength_probability([0.5], [[1.0, 0.2]]).tolist() == [0.0]


@pytest.mark.parametrize("predicate", ["support", "support_prior",
                                       "support_graded"])
def test_supports_equal_their_closed_forms(engine, predicate):
    rules = yaml.safe_load(ENGINES.read_text())["engines"]["biomedsql"][
        "datalog_init"]
    with engine.scope:
        res = engine.execute_datalog_program(
            rules + f"ans(g, d, p) :- {predicate}(g, d, p)"
        )
        got = {(g, d): p for g, d, p in res.as_pandas_dataframe()
               .itertuples(index=False)}
    # G1: strongest probes 1e-9, 1e-6, 0.5; G2: 1e-2
    expected = {
        "support": {("G1", "PD"): 2 / 3, ("G2", "AD"): 0.0},
        "support_prior": {("G1", "PD"): (2 + A) / (3 + A + B),
                          ("G2", "AD"): A / (1 + A + B)},
        "support_graded": {("G1", "PD"): (1 + 0.8 + 0.01 + A) / (3 + A + B),
                           ("G2", "AD"): (0.01 + A) / (1 + A + B)},
    }[predicate]
    for key, value in expected.items():
        assert got.get(key, 0.0) == pytest.approx(value, abs=1e-12)
