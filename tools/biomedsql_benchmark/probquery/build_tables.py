"""Materialise the uncertain and deterministic tables of PROTOCOL.md.

    uv run --no-sync python build_tables.py

Writes ``data/<table>.parquet`` (git-ignored) from the five BiomedSQL tables
read as the ``biomedsql`` engine reads them, and prints the row counts and the
probability distributions.

Uncertain tables (every row an independent event with probability ``prob``):

* ``test_evidence(dataset, gene, disease, prob)``: one row per SMR test (QTL
  dataset ``Omic``, ``Gene``, ``Disease``), ``prob`` the ``evidence_strength``
  bin (``engines.yaml``, biomedsql) of the -log10 of the smallest
  ``p_SMR_multi`` over the gene's probes; rows with prob < 0.01 are dropped.
* ``dataset_reliable(dataset, prob)``: brain 0.9, blood 0.8, other organs 0.7
  (organ of the dataset's tissue, ``organ_of`` of the biomedsql engine).
* ``indication_success(drug, mesh_heading, prob)``: one row per drug name and
  MeSH heading of ``drug_indication``, by the largest ``max_phase_for_ind`` of
  the pair: 4 -> 1.0, 3 -> 0.58, 2 -> 0.29, 1 -> 0.13, else (0.5) 0.05.
* ``snp_causal(snp, gene, disease, prob)``: one row per variant of ``pd_gwas``
  (disease 'PD') and ``ad_gwas`` ('AD') with p < 1e-5 and a nearest gene, for
  each gene of ``nearestGene`` split on ';' (a variant between two genes is a
  row for each); prob 0.5 if p < 5e-8, else 0.1; the smallest p of a repeated
  (snp, gene, disease) counts.

Deterministic tables:

* ``drug_target(drug, gene)``: distinct (drugName, approvedSymbol) of
  ``drug_indication``;
* ``drug_approved(drug)``: distinct drugName of ``drug`` with drugIsApproved;
* ``dataset_organ(dataset, organ)``: the 28 QTL datasets of ``smr``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "neurolang_data/probe/goldq/biomedsql/db"
DATA = HERE / "data"
ENGINES = ROOT / "packages/neurolang_apps/neurolang_apps/utils/engines"

RELIABILITY = {"brain": 0.9, "blood": 0.8}
OTHER_RELIABILITY = 0.7
PHASE_SUCCESS = {4.0: 1.0, 3.0: 0.58, 2.0: 0.29, 1.0: 0.13}
OTHER_PHASE_SUCCESS = 0.05
MIN_EVIDENCE = 0.01

UNCERTAIN = {
    "test_evidence": ["dataset", "gene", "disease", "prob"],
    "dataset_reliable": ["dataset", "prob"],
    "indication_success": ["drug", "mesh_heading", "prob"],
    "snp_causal": ["snp", "gene", "disease", "prob"],
}
DETERMINISTIC = {
    "drug_target": ["drug", "gene"],
    "drug_approved": ["drug"],
    "dataset_organ": ["dataset", "organ"],
}


def evidence_strength():
    cfg = yaml.safe_load(open(ENGINES / "engines.yaml"))
    return cfg["engines"]["biomedsql"]["python_init_options"][
        "evidence_strength"
    ]


def source(name):
    from neurolang_apps.utils.engines.biomedsql.init import TABLES

    return DB / f"{TABLES[name]}.parquet"


def build():
    from neurolang_apps.utils.engines.biomedsql.init import (
        organ_of,
        strength_probability,
    )

    smr = pd.read_parquet(
        source("smr"),
        columns=["Omic", "Omic_tissue", "Gene", "Disease", "p_SMR_multi"],
    )
    tests = (
        smr.groupby(["Omic", "Gene", "Disease"], observed=True)
        .p_SMR_multi.min()
        .reset_index()
    )
    tests["prob"] = strength_probability(
        tests.p_SMR_multi, evidence_strength()
    )
    test_evidence = tests.loc[
        tests.prob >= MIN_EVIDENCE, ["Omic", "Gene", "Disease", "prob"]
    ]
    test_evidence.columns = UNCERTAIN["test_evidence"]

    datasets = smr.drop_duplicates("Omic")[["Omic", "Omic_tissue"]]
    dataset_organ = pd.DataFrame(
        {
            "dataset": datasets.Omic.astype(str),
            "organ": [organ_of(t) for t in datasets.Omic_tissue],
        }
    ).sort_values("dataset")
    dataset_reliable = pd.DataFrame(
        {
            "dataset": dataset_organ.dataset,
            "prob": [
                RELIABILITY.get(o, OTHER_RELIABILITY)
                for o in dataset_organ.organ
            ],
        }
    )

    di = pd.read_parquet(
        source("drug_indication"),
        columns=["drugName", "approvedSymbol", "mesh_heading",
                 "max_phase_for_ind"],
    )
    di = di[(di.drugName != "") & di.drugName.notna()]
    phase = (
        di[di.mesh_heading.fillna("") != ""]
        .groupby(["drugName", "mesh_heading"])
        .max_phase_for_ind.max()
        .reset_index()
    )
    indication_success = pd.DataFrame(
        {
            "drug": phase.drugName,
            "mesh_heading": phase.mesh_heading,
            "prob": [
                PHASE_SUCCESS.get(float(p), OTHER_PHASE_SUCCESS)
                for p in phase.max_phase_for_ind
            ],
        }
    )
    drug_target = (
        di[di.approvedSymbol.fillna("") != ""][["drugName", "approvedSymbol"]]
        .drop_duplicates()
        .rename(columns={"drugName": "drug", "approvedSymbol": "gene"})
    )
    drug = pd.read_parquet(
        source("drug"), columns=["drugName", "drugIsApproved"]
    )
    drug_approved = pd.DataFrame(
        {
            "drug": sorted(
                set(drug.drugName[drug.drugIsApproved.astype(bool)]) - {""}
            )
        }
    )

    snps = []
    for name, disease in (("pd_gwas", "PD"), ("ad_gwas", "AD")):
        g = pd.read_parquet(source(name), columns=["SNP", "p", "nearestGene"])
        g = g[(g.p < 1e-5) & (g.nearestGene.fillna("") != "")]
        g = g.assign(gene=g.nearestGene.str.split(";")).explode("gene")
        g["gene"] = g.gene.str.strip()
        g = g[g.gene != ""]
        g = g.groupby(["SNP", "gene"]).p.min().reset_index()
        snps.append(
            pd.DataFrame(
                {
                    "snp": g.SNP,
                    "gene": g.gene,
                    "disease": disease,
                    "prob": np.where(g.p < 5e-8, 0.5, 0.1),
                }
            )
        )
    snp_causal = pd.concat(snps, ignore_index=True)

    tables = {
        "test_evidence": test_evidence,
        "dataset_reliable": dataset_reliable,
        "indication_success": indication_success,
        "snp_causal": snp_causal,
        "drug_target": drug_target,
        "drug_approved": drug_approved,
        "dataset_organ": dataset_organ,
    }
    for name, df in tables.items():
        columns = {**UNCERTAIN, **DETERMINISTIC}[name]
        df = df[columns].reset_index(drop=True)
        key = [c for c in columns if c != "prob"]
        assert not df.duplicated(key).any(), name
        assert not df.isna().any().any(), name
        tables[name] = df
    return tables


def main():
    DATA.mkdir(exist_ok=True)
    for name, df in build().items():
        df.to_parquet(DATA / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows")
        if "prob" in df:
            print(
                "   prob:",
                df.prob.round(4).value_counts().sort_index().to_dict(),
            )
        if name == "dataset_organ":
            print("   organ:", df.organ.value_counts().to_dict())


if __name__ == "__main__":
    main()
