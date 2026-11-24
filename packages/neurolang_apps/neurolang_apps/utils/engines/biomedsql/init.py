"""
BiomedSQL engine initialization.

BiomedSQL (NIH-CARD, https://huggingface.co/datasets/NIH-CARD/BiomedSQL,
licence openrail++) is a text-to-SQL benchmark over tables of the genetics of
neurodegenerative diseases and of drugs. This module reads four of its
parquet tables, from ``{data_dir}/biomedsql/db/`` (downloaded there when
missing), into relations:

* ``pd_gwas(uuid, snp, freq, b, se, p, chr, bp, nearest_gene)``: the
  Parkinson's disease GWAS meta-analysis of Nalls et al. 2019 (without
  23andMe), positions in GRCh38.
* ``smr(uuid, qtl_dataset, disease, gene, top_snp, p_smr_multi, p_heidi)``:
  Mendelian-randomisation (SMR) tests of whether a gene's expression or
  methylation mediates a disease (AD, PD, ALS, FTD, LBD, PSP), one per QTL
  dataset, gene probe and disease.
* ``drug(uuid, chembl_id, name, trade_names, synonyms, molecular_type,
  action_type, mechanism, year_approved, max_phase, approved, black_box)``.
* ``drug_indication(uuid, name, trade_names, synonyms, target, mesh_heading,
  efo_term, max_phase)``: a drug's targets and the diseases it is used or
  tested for, with the highest clinical-trial phase for that indication.

And the evidence of the SMR table as tokens, each QTL dataset an equally
likely unit of evidence, as studies are in the neuroimaging engines:

* ``qtl_dataset(q, tissue, kind, organ)``: the 28 QTL datasets; organ is
  brain, blood, spinal cord, cerebrospinal fluid, liver, muscle or nerve.
* ``qtl_tested(q, gene, disease)``: q tested the gene for the disease.
* ``qtl_significant(q, gene, disease)``: and the test passed BiomedSQL's
  threshold, p_smr_multi < 2.95e-6 (any probe of the gene).
* ``selected_qtl(q)``: a uniform probabilistic choice of one QTL dataset.
"""

import logging
import urllib.request
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)
URL = (
    "https://huggingface.co/datasets/NIH-CARD/BiomedSQL/resolve/main/"
    "db_data/{}.parquet"
)
SMR_THRESHOLD = 2.95e-6

TABLES = {
    "pd_gwas": ("ParkinsonDisease_CompleteGeneData_No23andMe", {
        "UUID": "uuid", "SNP": "snp", "freq": "freq", "b": "b", "se": "se",
        "p": "p", "chr_38": "chr", "bp_38": "bp",
        "nearestGene": "nearest_gene"}),
    "smr": ("NeurodegenerativeDiseases_SMR_Genes_Full", {
        "UUID": "uuid", "Omic": "qtl_dataset", "Disease": "disease",
        "Gene": "gene", "topRSID": "top_snp", "p_SMR_multi": "p_smr_multi",
        "p_HEIDI": "p_heidi"}),
    "drug": ("DrugGeneTargets_ComprehensiveAnnotations_updated", {
        "UUID": "uuid", "chemblIdentifier": "chembl_id", "drugName": "name",
        "tradeNames_string": "trade_names", "drugSynonyms_string": "synonyms",
        "drugMolecularType": "molecular_type", "drugActionType": "action_type",
        "drugMechanismOfAction": "mechanism",
        "yearOfFirstApproval": "year_approved",
        "maxClinicalTrialPhase": "max_phase", "drugIsApproved": "approved",
        "blackBoxWarning": "black_box"}),
    "drug_indication": ("DrugTargets_IndicationsAndTherapeuticUses", {
        "UUID": "uuid", "drugName": "name", "tradeNames_list": "trade_names",
        "syns_list": "synonyms", "approvedSymbol": "target",
        "mesh_heading": "mesh_heading", "efo_term": "efo_term",
        "max_phase_for_ind": "max_phase"}),
}


ORGANS = {
    "blood": ("whole blood",),
    "spinal cord": ("spinalcord", "spinal cord"),
    "cerebrospinal fluid": ("cerebral spinal fluid",),
    "liver": ("liver",),
    "muscle": ("muscle",),
    "nerve": ("tibial nerve",),
}


def organ_of(tissue: str) -> str:
    """The organ of a QTL dataset's tissue: brain unless it is one of
    ORGANS (every other tissue of the SMR table is a brain region)."""
    t = tissue.lower()
    for organ, names in ORGANS.items():
        if any(n in t for n in names):
            return organ
    return "brain"


def _table(db: Path, source: str, columns: dict) -> pd.DataFrame:
    path = db / f"{source}.parquet"
    if not path.exists():
        db.mkdir(parents=True, exist_ok=True)
        LOG.info("Downloading %s from BiomedSQL", source)
        urllib.request.urlretrieve(URL.format(source), path)
    df = pd.read_parquet(path, columns=list(columns)).rename(columns=columns)
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("bool"):
            df[col] = df[col].fillna("").astype(str)
    return df[list(columns.values())]


def init_engine(nl, mask, data_dir: Path) -> None:
    db = Path(data_dir) / "biomedsql" / "db"
    frames = {
        name: _table(db, src, cols) for name, (src, cols) in TABLES.items()
    }
    for name, df in frames.items():
        nl.add_tuple_set(df, name=name)

    smr = frames["smr"]
    tests = (
        smr.groupby(["qtl_dataset", "gene", "disease"], observed=True)
        .p_smr_multi.min().reset_index()
    )
    nl.add_tuple_set(
        tests[["qtl_dataset", "gene", "disease"]], name="qtl_tested"
    )
    nl.add_tuple_set(
        tests.loc[tests.p_smr_multi < SMR_THRESHOLD,
                  ["qtl_dataset", "gene", "disease"]],
        name="qtl_significant",
    )
    raw = pd.read_parquet(
        db / f"{TABLES['smr'][0]}.parquet",
        columns=["Omic", "Omic_tissue", "Omic_type"],
    ).drop_duplicates("Omic").astype(str)
    raw["organ"] = [organ_of(t) for t in raw.Omic_tissue]
    nl.add_tuple_set(
        raw.itertuples(index=False, name=None), name="qtl_dataset"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(q,) for q in sorted(raw.Omic.astype(str))], name="selected_qtl"
    )
