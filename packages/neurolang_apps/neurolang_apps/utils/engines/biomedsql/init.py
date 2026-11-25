"""
BiomedSQL engine initialization.

BiomedSQL (NIH-CARD, https://huggingface.co/datasets/NIH-CARD/BiomedSQL,
licence openrail++) is a text-to-SQL benchmark over tables of the genetics of
neurodegenerative diseases and of drugs. This module reads five of its
parquet tables, with all their columns, under BiomedSQL's own column names and
in their order, from ``{data_dir}/biomedsql/db/`` (downloaded there when
missing), into relations:

* ``pd_gwas``: the Parkinson's disease GWAS meta-analysis of Nalls et al.
  2019 (without 23andMe), 7.8 million variants.
* ``ad_gwas``: a combined Alzheimer's disease GWAS, 21 million variants, with
  the same columns as ``pd_gwas``.
* ``smr``: Mendelian-randomisation (SMR) tests of whether a gene's expression
  or methylation mediates a disease (AD, PD, ALS, FTD, LBD, PSP), one per QTL
  dataset, gene probe and disease.
* ``drug``: drug annotations, one row per drug.
* ``drug_indication``: a drug's targets and the diseases it is used or tested
  for.

Missing text is the empty string; missing numbers stay NaN, which compares
false; the booleans (drugIsApproved...) are the text 'True' and 'False'.
The tables together take about 8.5 GB of memory.

And the evidence of the SMR table as tokens, each QTL dataset an equally
likely unit of evidence, as studies are in the neuroimaging engines:

* ``qtl_dataset(qtl_dataset, tissue, kind, organ)``: the 28 QTL datasets;
  organ is brain, blood, spinal cord, cerebrospinal fluid, liver, muscle or
  nerve.
* ``qtl_tested(qtl_dataset, gene, disease)``: the dataset tested the gene.
* ``qtl_significant(qtl_dataset, gene, disease)``: and the test passed
  BiomedSQL's threshold, p_SMR_multi < 2.95e-6 (any probe of the gene).
* ``selected_qtl(qtl_dataset)``: a uniform probabilistic choice of one QTL
  dataset.

And the same evidence with a prior, two imaginary units of evidence, as the
``prior_significant`` and ``prior_none`` options say how much each weighs
against one dataset: ``prior_significant`` is tested and significant for
every (gene, disease) pair, ``prior_none`` is tested and never significant.

* ``evidence_unit(unit)``: the 28 datasets and the two imaginary units.
* ``unit_tested(unit, gene, disease)``, ``unit_significant(unit, gene,
  disease)``: the tests of the datasets, and those of the imaginary units.
* ``selected_unit(unit)``: a probabilistic choice of one unit, a dataset with
  weight 1, the imaginary units with their weights (normalised).

And the same tests as graded evidence, as strong as their p-value:

* ``unit_evidence(unit, gene, disease)``: a probabilistic fact for each test of
  ``unit_tested``, with the probability that the ``evidence_strength`` option
  gives to the -log10 p_SMR_multi of its strongest probe; ``prior_significant``
  is evidence with probability 1, ``prior_none`` is not evidence.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
URL = (
    "https://huggingface.co/datasets/NIH-CARD/BiomedSQL/resolve/main/"
    "db_data/{}.parquet"
)
SMR_THRESHOLD = 2.95e-6

TABLES = {
    "pd_gwas": "ParkinsonDisease_CompleteGeneData_No23andMe",
    "ad_gwas": "AlzheimerDisease_CombinedGeneData_UUID",
    "smr": "NeurodegenerativeDiseases_SMR_Genes_Full",
    "drug": "DrugGeneTargets_ComprehensiveAnnotations_updated",
    "drug_indication": "DrugTargets_IndicationsAndTherapeuticUses",
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


def _table(db: Path, source: str) -> pd.DataFrame:
    path = db / f"{source}.parquet"
    if not path.exists():
        db.mkdir(parents=True, exist_ok=True)
        LOG.info("Downloading %s from BiomedSQL", source)
        urllib.request.urlretrieve(URL.format(source), path)
    df = pd.read_parquet(path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)
        elif df[col].dtype == bool:     # no True and False in Datalog
            df[col] = df[col].map({True: "True", False: "False"})
    return df


PRIOR_SIGNIFICANT = "prior_significant"
PRIOR_NONE = "prior_none"


def strength_probability(p_values, evidence_strength):
    """The probability of evidence of tests of these p-values:
    ``evidence_strength`` lists (lowest -log10 p, probability) pairs in
    increasing order; below the first, the probability is 0."""
    lows, probs = zip(*evidence_strength)
    strength = -np.log10(np.clip(np.asarray(p_values, float), 1e-300, 1))
    i = np.searchsorted(np.asarray(lows, float), strength, side="right")
    return np.concatenate([[0.0], probs])[i]


def init_engine(
    nl,
    mask,
    data_dir: Path,
    prior_significant: float = 0.01,
    prior_none: float = 0.5,
    evidence_strength=((0.0, 0.0025), (5.53, 1.0)),
) -> None:
    db = Path(data_dir) / "biomedsql" / "db"
    frames = {name: _table(db, source) for name, source in TABLES.items()}
    for name, df in frames.items():
        nl.add_tuple_set(df, name=name)

    smr = frames["smr"]
    tests = (
        smr.groupby(["Omic", "Gene", "Disease"], observed=True)
        .p_SMR_multi.min().reset_index()
    )
    nl.add_tuple_set(tests[["Omic", "Gene", "Disease"]], name="qtl_tested")
    nl.add_tuple_set(
        tests.loc[
            tests.p_SMR_multi < SMR_THRESHOLD, ["Omic", "Gene", "Disease"]
        ],
        name="qtl_significant",
    )
    raw = smr[["Omic", "Omic_tissue", "Omic_type"]].drop_duplicates("Omic")
    raw = raw.astype(str)
    raw["organ"] = [organ_of(t) for t in raw.Omic_tissue]
    nl.add_tuple_set(
        raw.itertuples(index=False, name=None), name="qtl_dataset"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(q,) for q in sorted(raw.Omic)], name="selected_qtl"
    )

    pairs = tests[["Gene", "Disease"]].drop_duplicates()
    imaginary = {}
    for unit in (PRIOR_SIGNIFICANT, PRIOR_NONE):
        imaginary[unit] = pd.DataFrame(
            {"Omic": unit, "Gene": pairs.Gene, "Disease": pairs.Disease}
        )
    significant = tests.loc[
        tests.p_SMR_multi < SMR_THRESHOLD, ["Omic", "Gene", "Disease"]
    ]
    nl.add_tuple_set(
        pd.concat(
            [tests[["Omic", "Gene", "Disease"]], *imaginary.values()]
        ),
        name="unit_tested",
    )
    nl.add_tuple_set(
        pd.concat([significant, imaginary[PRIOR_SIGNIFICANT]]),
        name="unit_significant",
    )
    weights = {q: 1.0 for q in sorted(raw.Omic)}
    weights[PRIOR_SIGNIFICANT] = float(prior_significant)
    weights[PRIOR_NONE] = float(prior_none)
    total = sum(weights.values())
    nl.add_tuple_set([(u,) for u in weights], name="evidence_unit")
    nl.add_probabilistic_choice_from_tuples(
        [(w / total, u) for u, w in weights.items()], name="selected_unit"
    )
    evidence = tests[["Omic", "Gene", "Disease"]].assign(
        probability=strength_probability(tests.p_SMR_multi, evidence_strength)
    )
    evidence = pd.concat([
        evidence[evidence.probability > 0],
        imaginary[PRIOR_SIGNIFICANT].assign(probability=1.0),
    ])
    nl.add_probabilistic_facts_from_tuples(
        evidence[["probability", "Omic", "Gene", "Disease"]].itertuples(
            index=False, name=None
        ),
        name="unit_evidence",
    )
