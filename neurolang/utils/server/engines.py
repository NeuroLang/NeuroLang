from pathlib import Path
from typing import Iterable
from neurolang.frontend import NeurolangPDL
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image


def load_neurosynth_data(nl):
    data_dir = Path("neurolang_data")
    mni_t1 = nib.load(
        datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
    )
    mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)

    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        data_dir / "neurosynth",
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )

    ns_database = pd.read_csv(ns_database_fn, sep="\t")
    ijk_positions = np.round(
        nib.affines.apply_affine(
            np.linalg.inv(mni_t1_4mm.affine),
            ns_database[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    ns_database["i"] = ijk_positions[:, 0]
    ns_database["j"] = ijk_positions[:, 1]
    ns_database["k"] = ijk_positions[:, 2]

    ns_features = pd.read_csv(ns_features_fn, sep="\t")
    ns_docs = ns_features[["pmid"]].drop_duplicates()
    ns_terms = pd.melt(
        ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
    ).query("TfIdf > 1e-3")[["pmid", "term"]]

    activations = nl.add_tuple_set(ns_database, name="activations")
    terms = nl.add_tuple_set(ns_terms, name="terms")
    docs = nl.add_uniform_probabilistic_choice_over_set(ns_docs, name="docs")


def init_frontend():
    """
    Create a Neurolang Probabilistic engine and add some aggregation methods.

    Returns
    -------
    NeurolangPDL
        the Neurolang engine
    """
    nl = NeurolangPDL()

    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    return nl
