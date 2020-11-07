import os
from typing import Iterable, Tuple, Union

import neurosynth
import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd

from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay, NeurolangPDL

###############################################################################
# Data preparation
# ----------------

###############################################################################
# Load the MNI atlas and resample it to 4mm voxels

mni_t1 = nibabel.load(nilearn.datasets.fetch_icbm152_2009()["t1"])
mni_t1_4mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 4)

###############################################################################
# Define a function that transforms TFIDF features to probabilities


def tfidf_to_probability(
    tfidf: Union[float, np.array],
    alpha: float = 3000,
    tau: float = 0.01,
) -> Union[float, np.array]:
    """
    Threshold TFIDF features to interpret them as probabilities using a sigmoid
    function.

    The formula for this function is

        omega(x ; alpha, tau) = sigma(alpha * (x - tau))

    where sigma is the sigmoid function.

    Parameters
    ----------
    tfidf : float or np.array of floats
        TFIDF (Term Frequency Inverse Document Frequency) features.

    alpha : float
        Parameter used to control the smoothing of the thresholding by the
        sigmoid curve. The larger the value of alpha, the smoother the
        resulting values. The smaller the value of alpha, the closer this
        function will be to a hard-thresholding 1[x > tau].

    tau : float
        Threshold at which the function is centered.

    Returns
    -------
    float or np.array of floats
        Thresholded values between 0 and 1 that can be interpreted as
        probabilities in a probabilistic model.

    """
    return 1 / (1 + np.exp(-alpha * (tfidf - tau)))


term_1 = "memory"
term_2 = "auditory"
terms = [term_1, term_2]

###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = NeurolangPDL()

###############################################################################
# Adding new aggregation function to build a region overlay


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
    )


###############################################################################
# Loading the database

ns_database_fn, ns_features_fn = nilearn.datasets.utils._fetch_files(
    "neurolang/frontend/neurosynth_data",
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

ns_database = pd.read_csv(ns_database_fn, sep=f"\t")
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
ns_database["i"] = ijk_positions[:, 0]
ns_database["j"] = ijk_positions[:, 1]
ns_database["k"] = ijk_positions[:, 2]
ns_database = set(
    ns_database[["i", "j", "k", "id"]].itertuples(name=None, index=False)
)

ns_features = pd.read_csv(ns_features_fn, sep=f"\t")
ns_docs = ns_features[["pmid"]].drop_duplicates()
ns_terms = pd.melt(
    ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
)
ns_terms = ns_terms.loc[ns_terms.term.isin(["memory", "auditory"])]
ns_terms["prob"] = tfidf_to_probability(ns_terms["TfIdf"])
ns_terms = set(
    ns_terms[["prob", "term", "pmid"]].itertuples(name=None, index=False)
)
TermInStudy = nl.add_probabilistic_facts_from_tuples(
    ns_terms, name="TermInStudy"
)
VoxelReported = nl.add_tuple_set(ns_database, name="VoxelReported")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs.values, name="SelectedStudy"
)

###############################################################################
# Probabilistic program and querying

with nl.environment as e:
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Activation[e.i, e.j, e.k] = (
        e.SelectedStudy[e.s] & e.VoxelReported[e.i, e.j, e.k, e.s]
    )
    e.probmap[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.Activation[e.i, e.j, e.k]
    ) // (e.TermAssociation["auditory"] & e.TermAssociation["memory"])
    e.img[e.agg_create_region_overlay[e.i, e.j, e.k, e.p]] = e.probmap[
        e.i, e.j, e.k, e.p
    ]
    img_query = nl.query((e.x,), e.img[e.x])

###############################################################################
# Plotting results
# --------------------------------------------

result_image = img_query.fetch_one()[0].spatial_image()
img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
