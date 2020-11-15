# -*- coding: utf-8 -*-
r"""
NeuroLang Example based Implementing a NeuroSynth Query
====================================================

"""


from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image, plotting

from neurolang import ExplicitVBROverlay, NeurolangPDL

###############################################################################
# Data preparation
# ----------------

###############################################################################
# Load the MNI atlas and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()["t1"])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)

###############################################################################
# Load the NeuroSynth database

ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    "neurolang",
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
ns_terms.to_csv("term_documents.csv")
(
    ns_database[["x", "y", "z", "i", "j", "k", "id"]]
    .rename(columns={"id": "pmid"})
    .to_csv("document_activations.csv")
)


###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = NeurolangPDL()


###############################################################################
# Adding new aggregation function to build a region overlay
@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBROverlay:
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
    )


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name="activations")
terms = nl.add_tuple_set(ns_terms.values, name="terms")
docs = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs.values, name="docs"
)


###############################################################################
# Probabilistic program and querying

with nl.scope as e:
    e.vox_activation[e.i, e.j, e.k, e.d] = e.activations[
        e.d,
        ...,
        ...,
        ...,
        ...,
        "MNI",
        ...,
        ...,
        ...,
        ...,
        ...,
        ...,
        ...,
        e.i,
        e.j,
        e.k,
    ]
    e.probmap[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.vox_activation[e.i, e.j, e.k, e.d]
    ) // e.terms[e.d, "auditory"]
    e.img[e.agg_create_region_overlay[e.i, e.j, e.k, e.p]] = e.probmap[
        e.i, e.j, e.k, e.p
    ]
    img_query = nl.query((e.x,), e.img(e.x))


###############################################################################
# Plotting results
# --------------------------------------------

result_image = img_query.fetch_one()[0].spatial_image()
img = result_image.get_fdata()
plot = plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
plotting.show()
