# -*- coding: utf-8 -*-
"""
Two-term conjunctive Coordinate-Based Meta-Analysis (CBMA) forward inference \
with relaxed TFIDF thresholding on the Neurosynth database
============================================================================

This example first shows how to encode the Neurosynth Coordinate-Based
Meta-Analysis (CBMA) database in NeuroLang, with probabilistic term-to-study
associations using a sigmoid-based soft thresholding of TFIDF features.

It then shows how a two-term conjunctive query can be expressed and solved to
obtain an uncorreted forward inference map for studies associated with both
terms 'stimulus' and 'outcome'.
"""
from typing import Callable, Iterable

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
# Register NumPy's exponential function so we can use it to efficiently map \
# TFIDF features to probabilities

nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])


###############################################################################
# Loading the database

ns_database_fn, ns_features_fn = nilearn.datasets.utils._fetch_files(
    "neurolang",
    [
        (
            "database.txt",
            "https://github.com/neurosynth/neurosynth-data"
            "/raw/master/current_data.tar.gz",
            {"uncompress": True},
        ),
        (
            "features.txt",
            "https://github.com/neurosynth/neurosynth-data"
            "/raw/master/current_data.tar.gz",
            {"uncompress": True},
        ),
    ],
)

ns_database = pd.read_csv(ns_database_fn, sep="\t")
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
ns_database["i"] = ijk_positions[:, 0]
ns_database["j"] = ijk_positions[:, 1]
ns_database["k"] = ijk_positions[:, 2]
ns_database = ns_database[["i", "j", "k", "id"]]

ns_features = pd.read_csv(ns_features_fn, sep="\t")
ns_docs = ns_features[["pmid"]].drop_duplicates()
ns_tfidf = pd.melt(
    ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
).query("TfIdf > 0")[["pmid", "term", "TfIdf"]]

StudyTFIDF = nl.add_tuple_set(ns_tfidf, name="StudyTFIDF")
VoxelReported = nl.add_tuple_set(ns_database, name="VoxelReported")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs, name="SelectedStudy"
)

###############################################################################
# Probabilistic program and querying
#
# Compute a forward inference map for studies associated to both terms
# 'stimulus' and 'outcome'. In [1]_, a CBMA is carried out too "look for areas
# that exhibit reliable correlations with stimulus value at the time of outcome
# across studies". Only "contrasts that looked at parametric measures of
# subjective value during the outcome or reward consumption phase" are included
# in this meta-analysis. The ALE analysis identified two clusters: VMPFC/OFC
# and bilateral VSTR.
#
# .. [1] Clithero, John A., and Antonio Rangel. 2014. ‘Informatic Parcellation
#    of the Network Involved in the Computation of Subjective Value’. Social
#    Cognitive and Affective Neuroscience 9 (9): 1289–1302.
#    https://doi.org/10.1093/scan/nst106.

with nl.environment as e:
    (e.TermInStudy @ (1 / (1 + e.exp(-e.alpha * (e.tfidf - e.tau)))))[
        e.t, e.s
    ] = (e.StudyTFIDF[e.s, e.t, e.tfidf] & (e.alpha == 300) & (e.tau == 0.1))
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Activation[e.i, e.j, e.k] = (
        e.SelectedStudy[e.s] & e.VoxelReported[e.i, e.j, e.k, e.s]
    )
    e.probmap[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.Activation[e.i, e.j, e.k]
    ) // (e.TermAssociation["stimulus"] & e.TermAssociation["outcome"])
    e.img[e.agg_create_region_overlay[e.i, e.j, e.k, e.p]] = e.probmap[
        e.i, e.j, e.k, e.p
    ]
    img_query = nl.query((e.x,), e.img[e.x])

###############################################################################
# Plotting results
# --------------------------------------------

result_image = img_query.fetch_one()[0].spatial_image()
img = result_image.get_fdata()
nilearn.plotting.plot_stat_map(
    result_image,
    threshold=np.percentile(img[img > 0], 90),
    cut_coords=(-6,),
    display_mode="x",
    title="Expect vPCC and dPCC",
)
nilearn.plotting.plot_stat_map(
    result_image,
    threshold=np.percentile(img[img > 0], 90),
    cut_coords=(-8,),
    display_mode="y",
    title="Expect VSTR",
)
nilearn.plotting.plot_stat_map(
    result_image,
    threshold=np.percentile(img[img > 0], 90),
    cut_coords=(4,),
    display_mode="x",
    title="Expect VMPFC",
)
nilearn.plotting.plot_stat_map(
    result_image,
    threshold=np.percentile(img[img > 0], 90),
    cut_coords=(-6,),
    display_mode="z",
    title="Expect VMPFC",
)
nilearn.plotting.show()
