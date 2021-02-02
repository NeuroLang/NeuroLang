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
# Load the MNI gray matter mask and resample it to 2mm voxels

mni_mask = nilearn.image.resample_img(
    nibabel.load(nilearn.datasets.fetch_icbm152_2009()["gm"]), np.eye(3) * 2
)


###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = NeurolangPDL()

###############################################################################
# Adding new aggregation function to build a region overlay


@nl.add_symbol
def agg_create_region_overlay(
    x: Iterable, y: Iterable, z: Iterable, p: Iterable
) -> ExplicitVBR:
    voxels = nibabel.affines.apply_affine(
        np.linalg.inv(mni_mask.affine), np.c_[x, y, z]
    )
    return ExplicitVBROverlay(
        voxels, mni_mask.affine, p, image_dim=mni_mask.shape
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
# only keep coordinates and study PMIDs
ns_database = ns_database[["x", "y", "z", "id"]]

ns_features = pd.read_csv(ns_features_fn, sep="\t")
ns_docs = ns_features[["pmid"]].drop_duplicates()
ns_tfidf = pd.melt(
    ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
).query("TfIdf > 0")[["pmid", "term", "TfIdf"]]

StudyTFIDF = nl.add_tuple_set(ns_tfidf, name="StudyTFIDF")
PeakReported = nl.add_tuple_set(ns_database, name="PeakReported")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs, name="SelectedStudy"
)

Voxel = nl.add_tuple_set(
    nibabel.affines.apply_affine(
        mni_mask.affine, np.transpose(mni_mask.get_fdata().nonzero())
    ),
    name="Voxel",
)

###############################################################################
# Probabilistic program and querying
#
# Compute a forward inference map for studies associated to both terms
# 'stimulus' and 'outcome'. In [1]_, a CBMA is carried out to "look for areas
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

with nl.scope as e:
    e.VoxelReported[e.x, e.y, e.z, e.s] = (
        PeakReported(e.x, e.y, e.z, e.s)
        & Voxel(e.x1, e.y1, e.z1)
        & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x1, e.y1, e.z1))
        & (e.d < 7)
    )
    voxel_reported = nl.query(
        (e.x, e.y, e.z, e.s), e.VoxelReported(e.x, e.y, e.z, e.s)
    )
VoxelReported = nl.add_tuple_set(
    voxel_reported.as_pandas_dataframe(), name="VoxelReported"
)

with nl.scope as e:
    e.Query[e.x, e.y, e.z, e.s] = VoxelReported(e.x, e.y, e.z, e.s) & Voxel(
        e.x, e.y, e.z
    )
    sol = nl.query((e.x, e.y, e.z, e.s), e.Query(e.x, e.y, e.z, e.s))

with nl.scope as e:
    (e.TermInStudy @ (1 / (1 + e.exp(-e.alpha * (e.tfidf - e.tau)))))[
        e.t, e.s
    ] = (e.StudyTFIDF(e.s, e.t, e.tfidf) & (e.alpha == 300) & (e.tau == 1e-3))
    e.StudyMatchQuery[e.s] = e.TermInStudy("language", e.s) & e.TermInStudy(
        "networks", e.s
    )
    e.ProbMap[e.x, e.y, e.z, e.PROB[e.x, e.y, e.z]] = (
        VoxelReported(e.x, e.y, e.z, e.s) & SelectedStudy(e.s)
    ) // (e.StudyMatchQuery(e.s) & SelectedStudy(e.s))
    e.BrainImage[e.agg_create_region_overlay(e.x, e.y, e.z, e.p)] = e.ProbMap(
        e.x, e.y, e.z, e.p
    )
    img_query = nl.query((e.x,), e.BrainImage(e.x))
result_image = img_query.fetch_one()[0].spatial_image()
nilearn.plotting.plot_stat_map(result_image)

###############################################################################
# Plotting results
# --------------------------------------------

result_image = img_query.fetch_one()[0].spatial_image()
img = result_image.get_fdata()
nilearn.plotting.plot_stat_map(
    result_image,
    cut_coords=(-6,),
    display_mode="x",
    title="Expect vPCC and dPCC",
)
nilearn.plotting.plot_stat_map(
    result_image,
    cut_coords=(-8,),
    display_mode="y",
    title="Expect VSTR",
)
nilearn.plotting.plot_stat_map(
    result_image,
    cut_coords=(4,),
    display_mode="x",
    title="Expect VMPFC",
)
nilearn.plotting.plot_stat_map(
    result_image,
    cut_coords=(-6,),
    display_mode="z",
    title="Expect VMPFC",
)
nilearn.plotting.show()
