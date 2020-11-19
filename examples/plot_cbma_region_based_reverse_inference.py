# -*- coding: utf-8 -*-
"""
Region-Based Reverse Inference on the Neurosynth Coordinate-Based \
Meta-Analysis Database
============================================================================

Reported peak activation coordinates (foci) within the Neurosynth database are
used to model the association between neuroanatomical regions and neuroimaging
studies in a probabilistic NeuroLang program. Associations between terms and
studies, based on TFIDF features in the database, are also encoded in the
probabilistic program. Queries are solved to find the most probable term
associations for a pattern of region activations. This process of finding the
most probable term associations given a pattern of activations is sometimes
called "reverse inference".

"""
import pathlib
import typing

import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd

import neurolang.regions
from neurolang.frontend import NeurolangPDL

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
# Load the Destrieux example from nilearn and add it to the program

atlas_destrieux = nilearn.datasets.fetch_atlas_destrieux_2009()
atlas_labels = {
    label: str(name.decode("utf8"))
    for label, name in atlas_destrieux["labels"]
}
atlas_spatial_image = nibabel.load(atlas_destrieux["maps"])
atlas_spatial_image = nilearn.image.resample_to_img(
    atlas_spatial_image, mni_t1_4mm
)
dfs = list()
for label, name in atlas_destrieux["labels"]:
    voxels = np.transpose(
        (np.asanyarray(atlas_spatial_image.dataobj) == label).nonzero()
    )
    df = pd.DataFrame(voxels, columns=["i", "j", "k"])
    df["region_name"] = str(name.decode("utf-8"))
    dfs.append(df[["region_name", "i", "j", "k"]])
nl.add_tuple_set(pd.concat(dfs).values, name="AnatomicalRegion")

###############################################################################
# Loading the database


def load_neurosynth_data():
    cache_store_path = pathlib.Path("_cached_neurosynth.h5")
    if cache_store_path.is_file():
        with pd.HDFStore(cache_store_path) as store:
            ns_docs = store["docs"]
            ns_terms = store["terms"]
            ns_activations = store["activations"]
        return ns_activations.values, ns_terms.values, ns_docs.values
    ns_database_fn, ns_features_fn = nilearn.datasets.utils._fetch_files(
        "neurolang/frontend/neurosynth_data",
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

    ns_activations = pd.read_csv(ns_database_fn, sep="\t")
    ijk_positions = np.round(
        nibabel.affines.apply_affine(
            np.linalg.inv(mni_t1_4mm.affine),
            ns_activations[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    ns_activations["i"] = ijk_positions[:, 0]
    ns_activations["j"] = ijk_positions[:, 1]
    ns_activations["k"] = ijk_positions[:, 2]
    ns_activations = ns_activations[["i", "j", "k", "id"]]

    ns_features = pd.read_csv(ns_features_fn, sep="\t")
    ns_terms = pd.melt(
        ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
    ).query("TfIdf > 1e-3")[["term", "pmid"]]
    ns_docs = ns_features[["pmid"]].drop_duplicates()

    if cache_store_path.is_file():
        cache_store_path.unlink()

    with pd.HDFStore(cache_store_path) as store:
        store["docs"] = ns_docs
        store["terms"] = ns_terms
        store["activations"] = ns_activations

    return ns_activations.values, ns_terms.values, ns_docs.values


ns_activations, ns_terms, ns_docs = load_neurosynth_data()
TermInStudy = nl.add_tuple_set(ns_terms, name="TermInStudy")
FocusReported = nl.add_tuple_set(ns_activations, name="FocusReported")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs, name="SelectedStudy"
)


@nl.add_symbol
def agg_count(
    i: typing.Iterable,
    j: typing.Iterable,
    k: typing.Iterable,
) -> int:
    return len(i)


###############################################################################
# Probabilistic program and querying

with nl.environment as e:
    e.RegionVolume[e.r, e.agg_count[e.i, e.j, e.k]] = e.AnatomicalRegion[
        e.r, e.i, e.j, e.k
    ]
    e.StudyRegionIntersectionVolume[e.s, e.r, e.agg_count[e.i, e.j, e.k]] = (
        e.FocusReported[e.i, e.j, e.k, e.s]
        & e.AnatomicalRegion[e.r, e.i, e.j, e.k]
    )
    (e.RegionReported @ (e.v1 / e.v2))[e.r, e.s] = (
        e.StudyRegionIntersectionVolume[e.s, e.r, e.v1]
        & e.RegionVolume[e.r, e.v2]
    )
    e.RegionActivation[e.r] = e.RegionReported[e.r, e.s] & e.SelectedStudy[e.s]
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Query[e.t, e.r, e.PROB[e.t, e.r]] = e.TermAssociation[e.t] // (
        e.RegionActivation[e.r]
    )
    result = nl.query((e.t, e.r, e.p), e.Query[e.t, e.r, e.p])
