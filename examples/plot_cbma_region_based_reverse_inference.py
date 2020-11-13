# -*- coding: utf-8 -*-
"""
TODO
============================================================================

TODO

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
nl.add_atlas_set(
    "AnatomicalRegion", atlas_labels, nibabel.load(atlas_destrieux["maps"])
)

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

    # limit size
    ns_docs = ns_docs.iloc[:100, :]
    ns_terms = ns_terms.loc[ns_terms.pmid.isin(ns_docs.iloc[:, 0].values)]
    ns_activations = ns_activations.loc[
        ns_activations.id.isin(ns_docs.iloc[:, 0].values)
    ]

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
def agg_create_region(
    i: typing.Iterable,
    j: typing.Iterable,
    k: typing.Iterable,
) -> neurolang.regions.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return neurolang.regions.ExplicitVBR(
        voxels, mni_t1_4mm.affine, image_dim=mni_t1_4mm.shape
    )


@nl.add_symbol
def intersection(
    first: neurolang.regions.ExplicitVBR,
    second: neurolang.regions.ExplicitVBR,
) -> neurolang.regions.ExplicitVBR:
    return neurolang.regions.region_intersection({first, second})


@nl.add_symbol
def volume(region: neurolang.regions.ExplicitVBR) -> float:
    if isinstance(region, neurolang.regions.EmptyRegion):
        return 0.0
    return float(len(region.voxels))


###############################################################################
# Probabilistic program and querying

with nl.environment as e:
    e.StudyRegion[e.s, e.agg_create_region[e.i, e.j, e.k]] = FocusReported[
        e.i, e.j, e.k, e.s
    ]
    (e.RegionReported @ (e.volume[e.intersect] / e.volume[e.r]))[e.l, e.s] = (
        e.AnatomicalRegion[e.l, e.r]
        & e.StudyRegion[e.s, e.sr]
        & (e.intersect == e.intersection[e.r, e.sr])
    )
    e.RegionActivation[e.r] = e.RegionReported[e.r, e.s] & e.SelectedStudy[e.s]
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Query[e.t, e.PROB[e.t]] = e.TermAssociation[e.t] // (
        e.RegionActivation["L G_pariet_inf-Angular"]
    )
    result = nl.query((e.t, e.p), e.Query[e.t, e.p])
