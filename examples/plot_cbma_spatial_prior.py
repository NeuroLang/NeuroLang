# %%
"""
Example where a spatial prior is defined based on the distance between voxels \
and foci in a coordinate-based meta-analysis database
============================================================================

Each voxel's probability of being reported by a study is calculated based on
whether that particular study reports a focus (peak activation) near the voxel.
The probability is defined based on how far from the focus that voxel happens
to be.

"""
from typing import Callable, Iterable

import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd

from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay, NeurolangPDL

# ##############################################################################
# Data preparation
# ----------------

# ##############################################################################
# Load the MNI Gray Matter mask and resample it to 2mm voxels

mni_mask = nilearn.image.resample_img(
    nibabel.load(nilearn.datasets.fetch_icbm152_2009()["gm"]),
    np.eye(3) * 2
)

# ##############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = NeurolangPDL()


# ##############################################################################
# Adding new aggregation function to build a region overlay
# ----------------------------------

@nl.add_symbol
def agg_create_region_overlay_MNI(
    x: Iterable, y: Iterable, z: Iterable, p: Iterable
) -> ExplicitVBR:
    voxels = nibabel.affines.apply_affine(
        np.linalg.inv(mni_mask.affine),
        np.c_[x, y, z]
    )
    return ExplicitVBROverlay(
        voxels, mni_mask.affine, p, image_dim=mni_mask.shape
    )


# ##############################################################################
# Loading the database
# ----------------------------------

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
ns_database = ns_database[["x", "y", "z", "id"]]

ns_features = pd.read_csv(ns_features_fn, sep="\t")
ns_docs = ns_features[["pmid"]].drop_duplicates()
ns_terms = pd.melt(
    ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
).query("TfIdf > 1e-3")[["term", "pmid"]]

TermInStudy = nl.add_tuple_set(ns_terms, name="TermInStudy")
FocusReported = nl.add_tuple_set(ns_database, name="FocusReported")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    ns_docs, name="SelectedStudy"
)
Voxel = nl.add_tuple_set(
    nibabel.affines.apply_affine(
        mni_mask.affine,
        np.transpose(mni_mask.get_fdata().nonzero())
    ),
    name='Voxel'
)

# ##############################################################################
# Probabilistic program and querying
# ----------------------------------

nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])

with nl.scope as e:
    (e.VoxelReported @ e.exp(-e.d / 5.0))[e.x1, e.y1, e.z1, e.s] = (
        e.FocusReported(e.x2, e.y2, e.z2, e.s)
        & e.Voxel(e.x1, e.y1, e.z1)
        & (e.d == e.EUCLIDEAN(e.x1, e.y1, e.z1, e.x2, e.y2, e.z2))
        & (e.d < 4)
    )
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Activation[e.x, e.y, e.z] = (
        e.SelectedStudy[e.s] & e.VoxelReported[e.x, e.y, e.z, e.s]
    )
    e.probmap[e.x, e.y, e.z, e.PROB[e.x, e.y, e.z]] = (
        e.Activation[e.x, e.y, e.z]
    ) // e.TermAssociation["emotion"]
    e.img[e.agg_create_region_overlay_MNI[e.x, e.y, e.z, e.p]] = e.probmap[
        e.x, e.y, e.z, e.p
    ]
    img_query = nl.query((e.x,), e.img[e.x])

# ##############################################################################
# Plotting results
# --------------------------------------------

result_image = img_query.fetch_one()[0].spatial_image()
img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95),
    display_mode='y'
)
nilearn.plotting.show()
