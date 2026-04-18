# -*- coding: utf-8 -*-
r"""
CBMA Spatial Prior in SQUALL Controlled English
================================================

Reproduces the coordinate-based meta-analysis spatial decay prior — each
voxel's probability of being reported by a study is weighted by
``max(exp(−d/5))`` over nearby foci — using `SQUALL controlled natural
language <https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the
IR builder DSL.

A small Python helper builds the ``near_focus`` proximity table
(voxel × study pairs within 1 voxel of a reported focus together with their
``exp(−d/5)`` proximity values) before the SQUALL program runs.  The remaining
five logic rules are then expressed as plain English sentences passed to
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`.  Compare with
:ref:`sphx_glr_auto_examples_plot_cbma_spatial_prior.py` which writes the same
computation using ``with nl.environment as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Voxel_reported every Max_proximity of the Near_focus
        per ?i1 and per ?j1 and per ?k1 and per ?s.

    define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
        such that ?s is a Selected_study.

    define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Probmap with probability every Activation (?i; ?j; ?k)
        conditioned to every Term_association ?t that is 'emotion'.

    define as Img every Agg_create_region_overlay of the Probmap.
"""

# %%
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Iterable

import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd
from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay, NeurolangPDL
from neurolang.frontend.neurosynth_utils import get_ns_mni_peaks_reported

# %%
###############################################################################
# Data preparation
# ----------------
# Set the data directory and load the MNI T1 atlas resampled to 2 mm voxels.

data_dir = Path.home() / "neurolang_data"

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

# %%
###############################################################################
# Set up engine and register aggregation symbol
# ----------------------------------------------
# ``max_proximity`` folds a column of ``exp(−d/5)`` proximity values into
# their maximum.  ``agg_create_region_overlay`` aggregates the final result
# into a brain overlay image.

nl = NeurolangPDL()


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_2mm.affine, p, image_dim=mni_t1_2mm.shape
    )


@nl.add_symbol
def max_proximity(values: Iterable) -> float:
    """Aggregation: max over a collection of exp(-d/5) proximity values."""
    return float(max(values))


# %%
###############################################################################
# Load the NeuroSynth database
# ----------------------------
# All relation names are lowercase to match SQUALL's case-folding convention.

peak_data = get_ns_mni_peaks_reported(data_dir)
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_2mm.affine),
        peak_data[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
peak_data["i"] = ijk_positions[:, 0]
peak_data["j"] = ijk_positions[:, 1]
peak_data["k"] = ijk_positions[:, 2]
peak_data = peak_data[["i", "j", "k", "id"]]

study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)

# %%
###############################################################################
# Build the ``near_focus`` proximity table (Python pre-computation)
# ------------------------------------------------------------------
# SQUALL's grammar supports only a single subject noun phrase per rule, so a
# two-noun cross-join (Voxel × Focus_reported with a distance filter) cannot be
# expressed as a SQUALL sentence.  The proximity table is therefore pre-computed
# here in Python.  For every voxel ``(i1,j1,k1)`` and study ``s``, it records
# ``exp(−d/5)`` for each reported focus ``(i2,j2,k2)`` within strictly less than
# 1 voxel distance ``d``.  This is the spatial-decay weight used in the
# coordinate-based meta-analysis model.

shape = mni_t1_2mm.get_fdata().shape
voxel_ijk = np.array(
    list(np.ndindex(*shape)), dtype=np.int32
)  # (N_voxels, 3)

foci_ijk = peak_data[["i", "j", "k"]].values.astype(np.int32)
foci_study = peak_data["id"].values

rows = []
for idx, (fi, fj, fk) in enumerate(foci_ijk):
    dists = np.linalg.norm(voxel_ijk - np.array([fi, fj, fk]), axis=1)
    mask = dists < 1.0
    prox = np.exp(-dists[mask] / 5.0)
    for (vi, vj, vk), p in zip(voxel_ijk[mask], prox):
        rows.append((int(vi), int(vj), int(vk), int(foci_study[idx]), float(p)))

near_focus_df = pd.DataFrame(rows, columns=["i1", "j1", "k1", "s", "proximity"])
nl.add_tuple_set(near_focus_df, name="near_focus")

# %%
###############################################################################
# SQUALL controlled-English program
# ----------------------------------
# Five sentences replace the ``with nl.environment as e:`` block.
#
# Sentence 1  ``every Max_proximity of the Near_focus per ?i1 and per …``
#             — arbitrary-functor aggregation; four ``per`` dims (joined with
#             ``and``) define the groupby key ``(i1, j1, k1, s)``.
# Sentences 2–3  ``such that ?s is a Selected_study`` — existential study
#             filter.  ``?tfidf`` binds the TF-IDF weight column but is
#             projected away (absent from the rule head); SQUALL requires every
#             argument in a tuple label to be a named variable — anonymous
#             wildcards are not allowed in that position.
# Sentence 4  ``with probability … conditioned to … that is 'emotion'`` — MARG
#             query.
# Sentence 5  ``every Agg_create_region_overlay of the Probmap`` — brain image
#             aggregation.

squall_program = """
define as Voxel_reported every Max_proximity of the Near_focus
    per ?i1 and per ?j1 and per ?k1 and per ?s.

define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap.
"""

nl.execute_squall_program(squall_program)

# %%
###############################################################################
# Solve and retrieve the probability map
# --------------------------------------

solution = nl.solve_all()
result_image = (
    solution["img"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)

# %%
###############################################################################
# Plot
# ----

img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
