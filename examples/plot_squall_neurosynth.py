# -*- coding: utf-8 -*-
r"""
NeuroSynth Query in SQUALL Controlled English
=============================================

Reproduces the NeuroSynth forward model — P(Activation | Term = 'auditory') —
using `SQUALL controlled natural language
<https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the IR builder
DSL.

The four logic rules plus an ``obtain`` query are expressed as plain English
sentences and executed via
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`, which
returns the query result directly when an ``obtain`` clause is present. Compare
with :ref:`sphx_glr_auto_examples_plot_neurosynth_implementation.py` which
writes identical rules using ``with nl.scope as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Activation every Peak_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
        such that ?s is a Selected_study.

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association ?t that is 'auditory'.

    define as Activation_given_term_image
        every Agg_create_region_overlay of the Activation_given_term (?i; ?j; ?k; ?p).

    obtain every Activation_given_term_image (?x).
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
from neurolang import ExplicitVBROverlay, NeurolangPDL
from neurolang.frontend.neurosynth_utils import get_ns_mni_peaks_reported

# %%
###############################################################################
# Data preparation
# ----------------
# Set the data directory and load the MNI T1 atlas resampled to 4 mm voxels.

data_dir = Path.home() / "neurolang_data"

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_4mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 4)

# %%
###############################################################################
# Set up the probabilistic engine and register helper symbols
# -----------------------------------------------------------

nl = NeurolangPDL()


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBROverlay:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
    mni_coords = np.c_[i, j, k]
    return ExplicitVBROverlay(
        mni_coords, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
    )


# %%
###############################################################################
# Load the NeuroSynth database
# ----------------------------
# Register peaks, study IDs, and term–study associations as extensional facts.
# Relation names are lowercase so they match SQUALL's case-folding convention.

peak_data = get_ns_mni_peaks_reported(data_dir)
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        peak_data[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
peak_data["i"] = ijk_positions[:, 0]
peak_data["j"] = ijk_positions[:, 1]
peak_data["k"] = ijk_positions[:, 2]
peak_data = peak_data[["i", "j", "k", "id"]]

nl.add_tuple_set(peak_data, name="peak_reported")
study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)

# %%
###############################################################################
# SQUALL controlled-English program
# ----------------------------------
# Four ``define`` sentences plus one ``obtain`` clause replace the entire
# ``with nl.scope as e:`` block and the separate ``nl.query()`` call.
#
# * ``such that ?s is a Selected_study``         — existential study filter
# * ``?t that is 'auditory'``                    — string equality on term
# * ``with probability … conditioned to``        — MARG conditional probability
# * ``(?i; ?j; ?k; _)``                          — ``_`` drops the study column
#   from the conditioned head (anonymous wildcard)
# * ``of the Activation_given_term (?i;?j;?k;?p)`` — explicit tuple label so
#   the aggregation functor receives all four columns (i, j, k, probability)
# * ``obtain every Activation_given_term_image (?x)`` — runs only this query
#
# Note: ``?tfidf`` binds the TF-IDF weight column but is absent from the rule
# head (projected away); SQUALL requires every argument in a tuple label to be
# a named variable — anonymous wildcards are not supported in that position.

squall_program = """
define as Activation every Peak_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
    such that ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?i; ?j; ?k; _)
    conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term (?i; ?j; ?k; ?p).

obtain every Activation_given_term_image (?x).
"""

result_set = nl.execute_squall_program(squall_program)

# %%
###############################################################################
# Solve and retrieve the probability map
# --------------------------------------
# The ``obtain`` clause executes only the image query; ``result_set`` is the
# single-column result directly from :func:`execute_squall_program`.

result_image = result_set.as_pandas_dataframe().iloc[0, 0].spatial_image()

# %%
###############################################################################
# Plot
# ----

img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
