# -*- coding: utf-8 -*-
r"""
NeuroSynth Query in SQUALL Controlled English
=============================================

Reproduces the NeuroSynth forward model — P(Activation | Term = 'auditory') —
using `SQUALL controlled natural language
<https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the IR builder
DSL.

The four logic rules are expressed as plain English sentences and executed via
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`. Compare with
:ref:`sphx_glr_auto_examples_plot_neurosynth_implementation.py` which writes
identical rules using ``with nl.scope as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Activation every Peak_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
        such that ?s is a Selected_study.

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k)
        conditioned to every Term_association ?t that is 'auditory'.

    define as Activation_given_term_image
        every Agg_create_region_overlay of the Activation_given_term.
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
# Four sentences replace the entire ``with nl.scope as e:`` block.
#
# * ``such that ?s is a Selected_study``  — existential filter: study is selected
# * ``?t that is 'auditory'``             — string equality filter on the term
# * ``with probability … conditioned to`` — MARG query (conditional probability)
# * ``every Agg_create_region_overlay of the …`` — aggregation into brain image
#
# Note: SQUALL's tuple-label syntax ``(?s; ?t; ?tfidf)`` requires named
# variables — the grammar does not allow the ``_`` anonymous placeholder
# inside ``(; ;)`` tuples.  ``?tfidf`` binds the TF-IDF weight column
# but is intentionally absent from the rule head, so it is projected away.

squall_program = """
define as Activation every Peak_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
    such that ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'auditory'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term.
"""

nl.execute_squall_program(squall_program)

# %%
###############################################################################
# Solve and retrieve the probability map
# --------------------------------------

solution = nl.solve_all()
result_image = (
    solution["activation_given_term_image"]
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
