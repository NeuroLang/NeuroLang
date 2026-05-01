# -*- coding: utf-8 -*-
r"""
CBMA Spatial Prior in SQUALL Controlled English
================================================

Reproduces the coordinate-based meta-analysis spatial decay prior — each
voxel's probability of being reported by a study is weighted by
``max(exp(−d/5))`` over nearby foci — using `SQUALL controlled natural
language <https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the
IR builder DSL.

All five logic rules, including the distance-weighted voxel proximity
aggregation, are expressed as plain English sentences passed to
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`.  Compare with
:ref:`sphx_glr_auto_examples_plot_cbma_spatial_prior.py` which writes the same
computation using ``with nl.environment as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Voxel_reported with a probability of
        the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
            per ?i1, ?j1, ?k1 and per ?s
            where (?i1; ?j1; ?k1) is a Voxel
            and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.

    define as Term_association every Term_in_study_tfidf (?s; ?t; _)
        such that ?s is a Selected_study.

    define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Probmap with probability every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (_; ?t) such that ?t is 'emotion'.

    define as Img every Agg_create_region_overlay of the Probmap (?i; ?j; ?k; ?p).
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
# Data preparation
# ----------------
# Set the data directory and load the MNI T1 atlas resampled to 2 mm voxels.

data_dir = Path.home() / "neurolang_data"

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

# %%
# Set up engine and register aggregation symbols
# -----------------------------------------------
# ``euclidean`` computes the Euclidean distance between two voxel coordinates.
# ``agg_max_proximity`` folds a collection of distances into
# ``max(exp(−d/5))``, the spatial-decay weight used in coordinate-based
# meta-analysis.  ``agg_create_region_overlay`` assembles the final
# probability map into a brain overlay image.

nl = NeurolangPDL()


def euclidean(
    i1: int, j1: int, k1: int, i2: int, j2: int, k2: int
) -> float:
    """Euclidean distance between two voxel coordinates (numpy-vectorised)."""
    return np.sqrt(
        (i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2
    )


nl.add_symbol(euclidean, name="EUCLIDEAN")


@nl.add_symbol
def agg_max_proximity(d_values: Iterable) -> float:
    """Aggregate: max exp(−d/5) over a collection of distances."""
    return float(np.max(np.exp(-np.asarray(d_values) / 5.0)))


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_2mm.affine, p, image_dim=mni_t1_2mm.shape
    )


# %%
# Load the NeuroSynth database
# ----------------------------
# Register peaks, study IDs, and term–study associations as extensional facts.
# Relation names are lowercase so they match SQUALL's case-folding convention.
# The full voxel grid is registered so the SQUALL rule can cross-join voxels
# with foci and filter by distance inline — no Python pre-computation needed.

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

nl.add_tuple_set(peak_data, name="focus_reported")

study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)

# Full voxel grid (vectorised, no loop)
shape = mni_t1_2mm.get_fdata().shape
voxel_df = pd.DataFrame(
    np.array(list(np.ndindex(*shape)), dtype=np.int32),
    columns=["i", "j", "k"],
)
nl.add_tuple_set(voxel_df, name="voxel")

# %%
# SQUALL controlled-English program
# ----------------------------------
# Five sentences replace the ``with nl.environment as e:`` block.
#
# Sentence 1  Probabilistic aggregation: ``per ?i1, ?j1, ?k1 and per ?s``
#             groups by the four head variables; ``where (?i1; ?j1; ?k1) is
#             a Voxel`` constrains the focus coordinates to voxel grid
#             membership; ``EUCLIDEAN(…) is lower than 5`` filters by
#             proximity inline — no relay variable needed.
# Sentences 2-3  ``such that ?s is a Selected_study`` — existential study
#             filter; ``_`` (anonymous wildcard) drops the tfidf column from
#             Term_association's head so only (study, term) are exposed.
# Sentence 4  ``conditioned to every Term_association (_; ?t) such that ?t
#             is 'emotion'`` — MARG conditional probability query; ``_``
#             hides the study column in the conditioning noun phrase.
# Sentence 5  ``every Agg_create_region_overlay of the Probmap`` — brain image
#             aggregation.

squall_program = """
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and per ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k; _)
    conditioned to every Term_association (_; ?t) such that ?t is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap (?i; ?j; ?k; ?p).
"""

nl.execute_squall_program(squall_program)
# %%
# Solve and retrieve the probability map
# --------------------------------------

solution = nl.solve_all()
print(solution.keys())
result_image = (
    solution["img"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)

# %%
# Plot
# ----

img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
