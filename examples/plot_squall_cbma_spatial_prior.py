# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
r"""
CBMA Spatial Prior in SQUALL Controlled English.
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

    define as Reported_voxel with a probability of
        the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
            for each ?i1, ?j1, ?k1 and for each ?s
            where (?i1; ?j1; ?k1) is a Voxel
            and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 2.

    define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
        where ?s is a Selected_study.

    define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
        where ?s is a Selected_study.

    define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
        given every Study_term (_; ?t) where ?t is 'emotion'.

    obtain every Activation_map (?i; ?j; ?k; ?p) as Image.

The result from ``obtain … as`` is a
:class:`~neurolang.utils.relational_algebra_set.NamedRelationalAlgebraFrozenSet`.
We convert to a pandas DataFrame and assemble the brain overlay using the
:func:`~neurolang.frontend.ExplicitVBROverlay` helper — see ``brain_image``
below.
"""

# %%
import warnings

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

warnings.filterwarnings("ignore")

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
# ``kernelized_max_proximity`` folds a collection of distances into
# ``max(exp(−d/5))``, the spatial-decay weight used in coordinate-based
# meta-analysis.  ``brain_image`` assembles the final
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
def kernelized_max_proximity(d_values: Iterable) -> float:
    """Aggregate: max exp(−d/5) over a collection of distances."""
    return float(np.max(np.exp(-np.asarray(d_values) / 5.0)))


@nl.add_symbol
def brain_image(
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

nl.add_tuple_set(peak_data, name="reported_focus")

study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_with_tfidf", tfidf_threshold=1e-3
)


# Full voxel grid (vectorised, no loop)
shape = mni_t1_2mm.get_fdata().shape
voxel_df = pd.DataFrame(
    np.array(list(np.ndindex(*shape)), dtype=np.int32),
    columns=["i", "j", "k"],
)
nl.add_tuple_set(voxel_df.drop_duplicates(), name="voxel")

# %%
from neurolang import config as nconfig
nconfig.disable_expression_type_printing()

# %%
# SQUALL controlled-English program
# ----------------------------------
# Five sentences replace the ``with nl.environment as e:`` block.
#
# Sentence 1  Probabilistic aggregation: ``for each ?i1, ?j1, ?k1 and for each ?s``
#             groups by the four head variables; ``where (?i1; ?j1; ?k1) is
#             a Voxel`` constrains the focus coordinates to voxel grid
#             membership; ``where EUCLIDEAN(…) is lower than 2`` filters by
#             proximity inline — no relay variable needed.
# Sentences 2-3  ``where ?s is a Selected_study`` — existential study
#             filter; ``_`` (anonymous wildcard) drops the tfidf column from
#             Study_term's head so only (study, term) are exposed.
# Sentence 4  ``given every Study_term (_; ?t) where ?t is 'emotion'`` —
#             conditional probability query with ``inferred probability``; ``_``
#             hides the study column in the conditioning noun phrase.
# Sentence 5  ``obtain the Brain_image of the Activation_map … as Image`` —
#             brain image aggregation using the ``obtain … as`` form.

squall_program = """
define as Reported_voxel with a probability of
    the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
        for each ?i1, ?j1, ?k1 and for each ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 2.

define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
    where ?s is a Selected_study.

define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
    where ?s is a Selected_study.

define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
    given every Study_term (_; ?t) where ?t is 'emotion'.

obtain every Activation_map (?i; ?j; ?k; ?p) as Image.
"""

with nl.scope:
    result = nl.execute_squall_program(squall_program)

    # The obtain … as query returns a NamedRelationalAlgebraFrozenSet.
    # Convert to a pandas DataFrame for column access, then assemble the
    # brain overlay.
    df = result.as_pandas_dataframe()
    image = brain_image(df["i"], df["j"], df["k"], df["p"])
    spatial_img = image.spatial_image()
    print("Brain image assembled — shape:", spatial_img.shape)
    print(f"Non-zero voxels: {np.count_nonzero(spatial_img.get_fdata())}")
    print(f"Rules defined: {len(nl.current_program)}")
    for r in nl.current_program:
        print(f"  - {r.expression.consequent.functor}")

    # Plot the result
    nilearn.plotting.plot_stat_map(
        image, bg_img=mni_t1_2mm, title="SQUALL CBMA Spatial Prior",
    )

print("Done")
