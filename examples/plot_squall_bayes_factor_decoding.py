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
Bayes Factor Decoding of the Right Fusiform Gyrus in SQUALL Controlled English.
===============================================================================

Performs reverse-inference decoding of the right fusiform gyrus from the
`Julich-Brain v2.9 <https://doi.org/10.1126/science.abb4588>`_ atlas using
Bayes Factors, expressed entirely in
`SQUALL controlled natural language <https://doi.org/10.18653/v1/2020.acl-main.235>`_.

For each cognitive term in the Neurosynth database the Bayes Factor quantifies
the evidence that the right fusiform gyrus is specifically associated with that
term:

.. math::

   \mathrm{BF}(r, t) =
   \frac{P(T{=}t \mid R{=}r)}{P(T{=}t \mid R{\neq}r)}
   =
   \frac{P(R,T)/P(R)}{(P(T) - P(R,T))/(1 - P(R))}

Following Jeffreys (1961) a threshold :math:`\mathrm{BF} > \sqrt{10} \approx 3.16`
indicates "substantial" evidence of association.

The three probability distributions — joint :math:`P(R,T)`, marginal
:math:`P(R)`, and marginal :math:`P(T)` — are each expressed as a single SQUALL
sentence.  Region selection is placed in the ``obtain`` clause so that
NeuroLang's magic-sets optimisation pushes the filter backwards through all three
rules, limiting computation to the target region.
Compare with :ref:`sphx_glr_auto_examples_plot_squall_cbma_spatial_prior.py`
which demonstrates the same SQUALL machinery for a spatial-prior computation.

The Julich-Brain atlas represents the fusiform gyrus as four cytoarchitectonic
areas: ``Area FG1 (FusG) right``, ``Area FG2 (FusG) right``,
``Area FG3 (FusG) right``, and ``Area FG4 (FusG) right``.  These are unioned
into a single ``'right fusiform gyrus'`` label for the decoding analysis.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Active_region every Region that a Selected_study activates.

    define as Region_probability with inferred probability
        every Active_region.

    define as Mentioned_term every Term that a Selected_study mentions.

    define as Term_probability with inferred probability
        every Mentioned_term.

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

    define as Joint_probability with inferred probability
        every Cooccurrence (?r; ?t).

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

The first two pairs of ``define`` sentences use natural-language
relative clauses (``every Region that a Selected_study activates``) to
build binary intermediate rules that inject the probabilistic
``Selected_study`` choice into the body with **zero explicit variables**.
The ternary cooccurrence rule is now expressed entirely in natural
English using the new compound-quantifier and anaphora support —
``for every Region and for every Term where a Selected_study activates
the Region and mentions the Term`` — so the join is done by the NeuroLang
solver with no ``~`` inverse-verb prefix and no explicit labels.
The three ``with inferred probability`` rules then ask the solver for
the marginalised probabilities — study is quantified away so only region
and/or term remain in the rule heads.
A new ``define as Bayes_factor`` rule computes the Bayes Factor formula
directly in SQUALL using **arithmetic expressions** (``+``, ``-``, ``*``,
``/``) and **bare predicate calls** (``Joint_probability (?r, ?t, ?p_rt)``)
in the rule body — the formula :math:`\mathrm{BF}= \frac{P(R,T)/P(R)}
{(P(T)-P(R,T))/(1-P(R))}` is expressed inline with ``?bf is``.
The ``obtain … as`` clause returns the ranked results directly.
"""

# %%
import warnings
from pathlib import Path

import nibabel
import nibabel.affines
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd
import siibra
from nilearn.surface import vol_to_surf

from neurolang.frontend import NeurolangPDL
from neurolang.frontend.neurosynth_utils import (
    get_ns_mni_peaks_reported,
    get_ns_term_study_associations,
)

warnings.filterwarnings("ignore")

# %%
# Constants
# ---------

data_dir = Path.home() / "neurolang_data"
JULICH_VERSION = "2.9"

# The Julich-Brain atlas represents the fusiform gyrus as four cytoarchitectonic
# areas; we union them under a single label for the analysis.
FUSIFORM_AREAS = [
    "Area FG1 (FusG) right",
    "Area FG2 (FusG) right",
    "Area FG3 (FusG) right",
    "Area FG4 (FusG) right",
]
TARGET_LABEL = "right fusiform gyrus"

BF_THRESHOLD = np.sqrt(10)   # Jeffreys "substantial" evidence ≈ 3.16
TOP_N = 20                   # terms to display in bar chart

# %%
# Data preparation — MNI atlas
# ----------------------------
# Load the ICBM152 T1 template and downsample to 2 mm isotropic voxels.

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

# %%
# Data preparation — right fusiform gyrus mask (siibra / Julich-Brain)
# --------------------------------------------------------------------
# Fetch the Julich-Brain v2.9 labelled map and build a binary union mask for
# the right fusiform gyrus (Areas FG1–FG4).  Each area is fetched individually
# with ``julich_map.fetch(region=name)`` (returns a 0/1 NIfTI), then the four
# masks are OR-combined and resampled to the 2 mm MNI grid.
# The mask image is used only for the region anatomy plot (section 6a).

julich_map = siibra.get_map(
    parcellation=f"julich {JULICH_VERSION}",
    space="mni152",
    maptype=siibra.MapType.LABELLED,
)

# Fetch and union the four FG areas into one binary mask.
area_masks = [julich_map.fetch(region=area) for area in FUSIFORM_AREAS]
combined_arr = np.zeros(area_masks[0].shape, dtype=np.uint8)
for m in area_masks:
    combined_arr |= (m.get_fdata() > 0).astype(np.uint8)

region_mask_img = nibabel.Nifti1Image(combined_arr, area_masks[0].affine)
region_mask_2mm = nilearn.image.resample_to_img(
    region_mask_img, mni_t1_2mm, interpolation="nearest"
)
print(
    f"Right fusiform gyrus mask: "
    f"{int(region_mask_2mm.get_fdata().sum())} voxels at 2 mm"
)

# %%
# Data preparation — Neurosynth peaks → activates
# -------------------------------------------------
# Load reported activation foci from Neurosynth, convert MNI (x,y,z) to
# voxel indices in the 2 mm MNI grid, then use the Julich-Brain labelled map
# to assign each peak to an anatomical region.
#
# The four fusiform gyrus areas (FG1–FG4 right) are unified under
# ``TARGET_LABEL`` so the SQUALL ``obtain`` filter ``where ?r is 'right
# fusiform gyrus'`` matches them.
#
# ``activates(study_id, region)`` has study_id first so the SQUALL
# ``Activates (?s; ?r)`` clauses join on column 0.

peak_data = get_ns_mni_peaks_reported(data_dir)

ijk = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_2mm.affine),
        peak_data[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
peak_data = peak_data.copy()
peak_data["i"] = ijk[:, 0]
peak_data["j"] = ijk[:, 1]
peak_data["k"] = ijk[:, 2]

# Build label → region-name lookup from the Julich-Brain map.
# get_index(region_name) returns a MapIndex with .label (int) attribute.
label_to_name = {
    julich_map.get_index(r).label: r
    for r in julich_map.regions
}

# Resample the full labelled volume to the 2 mm MNI grid.
label_vol_2mm = nilearn.image.resample_to_img(
    julich_map.fetch(),
    mni_t1_2mm,
    interpolation="nearest",
)
label_arr = label_vol_2mm.get_fdata().astype(int)

# Keep only peaks inside the image bounds.
shape = label_arr.shape
in_bounds = (
    (peak_data["i"] >= 0) & (peak_data["i"] < shape[0]) &
    (peak_data["j"] >= 0) & (peak_data["j"] < shape[1]) &
    (peak_data["k"] >= 0) & (peak_data["k"] < shape[2])
)
peak_data = peak_data[in_bounds].copy()

# Look up the region name for each peak voxel.
peak_labels = label_arr[
    peak_data["i"].values,
    peak_data["j"].values,
    peak_data["k"].values,
]
peak_data["region"] = [label_to_name.get(int(lbl), None) for lbl in peak_labels]
peak_data = peak_data.dropna(subset=["region"])

# Unify the four fusiform areas under TARGET_LABEL.
peak_data["region"] = peak_data["region"].apply(
    lambda r: TARGET_LABEL if r in FUSIFORM_AREAS else r
)

study_activates_df = peak_data[["id", "region"]].drop_duplicates()
print(f"activates: {len(study_activates_df)} (study, region) pairs")
fusiform_studies = study_activates_df[
    study_activates_df["region"] == TARGET_LABEL
]
print(f"  → {len(fusiform_studies)} studies activate the right fusiform gyrus")

# %%
# Set up NeuroLang engine
# -----------------------

nl = NeurolangPDL()

# %%
# Register extensional relations
# ------------------------------
# ``activates(study_id, region)``  — from peak-to-region assignment above.
# ``mentions(study_id, term)``     — from Neurosynth TF-IDF associations.
# All three have ``study_id`` first so the SQUALL natural-language clauses
# join naturally on column 0.

term_data = get_ns_term_study_associations(data_dir, tfidf_threshold=1e-3)
study_mentions_df = term_data[["id", "term"]].drop_duplicates()

# Register relations with the names SQUALL expects (lowercase for case-folding).
nl.add_tuple_set(study_activates_df, name="activates")
nl.add_tuple_set(study_mentions_df, name="mentions")

# Register unary type predicates so SQUALL natural-language nouns resolve.
study_ids = sorted(
    set(study_activates_df["id"]) & set(study_mentions_df["id"])
)
nl.add_tuple_set(
    study_activates_df[["region"]].drop_duplicates().rename(
        columns={"region": "region"}
    ),
    name="region",
)
nl.add_tuple_set(
    study_mentions_df[["term"]].drop_duplicates().rename(
        columns={"term": "term"}
    ),
    name="term",
)

# Uniform probabilistic choice over all studies that appear in both relations.
study_ids_df = pd.DataFrame({"id": study_ids})
nl.add_uniform_probabilistic_choice_over_set(
    study_ids_df, name="selected_study"
)
print(
    f"Studies: {len(study_ids_df)}, "
    f"term-study pairs: {len(study_mentions_df)}"
)

# %%
# SQUALL controlled-English program
# ----------------------------------
# Natural-language relative clauses (``every Region that a Selected_study
# activates``) build binary intermediate rules that join the deterministic
# Neurosynth data with the probabilistic ``Selected_study`` choice without
# any explicit variables.  The ternary cooccurrence rule uses the new
# compound-quantifier and anaphora support — ``for every Region and for
# every Term where a Selected_study activates the Region and mentions the
# Term`` — so the join is done by the NeuroLang solver with no ``~`` prefix
# and no explicit labels.  The three ``with inferred probability`` rules ask
# the solver for the marginalised probabilities — study is quantified away
# so only region and/or term remain in the rule heads.
#
# The Bayes Factor rule uses the new **arithmetic expressions** and **bare
# predicate calls** to compute :math:`\mathrm{BF}= \frac{P(R,T)/P(R)}
# {(P(T)-P(R,T))/(1-P(R))}` directly in SQUALL, avoiding post-hoc pandas
# computation.  The ``obtain … as`` clause returns the ranked results.

squall_program = """
define as Active_region every Region that a Selected_study activates.

define as Region_probability with inferred probability
    every Active_region.

define as Mentioned_term every Term that a Selected_study mentions.

define as Term_probability with inferred probability
    every Mentioned_term.

define as Cooccurrence
    for every Region and for every Term
    where a Selected_study activates the Region and mentions the Term.

define as Joint_probability with inferred probability
    every Cooccurrence (?r; ?t).

define as Bayes_factor (?r; ?t; ?bf)
    where Joint_probability (?r, ?t, ?p_rt)
    and Region_probability (?r, ?p_r)
    and Term_probability (?t, ?p_t)
    and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

obtain every Bayes_factor (?r; ?t; ?bf)
    where ?r is 'right fusiform gyrus' as BF.
"""

# %%
# Execute SQUALL program
# ----------------------
# ``execute_squall_program`` with named ``obtain … as`` clauses returns a
# dict of results directly — no ``solve_all()`` needed.

result = nl.execute_squall_program(squall_program)

# %%
# Bayes Factor results
# --------------------
# The Bayes Factor formula :math:`\mathrm{BF}= \frac{P(R,T)/P(R)}
# {(P(T)-P(R,T))/(1-P(R))}` was computed directly inside the SQUALL
# program via arithmetic expressions and bare predicate calls in the
# ``Bayes_factor`` rule — the ``obtain … as BF`` clause returns the
# ranked per-term BF values without any post-hoc pandas computation.

bf_df = result.as_pandas_dataframe()
bf_df.columns = ["region", "term", "bf"]

top_terms = (
    bf_df[bf_df["bf"] > BF_THRESHOLD]
    .sort_values("bf", ascending=False)
    .head(TOP_N)
)
print(top_terms[["term", "bf"]].to_string(index=False))

# %%
# Plot — right fusiform gyrus anatomy (ventral view)
# ---------------------------------------------------
# Show the right fusiform gyrus ROI on the fsaverage5 inflated surface,
# right hemisphere, ventral viewpoint with the anterior pole at the top.
# Sulcal depth map provides shading to aid anatomical orientation.

fsaverage = nilearn.datasets.fetch_surf_fsaverage("fsaverage5")

try:
    # vol_to_surf requires an integer-typed NIfTI; cast before projecting.
    region_mask_int = nibabel.Nifti1Image(
        region_mask_2mm.get_fdata().astype(np.int16),
        region_mask_2mm.affine,
    )
    roi_texture = vol_to_surf(region_mask_int, fsaverage["pial_right"])
    # vol_to_surf can return interpolated float values; binarise before plotting.
    roi_texture = (roi_texture > 0.5).astype(int)
    nilearn.plotting.plot_surf_roi(
        surf_mesh=fsaverage["infl_right"],
        roi_map=roi_texture,
        hemi="right",
        view="ventral",
        bg_map=fsaverage["sulc_right"],
        bg_on_data=True,
        title=f"Right fusiform gyrus (Julich-Brain v{JULICH_VERSION})",
        colorbar=False,
    )
except Exception as e:
    print(f"Surface plot skipped: {e}")
    nilearn.plotting.plot_roi(
        region_mask_2mm,
        bg_img=mni_t1_2mm,
        display_mode="z",
        cut_coords=5,
        title=f"Right fusiform gyrus (Julich-Brain v{JULICH_VERSION})",
    )
nilearn.plotting.show()

# %%
# Plot — top terms by Bayes Factor
# --------------------------------
# Horizontal bar chart of the top-N cognitive terms ranked by BF for the
# right fusiform gyrus.  A vertical dashed line marks the Jeffreys
# ``BF > sqrt(10)`` threshold for "substantial" evidence.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 0.4 * len(top_terms) + 1))

ax.barh(
    top_terms["term"],
    top_terms["bf"],
    color="steelblue",
    edgecolor="white",
)
ax.axvline(
    BF_THRESHOLD,
    color="tomato",
    linestyle="--",
    linewidth=1.2,
    label=r"BF = $\sqrt{10}$ ≈ 3.16",
)
ax.set_xlabel("Bayes Factor")
ax.set_title(
    f"Top-{TOP_N} cognitive terms for the right fusiform gyrus\n"
    f"(Julich-Brain v{JULICH_VERSION}, Neurosynth)"
)
ax.legend(frameon=False)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print("Done")
