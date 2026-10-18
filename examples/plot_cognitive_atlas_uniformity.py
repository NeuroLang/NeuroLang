# -*- coding: utf-8 -*-
r"""
Uniformity Map of a Cognitive Atlas Term, Linked to the Studies With SPECTER2.
==============================================================================

Neurosynth maps a *term* by the studies that mention it.  The Cognitive
Atlas defines its concepts with a text, and this example links that text to
the studies without any mention: the definition of the concept the term
names in the Cognitive Atlas ontology, loaded by the ``neurostore`` engine
(see :ref:`tutorial-cognitive-atlas`), is embedded with SPECTER2 as the
free-text query of ``query_similarity(query, study)``, which gives the cosine
similarity of the title and abstract of every study to it (see
:ref:`tutorial-text-embeddings`).  Every rule has the term as its first
argument, here ``'working memory'``::

    definition(t, d) :- cogat_term(c, t) & skos:definition(c, d).
    similarity(t, s, v) :- study(s) & definition(t, d) &
        (v == query_similarity(d, s)).

The 5% of the studies closest to the definition play the role of the studies
of the term in Neurosynth's *uniformity test*: the map is the probability
:math:`P(A_v \mid T)` that such a study activates the voxel :math:`v`,
divided by its average :math:`P(A_V \mid T)` over the brain, which is 1 when
the studies activate the brain uniformly (see
:ref:`tutorial-neurosynth-tests`).  The map is compared with the one of the
studies that mention the term.

The first run downloads the NeuroStore release, the Cognitive Atlas ontology
and SPECTER2, and embeds the abstracts of the studies; it needs
``pip install "neurolang[embeddings]"``.  Embedding the 36,000 abstracts takes
minutes on a GPU (see :ref:`tutorial-text-embeddings` to compute them ahead of
time, e.g. on a cluster).
"""

# %%
import warnings

warnings.filterwarnings("ignore")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import nibabel
import numpy as np
from neurolang.utils.engine_registry import build_engine

# A term of the Cognitive Atlas, in lower case as the terms of the studies.
TERM = "working memory"
# The studies closest to the definition are those of the top TOP_PERCENT %.
TOP_PERCENT = 5

# %%
# Build the NeuroStore engine
# ---------------------------
# Peaks are mapped to the 2 mm MNI grid.  The engine loads the Cognitive
# Atlas ontology and declares ``query_similarity`` in its YAML configuration;
# nothing is embedded until a query uses the function.

data_dir = Path.home() / "neurolang_data"

start = time.perf_counter()
nl = build_engine("neurostore", data_dir, resolution=2)
print(f"Engine built in {time.perf_counter() - start:.1f}s")

mni = nilearn.datasets.load_mni152_template(resolution=2)

# %%
# The term in the ontology
# ------------------------
# The ontology is queried as the facts it contains: the term is the label
# of a concept, in lower case.

definition = nl.execute_datalog_program(
    f"ans(d) :- cogat_term(c, '{TERM}') & skos:definition(c, d)."
)
print(definition.as_pandas_dataframe().iloc[0, 0])

# %%
# Studies close to the definition
# -------------------------------
# The similarity of every study to the definition, and the cutoff of the top
# ``TOP_PERCENT`` % of them, an aggregation of the similarities.  A study
# without abstract has a similarity of 0 and is left out.

nl.execute_datalog_program(
    f"""
    definition(t, d) :- cogat_term(c, t) & skos:definition(c, d).
    similarity(t, s, v) :- study(s) & definition(t, d) &
        (v == query_similarity(d, s)) & (v > 0).
    cutoff(t, percentile(v, {100 - TOP_PERCENT})) :- similarity(t, s, v).
    similar(t, s) :- similarity(t, s, v) & cutoff(t, c) & (v >= c).
    term_study(t, s) :- mentions(s, t).
    """
)
start = time.perf_counter()
similarity = nl.execute_datalog_program(
    f"ans(s, v, h) :- similarity('{TERM}', s, v) & study_title(s, h)."
).as_pandas_dataframe()
print(f"Similarity query: {time.perf_counter() - start:.1f}s")
similarity.columns = ["study", "similarity", "title"]
similarity = similarity.sort_values("similarity", ascending=False)

similar = set(
    nl.execute_datalog_program(f"ans(s) :- similar('{TERM}', s).")
    .as_pandas_dataframe()
    .iloc[:, 0]
)
mentioning = set(
    nl.execute_datalog_program(f"ans(s) :- term_study('{TERM}', s).")
    .as_pandas_dataframe()
    .iloc[:, 0]
)
similarity["mentions"] = similarity.study.isin(mentioning)
print(
    f"{len(similarity)} studies, similarity mean "
    f"{similarity.similarity.mean():.3f}, sd {similarity.similarity.std():.3f}"
)
print(
    f"{len(similar)} studies in the top {TOP_PERCENT}%, of which "
    f"{len(similar & mentioning)} mention '{TERM}' "
    f"({len(mentioning)} mention it in all)"
)
for _, row in similarity.head(10).iterrows():
    mark = "*" if row.mentions else " "
    print(f"{row.similarity:.3f} {mark} {row.title[:80]}")

# %%
# Uniformity ratios
# -----------------
# ``selected_study`` is a uniform choice over the studies and
# ``selected_voxel`` over the voxels of the brain, so that the probabilities
# below are proportions of studies and of voxels.  A study activates the
# voxels within 3 voxels (6 mm) of one of its peaks.  The uniformity ratio
# of a selection of studies is :math:`P(A_v \mid T) / P(A_V \mid T)`, and the
# rules apply to both selections, ``similar`` and ``term_study``.

nl.execute_datalog_program(
    """
    activation(i, j, k, s) :- brain_mask(i, j, k) &
        peak_reported(i1, j1, k1, s) &
        (EUCLIDEAN(i, j, k, i1, j1, k1) <= 3).
    n_voxels(count(i, j, k)) :- brain_mask(i, j, k).
    selected_voxel(i, j, k) ^ (1 / m) :- brain_mask(i, j, k) & n_voxels(m).
    """
)
for name in ("similar", "term_study"):
    nl.execute_datalog_program(
        f"""
        {name}_given(t, i, j, k, PROB) :- (activation(i, j, k, s) &
            selected_study(s)) // ({name}(t, s) & selected_study(s)).
        {name}_any(t, PROB) :- (activation(i, j, k, s) &
            selected_voxel(i, j, k) & selected_study(s)) //
            ({name}(t, s) & selected_study(s)).
        {name}_uniformity(t, i, j, k, u) :- {name}_given(t, i, j, k, p) &
            {name}_any(t, pbar) & (u == p / pbar).
        """
    )


def uniformity_image(name):
    """Solve the uniformity ratios of a selection and return them as an image."""
    start = time.perf_counter()
    res = nl.execute_datalog_program(
        f"ans(i, j, k, u) :- {name}_uniformity('{TERM}', i, j, k, u)."
    )
    print(f"{name}: {time.perf_counter() - start:.1f}s")
    ijku = res.as_pandas_dataframe().values
    ijk = ijku[:, :3].astype(int)
    data = np.zeros(mni.shape)
    data[tuple(ijk.T)] = ijku[:, 3]
    return nibabel.Nifti1Image(data, mni.affine)


images = {
    f"top {TOP_PERCENT}% studies by similarity to the definition of "
    f"'{TERM}'": uniformity_image("similar"),
    f"studies mentioning '{TERM}'": uniformity_image("term_study"),
}

# %%
# Agreement of both maps
# ----------------------
# Pearson correlation over the voxels where either map is positive, and the
# Dice overlap of the top 5% of those voxels.

first, second = (image.get_fdata() for image in images.values())
mask = (first > 0) | (second > 0)
a, b = first[mask], second[mask]
top_a, top_b = a > np.quantile(a, 0.95), b > np.quantile(b, 0.95)
dice = 2 * (top_a & top_b).sum() / (top_a.sum() + top_b.sum())
print(f"r = {np.corrcoef(a, b)[0, 1]:.2f}, Dice top 5% = {dice:.2f}")

# %%
# Plot
# ----
# The voxels above the 95th percentile of each map.

fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 3))
for ax, (title, image) in zip(axes, images.items()):
    data = image.get_fdata()
    nilearn.plotting.plot_glass_brain(
        image,
        threshold=np.quantile(data[data > 0], 0.95),
        axes=ax,
        display_mode="ortho",
    )
    ax.set_title(title, fontsize=11)
nilearn.plotting.show()
