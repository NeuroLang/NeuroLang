# -*- coding: utf-8 -*-
r"""
Forward Inference From a Free-Text Query.
=========================================

NeuroSynth associates studies with the terms of a fixed vocabulary through
TF-IDF counts.  The ``neurosynth`` engine also embeds the title and abstract
of every study with SPECTER2 and exposes the cosine similarity to any
free-text query as a function::

    similarity(s, sim) :- study(s) & (
        sim == query_similarity("visual word form reading", s)
    ).

``query_similarity(query, s)`` returns the similarity of study ``s`` to the
query.  The query text is embedded once and the similarities of all the
studies are cached, so each call is a lookup.  The query can also be a
variable of a rule, bound by the rule that calls it::

    close(q, s) :- study(s) & (sim == query_similarity(q, s)) & (sim > .75).
    ans(s) :- close("visual word form reading", s).

This example lists the studies closest to "visual word form reading",
selects as many of them as there are studies tagged with the term
'reading', and compares the forward-inference maps
:math:`P(\text{activation} \mid \text{study selected})` of both selections.

The first run downloads the titles and abstracts from PubMed and embeds
them (see :ref:`tutorial-text-embeddings`); it needs
``pip install "neurolang[embeddings]"``.
"""

# %%
import warnings

warnings.filterwarnings("ignore")

import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
from neurolang.frontend.neurosynth_utils import StudyID, fetch_study_metadata
from neurolang.utils.engine_registry import build_engine

QUERY = "visual word form reading"
TERM = "reading"

# %%
# Build the NeuroSynth engine
# ---------------------------
# Peaks are mapped to 4 mm voxels.  The engine declares the
# ``query_similarity(query, study)`` function in its YAML configuration;
# nothing is embedded until a query uses it.

data_dir = Path.home() / "neurolang_data"

start = time.perf_counter()
nl = build_engine("neurosynth", data_dir, resolution=4)
print(f"Engine built in {time.perf_counter() - start:.1f}s")

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_4mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 4)

# %%
# Study titles come from the NeuroSynth metadata.

metadata = fetch_study_metadata(data_dir, verbose=0)
titles = metadata[["id", "title"]].drop_duplicates("id")
titles = titles.astype({"id": StudyID})
nl.add_tuple_set(titles, name="study_title")

# %%
# Studies closest to the query
# ----------------------------
# The first rule gives the similarity of every study to the query; the
# answer joins it with the titles.

start = time.perf_counter()
top = nl.execute_datalog_program(
    f"""
    similarity(s, v) :- study(s) & (v == query_similarity('{QUERY}', s)).
    reading_study(s) :- term_in_study_tfidf(s, '{TERM}', w).
    ans(s, v, t) :- similarity(s, v) & study_title(s, t).
    """
)
print(f"Similarity query: {time.perf_counter() - start:.1f}s")
top = top.as_pandas_dataframe()
top.columns = ["study", "similarity", "title"]
top = top.sort_values("similarity", ascending=False)

reading = nl.execute_datalog_program("ans(s) :- reading_study(s).")
reading = set(reading.as_pandas_dataframe().iloc[:, 0])
top["tagged"] = top.study.isin(reading)

print(
    f"{len(top)} studies, similarity mean {top.similarity.mean():.3f}, "
    f"sd {top.similarity.std():.3f}; {len(reading)} tagged '{TERM}'"
)
for n in (20, 100, len(reading)):
    print(f"top {n}: {top.tagged[:n].sum()} tagged '{TERM}'")
for _, row in top.head(10).iterrows():
    mark = "*" if row.tagged else " "
    print(f"{row.similarity:.3f} {mark} {row.study} {row.title[:70]}")

# %%
# Selecting studies in Datalog
# ----------------------------
# The text selection keeps as many studies as there are studies tagged
# with the term: the cutoff is the ``n``-th largest similarity, computed by
# an aggregation over the ``similarity`` rows.


@nl.add_symbol
def nth_largest(values: Iterable, n: Iterable) -> float:
    """Return the ``n``-th largest of ``values``."""
    return float(np.sort(np.asarray(values))[-int(np.asarray(n)[0])])


nl.execute_datalog_program(
    """
    n_reading(count(s)) :- reading_study(s).
    cutoff(nth_largest(v, n)) :- similarity(s, v) & n_reading(n).
    text_study(s) :- similarity(s, v) & cutoff(c) & (v >= c).
    """
)

# %%
# Forward-inference maps
# ----------------------
# ``selected_study`` is a uniform choice over the studies.  Each map is the
# probability that the selected study reports a peak in a voxel, given
# that it belongs to a selection.

nl.execute_datalog_program(
    """
    activation(i, j, k) :- selected_study(s) & peak_reported(i, j, k, s).
    in_text(s) :- selected_study(s) & text_study(s).
    in_term(s) :- selected_study(s) & reading_study(s).
    text_map(i, j, k, PROB) :- activation(i, j, k) // in_text(s).
    term_map(i, j, k, PROB) :- activation(i, j, k) // in_term(s).
    """
)


# Each study contributes only the voxels of its peaks, so the maps are
# smoothed with an 8 mm Gaussian kernel, as for coordinate-based
# meta-analysis maps, before they are compared and plotted.

SMOOTHING_FWHM = 8


def solve_map(name):
    """Solve a MARG query and return its smoothed probability image."""
    start = time.perf_counter()
    res = nl.execute_datalog_program(f"ans(i, j, k, p) :- {name}(i, j, k, p).")
    print(f"{name}: {time.perf_counter() - start:.1f}s")
    ijkp = res.as_pandas_dataframe().values
    ijk = ijkp[:, :3].astype(int)
    # A few peaks fall outside the MNI grid.
    inside = np.all((ijk >= 0) & (ijk < mni_t1_4mm.shape), axis=1)
    data = np.zeros(mni_t1_4mm.shape)
    data[tuple(ijk[inside].T)] = ijkp[inside, 3]
    image = nibabel.Nifti1Image(data, mni_t1_4mm.affine)
    return nilearn.image.smooth_img(image, SMOOTHING_FWHM)


maps = {
    f"top {len(reading)} studies by similarity": solve_map("text_map"),
    f"studies tagged '{TERM}'": solve_map("term_map"),
}

# %%
# Agreement with the term map
# ---------------------------
# Pearson correlation over the voxels where either map is positive, and
# Dice overlap of the top 5% of these voxels.

term = maps[f"studies tagged '{TERM}'"].get_fdata()


def agreement(image):
    """Return the correlation and top-5% Dice with the term map."""
    data = image.get_fdata()
    mask = (data > 0) | (term > 0)
    a, b = data[mask], term[mask]
    top_a = a > np.quantile(a, 0.95)
    top_b = b > np.quantile(b, 0.95)
    dice = 2 * (top_a & top_b).sum() / (top_a.sum() + top_b.sum())
    return np.corrcoef(a, b)[0, 1], dice


labels = {title: title for title in maps}
text_title = f"top {len(reading)} studies by similarity"
r, dice = agreement(maps[text_title])
print(f"{text_title}: r = {r:.2f}, Dice top 5% = {dice:.2f}")
labels[text_title] += f"\nr = {r:.2f}, Dice top 5% = {dice:.2f}"

# %%
# Plot
# ----
# Voxels above the 95th percentile of each map, interpolated to 2 mm for
# display.

fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 3))
for ax, (title, image) in zip(axes, maps.items()):
    data = image.get_fdata()
    nilearn.plotting.plot_glass_brain(
        nilearn.image.resample_img(
            image, np.eye(3) * 2, interpolation="continuous"
        ),
        threshold=np.quantile(data[data > 0], 0.95),
        axes=ax,
        display_mode="ortho",
    )
    ax.set_title(labels[title], fontsize=11)
nilearn.plotting.show()
