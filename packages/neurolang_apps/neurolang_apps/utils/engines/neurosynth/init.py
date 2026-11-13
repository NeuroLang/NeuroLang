"""
NeuroSynth engine initialization.

Registers the following predicates:

* ``peak_reported(i, j, k, study_id)`` — reported activation peaks
  converted to voxel coordinates, the Talairach ones converted to MNI
  (see :func:`process_peaks` for those of ``UNKNOWN`` space).
* ``peak_mni(x, y, z, study_id)`` — the same peaks in MNI coordinates, in
  millimetres.
* ``study(study_id)`` — all study identifiers.
* ``term_in_study_tfidf(study_id, term, tfidf)`` — term frequency–inverse
  document frequency values.
* ``term(term)`` — every term of the TF-IDF vocabulary.
* ``selected_study(study_id)`` — uniform probabilistic choice over
  studies.
* ``voxel(i, j, k)`` — every voxel index of the MNI template grid.
* ``brain_mask(i, j, k)`` — the voxels of the MNI template grid inside the
  MNI152 brain mask (whole brain, subcortical structures included).
* ``selected_voxel(i, j, k)`` — uniform probabilistic choice over the voxels
  of the brain mask.
* ``cortical_mask(i, j, k)`` — the voxels of the MNI template grid in the
  cortex, labelled by the Harvard–Oxford cortical atlas (maximum
  probability, 25% threshold, 2 mm).
* ``cognitive_task(task_id, name, description)`` — the tasks of the
  Cognitive Atlas, with their descriptions.
* ``study_title(study_id, title)``, ``study_year(study_id, year)`` and
  ``study_journal(study_id, journal)`` — the titles, years and journals of
  the studies, from the NeuroSynth metadata.
* ``study_abstract(study_id, abstract)`` — their abstracts, from PubMed
  (see :func:`study_abstracts`).

The ``datalog_init`` of the engine (``engines.yaml``) also derives
``mentions(study_id, term)`` and ``reports(study_id, i, j, k)``.

Predicate names are lowercase to match SQUALL's case-folding convention
(see ``intransitive`` in ``squall_syntax_lark.py``).
"""

import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.frontend import NeurolangPDL
from neurolang.frontend.neurosynth_utils import (
    StudyID,
    fetch_neurosynth_peak_data,
    fetch_study_metadata,
    get_ns_term_study_associations,
)
from neurolang.utils.nilearn_compat import (
    fetch_files,
    nilearn_datasets,
    nilearn_image,
)
from neurolang_apps.utils import engine_cache
from neurolang_apps.utils.text_embeddings import fetch_pubmed_abstracts

COGNITIVE_ATLAS_TASKS_URL = "https://www.cognitiveatlas.org/api/v-alpha/task"
ABSTRACTS_FILE = "pubmed_abstracts.json"


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path,
    unknown_space: str = "mni",
) -> None:
    """Populate a fresh engine with NeuroSynth data.

    Parameters
    ----------
    nl :
        Fresh :class:`~neurolang.frontend.NeurolangPDL` instance.
    mask :
        MNI template image for voxel-space coordinate mapping.
    data_dir :
        Directory under which downloaded data is cached.
    unknown_space :
        How the peaks of the studies of ``UNKNOWN`` space are taken (see
        :func:`process_peaks`); the engine's ``python_init_options``.

    """
    activations = fetch_neurosynth_peak_data(data_dir, verbose=0)
    peak_data = process_peaks(activations, mask, unknown_space=unknown_space)

    term_data = get_ns_term_study_associations(
        data_dir, verbose=0, convert_study_ids=True, tfidf_threshold=1e-3
    )
    term_data = term_data.rename(columns={"id": "study_id"})
    term_data["study_id"] = term_data["study_id"].astype(StudyID)
    study_ids = register_tables(nl, mask, data_dir, peak_data, term_data)
    metadata = fetch_study_metadata(data_dir, verbose=0)
    nl.add_tuple_set(study_titles(metadata, study_ids), name="study_title")
    metadata = metadata.rename(columns={"id": "study_id"}).astype(
        {"study_id": StudyID}
    )
    register_publications(
        nl, metadata[metadata["study_id"].isin(study_ids["study_id"])]
    )
    nl.add_tuple_set(
        study_abstracts(study_ids["study_id"], data_dir),
        name="study_abstract",
    )


def study_titles(
    metadata: pd.DataFrame, study_ids: pd.DataFrame
) -> pd.DataFrame:
    """Return the ``study_id, title`` rows of the studies with a title."""
    titles = metadata[["id", "title"]].rename(columns={"id": "study_id"})
    titles = titles.astype({"study_id": StudyID}).dropna()
    titles = titles[titles["title"].str.strip() != ""]
    return titles[titles["study_id"].isin(study_ids["study_id"])]


def register_publications(nl: NeurolangPDL, studies: pd.DataFrame) -> None:
    """
    Add to *nl* ``study_year(study_id, year)`` and ``study_journal(study_id,
    journal)``, the rows of *studies* (``study_id``, ``year``, ``journal``)
    where they are known.
    """
    for column, kind in (("year", int), ("journal", str)):
        table = studies.reindex(columns=["study_id", column]).dropna()
        if kind is str:
            table = table[table[column].astype(str).str.strip() != ""]
        table = table.astype({column: kind}).drop_duplicates()
        nl.add_tuple_set(table, name=f"study_{column}")


def study_abstracts(study_ids: pd.Series, data_dir: Path) -> pd.DataFrame:
    """
    Return the ``study_id, abstract`` rows of the studies with an abstract
    in PubMed: a study's id is its PubMed id.

    The abstracts are asked for to PubMed at the first build, by batches
    of two hundred (see
    :func:`~neurolang_apps.utils.text_embeddings.fetch_pubmed_abstracts`),
    and kept in ``DATA_DIR/neurosynth/pubmed_abstracts.json``: the next
    builds ask for none. Without the network the rows are those kept, with
    a warning; in a browser (Pyodide) those that came with the page.
    """
    path = Path(data_dir) / "neurosynth" / ABSTRACTS_FILE
    ids = [str(study) for study in study_ids]
    try:
        if sys.platform == "emscripten":
            raise OSError("a browser does not ask PubMed")
        records = fetch_pubmed_abstracts(ids, cache_path=path)
    except OSError as error:
        if sys.platform != "emscripten":
            warnings.warn(
                f"PubMed could not be asked for the abstracts ({error}): "
                "the studies without one kept have none, until the next "
                "build of the engine."
            )
        records = json.loads(path.read_text()) if path.exists() else {}
    rows = [
        (StudyID(study), records[study][1]) for study in ids
        if study in records and records[study][1].strip()
    ]
    return pd.DataFrame(rows, columns=["study_id", "abstract"])


def register_tables(
    nl: NeurolangPDL,
    mask: nib.Nifti1Image,
    data_dir: Path,
    peak_data: pd.DataFrame,
    term_data: pd.DataFrame,
) -> pd.DataFrame:
    """Register the predicates every coordinate-based engine provides.

    Shared by the ``neurosynth`` and ``neurostore`` engines.

    Parameters
    ----------
    nl :
        Fresh :class:`~neurolang.frontend.NeurolangPDL` instance.
    mask :
        MNI template image defining the voxel grid.
    data_dir :
        Directory under which downloaded data is cached.
    peak_data :
        Columns ``i, j, k, study_id``, as returned by
        :func:`process_peaks`.
    term_data :
        Columns ``study_id, term, tfidf`` with ``StudyID`` ids.

    Returns
    -------
    The distinct ``study_id`` frame, for engines registering more
    study-level relations.

    """
    study_ids = peak_data[["study_id"]].drop_duplicates()

    nl.add_tuple_set(peak_data, name="peak_reported")
    nl.add_tuple_set(peak_mni(peak_data, mask), name="peak_mni")
    nl.add_tuple_set(study_ids, name="study")
    nl.add_tuple_set(term_data, name="term_in_study_tfidf")
    # The vocabulary, stored once rather than derived from the 1M-row
    # TF-IDF table in every query that uses it.
    nl.add_tuple_set(
        term_data[["term"]].drop_duplicates().reset_index(drop=True),
        name="term",
    )
    # Every (i, j, k) index of the mask grid. Use the image shape only:
    # ``get_fdata`` would load the full image as float64 to read it.
    nl.add_tuple_set(
        np.indices(mask.shape[:3]).reshape(3, -1).T,
        name="voxel",
    )
    # The indices of a grid are distinct: say so, sparing the queries that
    # use the 8.7M-row grid its deduplication.
    nl.symbol_table["voxel"].value.might_have_duplicates = False
    # The voxels of the cortex and of the brain on the grid of the mask:
    # resampling the atlas and the brain mask needs nilearn, which takes
    # seconds to import, so they are kept (see engine_cache).
    voxels = engine_cache.keep_arrays(
        getattr(nl, "engine_cache_dir", None), "voxels",
        engine_cache.grid_digest(mask),
        lambda: {
            "cortical": _cortical_voxels(fetch_cortical_atlas(data_dir), mask),
            "brain": _brain_voxels(mask),
        },
    )
    nl.add_tuple_set(voxels["cortical"], name="cortical_mask")
    nl.symbol_table["cortical_mask"].value.might_have_duplicates = False
    brain = voxels["brain"]
    nl.add_tuple_set(brain, name="brain_mask")
    nl.symbol_table["brain_mask"].value.might_have_duplicates = False
    # A uniform choice over the voxels of the brain, as selected_study is over
    # the studies: the average over the brain of the uniformity test.
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame(brain, columns=["i", "j", "k"]), name="selected_voxel"
    )
    nl.add_tuple_set(
        fetch_cognitive_atlas_tasks(data_dir), name="cognitive_task"
    )
    nl.add_uniform_probabilistic_choice_over_set(study_ids, name="selected_study")
    return study_ids


def peak_mni(peak_data: pd.DataFrame, mask: nib.Nifti1Image) -> pd.DataFrame:
    """The peaks of *peak_data* (``i, j, k, study_id`` on the grid of the
    *mask*) as MNI coordinates in millimetres: ``x, y, z, study_id``, the
    centres of their voxels (the reported coordinates at the default 1 mm
    resolution)."""
    xyz = nib.affines.apply_affine(
        mask.affine, peak_data[["i", "j", "k"]].to_numpy(dtype=float)
    )
    return pd.DataFrame({
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "study_id": peak_data["study_id"].to_numpy(),
    })


def fetch_cortical_atlas(data_dir: Path) -> nib.Nifti1Image:
    """The Harvard–Oxford cortical atlas, maximum probability at a 25%
    threshold, on its 2 mm grid; 0 is outside the cortex."""
    atlas = nilearn_datasets().fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm", data_dir=str(data_dir), verbose=0
    )
    return nilearn_image().load_img(atlas.maps)


def fetch_cognitive_atlas_tasks(data_dir: Path) -> pd.DataFrame:
    """The tasks of the Cognitive Atlas, from its API, as the columns
    ``task_id``, ``name`` and ``description``, stripped. A task without
    a description has an empty one; one without a name is left out."""
    path = fetch_files(
        str(Path(data_dir) / "cognitive_atlas"),
        [(
            "tasks.json", COGNITIVE_ATLAS_TASKS_URL,
            {"move": "tasks.json"},
        )],
        verbose=0,
    )[0]
    with open(path) as file:
        tasks = pd.DataFrame(json.load(file))
    tasks = pd.DataFrame({
        "task_id": tasks["id"],
        "name": tasks["name"].fillna("").str.strip(),
        "description": tasks["definition_text"].fillna("").str.strip(),
    })
    return tasks[tasks["name"] != ""].reset_index(drop=True)


def _brain_voxels(mask: nib.Nifti1Image) -> np.ndarray:
    """The (i, j, k) indices of the voxels of the *mask* grid inside the
    MNI152 brain mask of nilearn (bundled with it, nothing is downloaded),
    resampled to that grid by nearest neighbour: the whole brain, the
    subcortical structures and the cerebellum included."""
    brain = nilearn_datasets().load_mni152_brain_mask(resolution=2)
    resampled = nilearn_image().resample_to_img(
        brain, mask, interpolation="nearest"
    )
    return np.argwhere(np.asanyarray(resampled.dataobj) > 0)


def _cortical_voxels(
    atlas: nib.Nifti1Image, mask: nib.Nifti1Image
) -> np.ndarray:
    """The (i, j, k) indices of the voxels of the *mask* grid labelled
    by the *atlas*, resampled to that grid by nearest neighbour."""
    resampled = nilearn_image().resample_to_img(
        atlas, mask, interpolation="nearest"
    )
    return np.argwhere(np.asanyarray(resampled.dataobj) > 0)


UNKNOWN_SPACES = {"mni": "MNI", "tal": "TAL", "drop": ""}


def process_peaks(
    activations: pd.DataFrame,
    mni_mask: nib.Nifti1Image,
    study_id_type: type = StudyID,
    unknown_space: str = "mni",
) -> pd.DataFrame:
    """Convert MNI / Talairach peak coordinates to voxel indices.

    Only the ``TAL`` peaks are converted to MNI.  The peaks of any other
    space than ``MNI`` and ``TAL`` (Neurosynth's ``UNKNOWN``) are taken as
    MNI when *unknown_space* is ``"mni"``, as Neurosynth and NiMARE do,
    converted as Talairach when it is ``"tal"``, and left out when it is
    ``"drop"``.  The ``study_id`` of the result is cast to *study_id_type*.
    """
    if unknown_space not in UNKNOWN_SPACES:
        raise ValueError(
            f"unknown_space must be one of {', '.join(UNKNOWN_SPACES)}, "
            f"not {unknown_space!r}"
        )
    space = activations.space.where(
        activations.space.isin(["MNI", "TAL"]), UNKNOWN_SPACES[unknown_space]
    )
    peaks = activations[["x", "y", "z", "id"]].rename(
        columns={"id": "study_id"}
    )
    mni_peaks = peaks.loc[space == "MNI"]
    non_mni_peaks = peaks.loc[space == "TAL"]

    proj_mat = np.linalg.pinv(
        np.array(
            [
                [0.9254, 0.0024, -0.0118, -1.0207],
                [-0.0048, 0.9316, -0.0871, -1.7667],
                [0.0152, 0.0883, 0.8924, 4.0926],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).T
    )
    projected = np.round(
        np.dot(
            np.hstack(
                (
                    non_mni_peaks[["x", "y", "z"]].values,
                    np.ones((len(non_mni_peaks), 1)),
                )
            ),
            proj_mat,
        )[:, 0:3]
    )
    projected_df = pd.DataFrame(projected, columns=["x", "y", "z"])
    projected_df["study_id"] = non_mni_peaks["study_id"].values
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int, "study_id": study_id_type}
    )

    ijk_positions = np.round(
        nib.affines.apply_affine(
            np.linalg.inv(mni_mask.affine),
            peak_data[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    peak_data["i"] = ijk_positions[:, 0]
    peak_data["j"] = ijk_positions[:, 1]
    peak_data["k"] = ijk_positions[:, 2]
    peak_data = peak_data[["i", "j", "k", "study_id"]]
    return peak_data

