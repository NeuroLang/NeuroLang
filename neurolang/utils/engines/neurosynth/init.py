"""
NeuroSynth engine initialization.

Registers the following predicates:

* ``peak_reported(i, j, k, study_id)`` — reported activation peaks
  converted to voxel coordinates.
* ``study(study_id)`` — all study identifiers.
* ``term_in_study_tfidf(term, tfidf, study_id)`` — term frequency–inverse
  document frequency values.
* ``selected_study(study_id)`` — uniform probabilistic choice over
  studies.
* ``voxel(i, j, k)`` — every voxel inside the MNI mask.

Predicate names are lowercase to match SQUALL's case-folding convention
(see ``intransitive`` in ``squall_syntax_lark.py``).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.frontend import NeurolangPDL
from neurolang.frontend.neurosynth_utils import (
    StudyID,
    fetch_neurosynth_peak_data,
    get_ns_term_study_associations,
)


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path
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

    """
    activations = fetch_neurosynth_peak_data(data_dir, verbose=0)
    peak_data = _process_peaks(activations, mask)
    study_ids = peak_data[["study_id"]].drop_duplicates()

    term_data = get_ns_term_study_associations(
        data_dir, verbose=0, convert_study_ids=True, tfidf_threshold=1e-3
    )
    term_data = term_data.rename(columns={"id": "study_id"})

    nl.add_tuple_set(peak_data, name="peak_reported")
    nl.add_tuple_set(study_ids, name="study")
    nl.add_tuple_set(term_data, name="term_in_study_tfidf")
    nl.add_tuple_set(
        np.hstack(
            np.meshgrid(
                *(np.arange(0, dim) for dim in mask.get_fdata().shape)
            )
        )
        .swapaxes(0, 1)
        .reshape(3, -1)
        .T,
        name="voxel",
    )
    nl.add_uniform_probabilistic_choice_over_set(study_ids, name="selected_study")

def _process_peaks(
    activations: pd.DataFrame, mni_mask: nib.Nifti1Image
) -> pd.DataFrame:
    """Convert MNI / Talairach peak coordinates to voxel indices."""
    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    non_mni_peaks = activations.loc[activations.space != "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})

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
    projected_df = pd.DataFrame(
        np.hstack(
            [projected, non_mni_peaks[["study_id"]].values]
        ),
        columns=["x", "y", "z", "study_id"],
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int}
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



