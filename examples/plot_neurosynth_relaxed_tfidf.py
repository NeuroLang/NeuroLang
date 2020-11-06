import os
import typing

import neurosynth
import nibabel
import nilearn.plotting
import numpy as np

from neurolang.frontend import NeurolangPDL


def tfidf_to_probability(
    tfidf: typing.Union[float, np.array],
    alpha: float = 3000,
    tau: float = 0.01,
) -> typing.Union[float, np.array]:
    """
    Threshold TFIDF features to interpret them as probabilities using a sigmoid
    function.

    The formula for this function is

        omega(x ; alpha, tau) = sigma(alpha * (x - tau))

    where sigma is the sigmoid function.

    Parameters
    ----------
    tfidf : float or np.array of floats
        TFIDF (Term Frequency Inverse Document Frequency) features.

    alpha : float
        Parameter used to control the smoothing of the thresholding by the
        sigmoid curve. The larger the value of alpha, the smoother the
        resulting values. The smaller the value of alpha, the closer this
        function will be to a hard-thresholding 1[x > tau].

    tau : float
        Threshold at which the function is centered.

    Returns
    -------
    float or np.array of floats
        Thresholded values between 0 and 1 that can be interpreted as
        probabilities in a probabilistic model.

    """
    return 1 / (1 + np.exp(-alpha * (tfidf - tau)))


term_1 = "memory"
term_2 = "auditory"
terms = [term_1, term_2]

nl = NeurolangPDL()
VoxelReported = nl.load_neurosynth_reported_activations(name="VoxelReported")
# TODO: currently, the process of adding these probabilistic facts cannot be
# done directly within the program because query-based probabilistic facts are
# not yet available. the definition of TermInStudy probabilistic facts within
# the program will later be possible using a rule
# TermInStudy(t, s) : sigmoid(alpha * (x - tau)) :- NeurosynthTFIDF(x, t, s)
# where x is the TFIDF feature stored within the NeurosynthTFIDF deterministic
# relation (loaded from the Neurosynth dataset)
study_tfidf = nl.neurosynth_db.ns_study_tfidf_feature_for_terms(terms)
study_tfidf["prob"] = tfidf_to_probability(study_tfidf["tfidf"])
# Create a TermInStudy probabilistic relation that model the association
# between a term and a study probabilistically from soft-thresholded TFIDF
# features
TermInStudy = nl.add_probabilistic_facts_from_tuples(
    set(
        study_tfidf[["prob", "term", "study_id"]].itertuples(
            name=None, index=False
        )
    ),
    name="TermInStudy",
    type_=typing.Tuple[float, str, int],
)
# Load all study IDs (PMIDs) in the Neurosynth database
StudyID = nl.load_neurosynth_study_ids(name="StudyID")
# Uniform probabilistic choice over study IDs
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    StudyID.value,
    name="SelectedStudy",
)
with nl.environment as e:
    e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
    e.Activation[e.v] = e.SelectedStudy[e.s] & e.VoxelReported[e.v, e.s]
    # The Query rule represents the calculation of conditional probabilities
    # Prob[ Activation(v) | TermAssociation(t1) and TermAssociation(t2) ]
    # for each voxel v, and pair of terms (t1, t2). This results in a
    # probabilistic brain map for activations in studies related to both terms
    # t1 and t2 (two-term conjunctive query).
    e.Query[e.v, e.PROB[e.v]] = (e.Activation[e.v]) // (
        e.TermAssociation["auditory"] & e.TermAssociation["memory"]
    )
    # Run the query
    result = nl.query((e.v, e.probability), e.Query[e.v, e.probability])

ns_base_img = nibabel.load(
    os.path.join(
        neurosynth.__path__[0],
        "resources/MNI152_T1_2mm_brain.nii.gz",
    )
)
ns_masker = neurosynth.mask.Masker(
    os.path.join(
        neurosynth.__path__[0],
        "resources/MNI152_T1_2mm_brain.nii.gz",
    )
)
n_voxels = ns_masker.get_mask().shape[0]
ns_affine = ns_base_img.affine

df = result.as_pandas_dataframe()
df.columns = ["voxel_id", "probability"]
stat_map = np.zeros(shape=n_voxels)
stat_map[df["voxel_id"]] = df["probability"]
img = nibabel.Nifti1Image(ns_masker.unmask(stat_map), ns_affine)
nilearn.plotting.plot_glass_brain(
    img,
    title=f"{term_1} & {term_2}",
    colorbar=True,
)
