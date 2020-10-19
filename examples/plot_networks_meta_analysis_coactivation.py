import os

import neurosynth
import nibabel
import nilearn.datasets
import nilearn.plotting
import numpy as np
import pandas as pd
import scipy.special
from neurosynth.analysis.stats import one_way
from neurosynth.base.imageutils import map_peaks_to_image
from neurosynth.base.transformations import xyz_to_mat

from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend

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
fsaverage = nilearn.datasets.fetch_surf_fsaverage()

nl = ProbabilisticFrontend()

nl.add_tuple_set(
    [
        ("pcc", "dmn"),
        ("acc", "dmn"),
        ("na", "dmn"),
        ("lpc", "dmn"),
        ("itc", "dmn"),
        ("sfc", "dmn"),
        # ("ips", "attention"),
        # ("sma", "attention"),
        # ("presma", "attention"),
        # ("alns", "attention"),
        # ("fef", "attention"),
        # ("dlpfc", "attention"),
        # ("ipcs", "attention"),
        # ("voc", "attention"),
        # ("ipl", "attention"),
        # ("vips", "attention"),
    ],
    name="in_network",
)


def load_destrieux_atlas(nl):
    destrieux = nilearn.datasets.fetch_atlas_destrieux_2009()
    destrieux_map = nibabel.load(destrieux["maps"])
    d = []
    for label_number, name in destrieux["labels"]:
        if label_number == 0:
            continue
        name = name.decode()
        region = nl.create_region(destrieux_map, label=label_number)
        if region is None:
            continue
        name = name.replace("-", "_").replace(" ", "_")
        d.append((name.lower(), region))
    nl.add_tuple_set(d, name="destrieux")


with nl.environment as e:
    # define posterior cingulate cortex by combining posterior ventral
    # cingulate gyrus and posterior dorsal cingulate gyrus defined in the
    # Destrieux atlas
    e.in_pcc[e.r] = e.destrieux["L G_cingul-Post-ventral", e.r]
    e.in_pcc[e.r] = e.destrieux["L G_cingul-Post-dorsal", e.r]
    e.roi["pcc", e.region_union[e.r]] = e.in_pcc[e.r]

    # define anterior cingulate cortex from the anterior cingulate gyrus and
    # sulcus defined in the Destireux atlas
    e.roi["acc", e.r] = e.destrieux["L G_and_S_cingul-Ant", e.r]

    # define the nucleus accumbens from the accumbens defined in the
    # Harvard-Oxford subcortical atlas
    e.roi["na", e.r] = e.harvard_oxford_subcortical["Accumbens", e.r]

    # define the lateral parietal cortex from the supramarginal gyrus and
    # intraparietal sulcus defined in the destrieux atlas, and from the angular
    # gyrus and superior parietal lobule defined in the Harvard-Oxford cortical
    # atlas
    e.in_lpc[e.r] = e.destrieux["L G_pariet_inf-Supramar", e.r]
    e.in_lpc[e.r] = e.destrieux["L S_intrapariet_and_P_trans", e.r]
    e.in_lpc[e.r] = e.harvard_oxford_cortical["Angular Gyrus", e.r]
    e.in_lpc[e.r] = e.harvard_oxford_cortical["Superior Parietal Lobule", e.r]
    e.roi["lpc", e.region_union[e.r]] = e.in_lpc[e.r]

    # define inferior temporal cortex from gyri defined in Harvard-Oxford
    # cortical atlas
    e.in_itc["itc", e.r] = e.harvard_oxford_cortical[
        "Inferior Temporal Gyrus, anterior division", e.r
    ]
    e.in_itc["itc", e.r] = e.harvard_oxford_cortical[
        "Inferior Temporal Gyrus, posterior division", e.r
    ]
    e.in_itc["itc", e.r] = e.harvard_oxford_cortical[
        "Inferior Temporal Gyrus, tmeporooccipital part", e.r
    ]
    e.roi["itc", e.region_union[e.r]] = e.in_lpc[e.r]

    # define superior frontal cortex from superior frontal gyrus defined in
    # Harvard-Oxford cortical atlas
    e.roi["sfc", e.r] = e.harvard_oxford_cortical[
        "Superior Frontal Gyrus", e.r
    ]

    # define network based on their set of regions of interest
    e.network[e.network, e.region_union[e.r]] = (
        e.roi[e.region, e.r] & e.in_network[e.region, e.network]
    )


def xyz_to_voxel_id(x, y, z, masker):
    # dirty way to get a voxel id from xyz coordinates
    xyz = np.array([[x, y, z]])
    ijk = xyz_to_mat(xyz)
    img = map_peaks_to_image(ijk, r=2, header=masker.get_header())
    return np.argwhere(masker.mask(img) > 0).flatten()[0]


seed_voxels = {
    "acc": (-2, 46, -4),
    # "cs": (-34, -26, 60),
    # left intraparietal sulculs
    # -> fronto-parietal "attention" network
    # "left_ips": (-26, -58, 48),
}
seed_voxel_ids = {
    region_name: xyz_to_voxel_id(*coordinates, ns_masker)
    for region_name, coordinates in seed_voxels.items()
}
SeedVoxel = nl.add_tuple_set(
    {
        (seed_voxel_id,)
        for seed_voxel_name, seed_voxel_id in seed_voxel_ids.items()
    },
    name="SeedVoxel",
)
VoxelReported = nl.load_neurosynth_reported_activations(name="VoxelReported")
VoxelNotReported = nl.load_neurosynth_non_reported_activations(
    name="VoxelNotReported",
    voxel_ids=set(seed_voxel_ids.values()),
)
StudyID = nl.load_neurosynth_study_ids(name="StudyID")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    StudyID.value,
    name="SelectedStudy",
)

with nl.environment as e:
    e.J1[e.v1, e.v2, e.PROB[e.v1, e.v2]] = (
        SelectedStudy[e.s]
        & e.VoxelReported[e.s, e.v1]
        & e.VoxelReported[e.s, e.v2]
        & e.SeedVoxel[e.v2]
    )
    e.M[e.v, e.PROB[e.v]] = e.SelectedStudy[e.s] & e.VoxelReported[e.s, e.v]
    e.M1[e.v, e.PROB[e.v]] = (
        e.SelectedStudy[e.s] & e.VoxelReported[e.s, e.v] & e.SeedVoxel[e.v]
    )
    e.J0[e.v1, e.v2, e.PROB[e.v1, e.v2]] = (
        SelectedStudy[e.s]
        & e.VoxelReported[e.s, e.v1]
        & e.VoxelNotReported[e.s, e.v2]
        & e.SeedVoxel[e.v2]
    )
    e.M0[e.v, e.PROB[e.v]] = (
        e.SelectedStudy[e.s] & e.VoxelNotReported[e.s, e.v] & e.SeedVoxel[e.v]
    )
    e.P1[e.v1, e.v2, e.p] = (
        e.J1[e.v1, e.v2, e.p_j] & e.M1[e.v2, e.p_m] & (e.p == (e.p_j / e.p_m))
    )
    e.P0[e.v1, e.v2, e.p] = (
        e.J0[e.v1, e.v2, e.p_j] & e.M0[e.v2, e.p_m] & (e.p == (e.p_j / e.p_m))
    )

with nl.environment as e:
    p = nl.query((e.v, e.p), e.M[e.v, e.p])
    p12 = nl.query((e.v1, e.v2, e.p), e.J1(e.v1, e.v2, e.p))
    p1 = nl.query((e.v1, e.v2, e.p), e.P1(e.v1, e.v2, e.p))
    p0 = nl.query((e.v1, e.v2, e.p), e.P0(e.v1, e.v2, e.p))

df_p = pd.DataFrame(p, columns=["v", "p"])
df_p0 = pd.DataFrame(p0, columns=["v1", "v2", "p0"])
df_p1 = pd.DataFrame(p1, columns=["v1", "v2", "p1"])
df_p12 = pd.DataFrame(p12, columns=["v1", "v2", "p12"])

d = (
    df_p0.merge(df_p1)
    .merge(df_p.rename(columns={"p": "marg_v1", "v": "v1"}))
    .merge(df_p.rename(columns={"p": "marg_v2", "v": "v2"}))
    .merge(df_p12)
)
n_studies = len(StudyID.value)
d["m"] = d.marg_v2 * n_studies
d["k"] = d.p12 * n_studies
d["n"] = d.marg_v1 * n_studies

d["log_lr"] = (
    d.k * np.log(d.p1)
    + (d.n - d.k) * np.log(1 - d.p1)
    + (d.m - d.k) * np.log(d.p0)
    + (n_studies - d.n - d.m + d.k) * np.log(1 - d.p0)
) - (
    d.k * np.log(d.marg_v2)
    + (d.n - d.k) * np.log(1 - d.marg_v2)
    + (d.m - d.k) * np.log(d.marg_v2)
    + (n_studies - d.n - d.m + d.k) * np.log(1 - d.marg_v2)
)

d["pval"] = scipy.special.chdtrc(1, d.log_lr)
threshold = 0.01 / n_voxels

for seed_voxel, dg in d.groupby("v2"):
    data = np.zeros(shape=n_voxels)
    data[dg["v1"]] = dg["pval"] <= threshold
    img = nibabel.Nifti1Image(masker.unmask(data), ns_affine)
    nilearn.plotting.plot_glass_brain(img)
