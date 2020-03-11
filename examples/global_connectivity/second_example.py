import os
import typing

import scipy
import pandas as pd
import numpy as np
import nibabel
import nilearn.plotting
import nilearn.datasets
import nilearn.surface
import neurosynth
import neurosynth.analysis.stats

import neurolang as nl
import neurolang.regions
import neurolang.datalog
import neurolang.datalog.chase

import os
from collections import defaultdict
import typing
import itertools

import neurosynth as ns
from neurosynth.base import imageutils
from neurosynth import Dataset
from nilearn import plotting
import numpy as np
import pandas as pd

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.expression_walker import (
    ExpressionBasicEvaluator,
    ReplaceSymbolsByConstants,
)
from neurolang.datalog.expressions import Implication, Fact, Conjunction
from neurolang.datalog.instance import SetInstance
from neurolang.probabilistic.expressions import ProbabilisticPredicate
from neurolang.probabilistic.probdatalog import ProbDatalogProgram
from neurolang.probabilistic.probdatalog_gm import (
    full_observability_parameter_estimation,
    ExtendedAlgebraSet,
    succ_query,
    ExtendedRelationalAlgebraSolver,
    DivideColumns,
    NaturalJoin,
)

class Datalog(
    nl.logic.expression_processing.TranslateToLogic,
    nl.datalog.aggregation.DatalogWithAggregationMixin,
    nl.datalog.DatalogProgram,
    nl.ExpressionBasicEvaluator,
):
    @staticmethod
    def function_region_union(
        region_set: typing.AbstractSet[nl.regions.Region],
    ) -> nl.regions.Region:
        return nl.regions.region_union(region_set)


class Chase(
    nl.datalog.aggregation.Chase, nl.datalog.chase.ChaseGeneral,
):
    pass


primary_visual_labels = {"visual_V1"}

ns_base_img = nibabel.load(
    os.path.join(
        neurosynth.__path__[0], "resources/MNI152_T1_2mm_brain.nii.gz",
    )
)
ns_masker = neurosynth.mask.Masker(
    os.path.join(
        neurosynth.__path__[0], "resources/MNI152_T1_2mm_brain.nii.gz",
    )
)
ns_affine = ns_base_img.affine
fsaverage = nilearn.datasets.fetch_surf_fsaverage()


def load_visual_pmap(path):
    img = nibabel.load(path)
    labels = np.array(img.header.get_volume_labels())
    label_to_pmap = dict()
    for label in primary_visual_labels:
        label_idx = next(iter(np.argwhere(labels == label)))
        label_to_pmap[label] = nibabel.Nifti1Image(
            (img.get_data()[..., label_idx] / 255).squeeze(), img.affine
        )
    return label_to_pmap["visual_V1"]


def pmap_to_roi(pmap):
    return nibabel.Nifti1Image(
        (pmap.get_data() > 0).astype(np.float32), pmap.affine
    )


def rois_to_atlas(rois):
    any_roi = next(iter(rois))
    affine = any_roi.affine
    shape = any_roi.get_data().shape
    new_data = np.zeros(shape, dtype=np.int32)
    for i, roi in enumerate(rois):
        new_data[roi.get_data() > 0.01] = i + 2
    return nibabel.Nifti1Image(new_data, affine)


def plot_roi(roi):
    nilearn.plotting.plot_roi(
        roi, display_mode="x", cut_coords=np.linspace(-50, 0, 5), cmap="Set3",
    )


def plot_stat_map(stat_map):
    nilearn.plotting.plot_stat_map(
        stat_map,
        display_mode="x",
        cut_coords=np.linspace(-30, 0, 5),
    )

def roi_to_nl_region(roi):
    r = nl.regions.ExplicitVBR(
        voxels=np.argwhere(roi.get_data() > 0),
        affine_matrix=roi.affine,
        image_dim=roi.get_data().shape,
    )
    r = nl.regions.ExplicitVBR(
        voxels=r.to_ijk(ns_affine),
        affine_matrix=ns_affine,
        image_dim=ns_base_img.get_data().shape,
    )
    return r

ns_query_result = pd.read_hdf(
    "examples/global_connectivity/visual_estimated_pFgA.h5", "estimation"
)

v = nl.Symbol("v")
PrimaryVisual = nl.Symbol("PrimaryVisual")
PrimaryAuditory = nl.Symbol("PrimaryAuditory")
PrimaryMotor = nl.Symbol("PrimaryMotor")
Somatosensory = nl.Symbol("Somatosensory")


facts = [
    nl.datalog.Fact(
        nl.Symbol("sensory_motor_region")(
            nl.Constant[nl.regions.ExplicitVBR](region),
        )
    )
    for region_name, region in region_rois.items()
]

r1 = nl.logic.Implication(
    nl.Symbol("sensory_motor")(
        nl.datalog.aggregation.AggregationApplication(
            nl.Symbol("region_union"), (nl.Symbol("x"),)
        )
    ),
    nl.Symbol("sensory_motor_region")(nl.Symbol("x")),
)

r2 = nl.logic.Implication(
    nl.Symbol("dmn_region")(
        nl.datalog.aggregation.AggregationApplication(
            nl.Symbol("region_union"), (nl.Symbol("y"),)
        )
    ),
    nl.logic.Conjunction(
        [
            nl.Symbol("voxel_id_region")(nl.Symbol("y"), nl.Symbol("x")),
            nl.Symbol("neurosynth_term_voxel_id")(
                nl.Constant("default mode"), nl.Symbol("y")
            ),
        ]
    ),
)

r3 = nl.logic.Implication(
    nl.Symbol("cognitive_control_region")(
        nl.datalog.aggregation.AggregationApplication(
            nl.Symbol("region_union"), (nl.Symbol("y"),)
        )
    ),
    nl.logic.Conjunction(
        [
            nl.Symbol("voxel_id_region")(nl.Symbol("y"), nl.Symbol("x")),
            nl.Symbol("neurosynth_term_voxel_id")(
                nl.Constant("cognitive control"), nl.Symbol("y")
            ),
        ]
    ),
)

program_code = nl.expressions.ExpressionBlock(facts + [r1, r2, r3])


def build_vid_to_region_tuple_set():
    voxels = np.ones(shape=ns_masker.mask(ns_base_img.get_data()).shape)
    unmasked = ns_masker.unmask(voxels)
    ijk = np.argwhere(unmasked > 0)
    regions = [
        nl.regions.ExplicitVBR(
            [coords],
            affine_matrix=ns_affine,
            image_dim=ns_base_img.get_data().shape,
        )
        for coords in ijk
    ]
    return set(
        (vid, region) for vid, region in zip(range(voxels.shape[0]), regions)
    )


dl = Datalog()
dl.add_extensional_predicate_from_tuples(
    nl.Symbol("voxel_id_region"), build_vid_to_region_tuple_set()
)
dl.add_extensional_predicate_from_tuples(
    nl.Symbol("neurosynth_term_voxel_id"),
    [
        (row.t, row.v)
        for _, row in ns_query_result[
            ns_query_result.probability > 0.03
        ].iterrows()
    ],
)
dl.walk(program_code)

chase = Chase(dl)
solution = chase.build_chase_solution()


def get_region(name):
    return next(solution[name].value.unwrapped_iter())[0]


regions = {
    name: get_region(name)
    for name in [
        "dmn_region",
        "cognitive_control_region",
        "sensory_motor_region",
    ]
}


def region_to_roi(region):
    image_dim = ns_base_img.get_data().shape
    new_data = np.zeros(shape=image_dim)
    img = region.spatial_image()
    if img.get_data().shape != image_dim:
        img = region.to_explicit_vbr(ns_affine, image_dim).spatial_image()
    new_data[img.get_data() > 0] = 1
    return nibabel.Nifti1Image(new_data.astype(int), ns_affine)


def regions_to_atlas(regions):
    image_dim = ns_base_img.get_data().shape
    new_data = np.zeros(shape=image_dim)
    for i, region in enumerate(regions):
        img = region.spatial_image()
        if img.get_data().shape != image_dim:
            img = region.to_explicit_vbr(ns_affine, image_dim).spatial_image()
        new_data[img.get_data() > 0] = i
    return nibabel.Nifti1Image(new_data.astype(int), ns_affine)


dmn_ns_data = pd.read_hdf(
    "examples/global_connectivity/neurosynth_association-test_z_FDR_0.01.h5",
    "default mode",
)
cognitive_control_ns_data = pd.read_hdf(
    "examples/global_connectivity/neurosynth_association-test_z_FDR_0.01.h5",
    "cognitive control",
)

atlas = regions_to_atlas(regions.values())
plot_roi(atlas)

dmn_ns = nibabel.Nifti1Image(
    ns_masker.unmask(dmn_ns_data.clip(0, np.inf)), ns_affine
)
cognitive_control_ns = nibabel.Nifti1Image(
    ns_masker.unmask(cognitive_control_ns_data.clip(0, np.inf)), ns_affine
)

dmn_ns_roi = pmap_to_roi(dmn_ns)
cognitive_control_ns_roi = pmap_to_roi(cognitive_control_ns)
dmn_ns_region = roi_to_nl_region(dmn_ns_roi)
cognitive_control_ns_region = roi_to_nl_region(cognitive_control_ns_roi)

atlas = regions_to_atlas(
    [
        dmn_ns_region,
        cognitive_control_ns_region,
        get_region("sensory_motor_region"),
    ]
)
plot_roi(atlas)

for region in [
    dmn_ns_region,
    cognitive_control_ns_region,
    get_region("sensory_motor_region"),
]:
    plot_roi(region_to_roi(region))


def p_to_z(p, sign):
    p = p / 2  # convert to two-tailed
    # prevent underflow
    p[p < 1e-240] = 1e-240
    # Convert to z and assign tail
    z = np.abs(scipy.stats.norm.ppf(p)) * sign
    # Set very large z's to max precision
    z[np.isinf(z)] = scipy.stats.norm.ppf(1e-240) * -1
    return z


itp_co_activation = pd.read_hdf(
    "examples/global_connectivity/interpretations_co_activation_visual.h5",
    "value"
)

itp_term_in_study = pd.read_hdf(
    "examples/global_connectivity/interpretations_term_in_study_visual.h5", "value"
)


def get_n_selected_active_voxels(term):
    n_selected_active_voxels = np.zeros(228453)
    d = (
        itp_co_activation.loc[itp_co_activation.t == term]
        .groupby("v")
        .count()
        .reset_index()
    )
    n_selected_active_voxels[d.v.values] = d.__interpretation_id__.values
    return n_selected_active_voxels


def get_n_unselected_active_voxels(term):
    n_unselected_active_voxels = np.zeros(228453)
    d = (
        itp_co_activation.loc[itp_co_activation.t != term]
        .groupby("v")
        .count()
        .reset_index()
    )
    n_unselected_active_voxels[d.v.values] = d.__interpretation_id__.values
    return n_unselected_active_voxels


def get_pAgF_from_nl_query_result(term):
    pAgF = np.zeros(228453)
    d = ns_query_result[ns_query_result.t == term]
    pAgF[d.v.values] = d.probability.values
    return pAgF


def correct_pmap(term, q=0.1):
    n_selected = itp_term_in_study.loc[itp_term_in_study.t == term].shape[0]
    n_unselected = itp_term_in_study.loc[itp_term_in_study.t != term].shape[0]
    n_selected_active_voxels = get_n_selected_active_voxels(term)
    n_unselected_active_voxels = get_n_unselected_active_voxels(term)
    # Two-way chi-square for specificity of activation
    cells = np.squeeze(
        np.array(
            [
                [n_selected_active_voxels, n_unselected_active_voxels],
                [
                    n_selected - n_selected_active_voxels,
                    n_unselected - n_unselected_active_voxels,
                ],
            ]
        ).T
    )
    p_vals = neurosynth.analysis.stats.two_way(cells)
    p_vals[p_vals < 1e-240] = 1e-240
    # pAgF = n_selected_active_voxels * 1.0 / n_selected
    pAgF = visual_pAgF
    pAgU = n_unselected_active_voxels * 1.0 / n_unselected
    z_sign = np.sign(pAgF - pAgU).ravel()
    pFgA_z = p_to_z(p_vals, z_sign)
    fdr_thresh = neurosynth.analysis.stats.fdr(p_vals, q)
    pFgA_z_FDR = neurosynth.imageutils.threshold_img(
        pFgA_z, fdr_thresh, p_vals, mask_out="above"
    )
    return pFgA_z_FDR


corrected_dmn_pmap = correct_pmap("default mode")
corrected_dmn_roi = (
    corrected_dmn_pmap.clip(0, np.inf) / corrected_dmn_pmap.max()
)
corrected_dmn_roi[corrected_dmn_roi > 0] = 1
corrected_dmn_roi_img = nibabel.Nifti1Image(
    ns_masker.unmask(corrected_dmn_roi), ns_affine
)

corrected_cognitive_control_pmap = correct_pmap("cognitive control")
corrected_cognitive_control_roi = (
    corrected_cognitive_control_pmap.clip(0, np.inf)
    / corrected_cognitive_control_pmap.max()
)
corrected_cognitive_control_roi[corrected_cognitive_control_roi > 0] = 1
corrected_cognitive_control_roi_img = nibabel.Nifti1Image(
    ns_masker.unmask(corrected_cognitive_control_roi), ns_affine
)

plot_roi(
    rois_to_atlas([corrected_dmn_roi_img, corrected_cognitive_control_roi_img])
)

plot_roi(corrected_dmn_roi_img)
plot_roi(corrected_cognitive_control_roi_img)

sensory_motor_roi_img = region_to_roi(get_region("sensory_motor_region"))

plot_roi(
    regions_to_atlas(
        [
            roi_to_nl_region(new_roi),
            roi_to_nl_region(corrected_dmn_roi_img),
            roi_to_nl_region(corrected_dmn_roi_img),
            roi_to_nl_region(corrected_cognitive_control_roi_img),
            roi_to_nl_region(new_roi),
        ]
    )
)
