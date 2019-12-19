import os
import typing

import pandas as pd
import numpy as np
import nibabel
import nilearn.plotting
import nilearn.datasets
import nilearn.surface
import neurosynth
import neurolang as nl
import neurolang.regions
import neurolang.datalog
import neurolang.datalog.chase


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
primary_auditory_labels = {"PAC_TE10", "PAC_TE11", "PAC_TE12"}
primary_motor_labels = {"PMC_4p", "PMC_4a"}
somatosensory_labels = {"PSC_1", "PSC_2", "PSC_3a", "PSC_3b"}
sensory_motor_labels = (
    primary_visual_labels
    | primary_auditory_labels
    | primary_motor_labels
    | somatosensory_labels
)

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


def load_sensory_motor_pmaps(path):
    img = nibabel.load(path)
    labels = np.array(img.header.get_volume_labels())
    label_to_pmap = dict()
    for label in sensory_motor_labels:
        label_idx = next(iter(np.argwhere(labels == label)))
        label_to_pmap[label] = nibabel.Nifti1Image(
            (img.get_data()[..., label_idx] / 255).squeeze(), img.affine
        )
    return label_to_pmap


def combine_pmaps_max(pmaps):
    affine = next(iter(pmaps)).affine
    new_data = np.max(
        np.vstack([pmap.get_data()[np.newaxis, ...] for pmap in pmaps]), axis=0
    )
    return nibabel.Nifti1Image(new_data, affine)


def combine_pmaps_exclude_overlapping(pmaps):
    any_pmap = next(iter(pmaps))
    affine = any_pmap.affine
    shape = any_pmap.get_data().shape
    counts = np.zeros(shape, dtype=np.int32)
    new_data = np.zeros(shape, dtype=np.float32)
    for pmap in pmaps:
        data = pmap.get_data()
        counts += (data > 0).astype(np.int32)
        new_data += data
    new_data[counts != 1] = 0
    return nibabel.Nifti1Image(new_data, affine)


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
        new_data[roi.get_data() > 0] = i
    return nibabel.Nifti1Image(new_data, affine)


def plot_roi(roi):
    nilearn.plotting.plot_roi(
        roi, display_mode="x", cut_coords=np.linspace(-50, 0, 5), cmap="Dark2",
    )


pmaps = load_sensory_motor_pmaps(
    "examples/global_connectivity/dset_mni+tlrc.BRIK"
)
primary_visual_pmap = combine_pmaps_max(
    [pmaps[label] for label in primary_visual_labels]
)
primary_auditory_pmap = combine_pmaps_max(
    [pmaps[label] for label in primary_auditory_labels]
)
primary_motor_pmap = combine_pmaps_max(
    [pmaps[label] for label in primary_motor_labels]
)
somatosensory_pmap = combine_pmaps_max(
    [pmaps[label] for label in somatosensory_labels]
)
primary_visual_roi = pmap_to_roi(primary_visual_pmap)
primary_auditory_roi = pmap_to_roi(primary_auditory_pmap)
primary_motor_roi = pmap_to_roi(primary_motor_pmap)
somatosensory_roi = pmap_to_roi(somatosensory_pmap)

v = nl.Symbol("v")
PrimaryVisual = nl.Symbol("PrimaryVisual")
PrimaryAuditory = nl.Symbol("PrimaryAuditory")
PrimaryMotor = nl.Symbol("PrimaryMotor")
Somatosensory = nl.Symbol("Somatosensory")


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


region_rois = dict(
    primary_visual=roi_to_nl_region(primary_visual_roi),
    primary_auditory=roi_to_nl_region(primary_auditory_roi),
    primary_motor=roi_to_nl_region(primary_motor_roi),
    somatosensory=roi_to_nl_region(somatosensory_roi),
)


ns_query_result = pd.read_hdf(
    "examples/global_connectivity/neurosynth_forward_maps.h5", "forward_maps"
)

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
        nl.regions.ExplicitVBR([coords], affine_matrix=ns_affine)
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
            ns_query_result.probability > 0.02
        ].iterrows()
    ],
)
dl.walk(program_code)

chase = Chase(dl)
solution = chase.build_chase_solution()
