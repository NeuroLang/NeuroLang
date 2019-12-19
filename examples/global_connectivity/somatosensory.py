import os

import pandas as pd
import numpy as np
import nibabel
import nilearn.plotting
import nilearn.datasets
import nilearn.surface
import neurosynth

primary_visual_labels = {"visual_V1"}
primary_auditory_labels = {"PAC_TE10", "PAC_TE11", "PAC_TE12"}
primary_motor_labels = {"PMC_4p", "PMC_4a"}
primary_somatosensory_labels = {"PSC_1", "PSC_2", "PSC_3a", "PSC_3b"}
sensory_motor_labels = (
    primary_visual_labels
    | primary_auditory_labels
    | primary_motor_labels
    | primary_somatosensory_labels
)


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
primary_somatosensory_pmap = combine_pmaps_max(
    [pmaps[label] for label in primary_somatosensory_labels]
)
primary_visual_roi = pmap_to_roi(primary_visual_pmap)
primary_auditory_roi = pmap_to_roi(primary_auditory_pmap)
primary_motor_roi = pmap_to_roi(primary_motor_pmap)
primary_somatosensory_roi = pmap_to_roi(primary_somatosensory_pmap)
sensory_motor_pmap = combine_pmaps_max(
    [
        primary_visual_pmap,
        primary_auditory_pmap,
        primary_motor_pmap,
        primary_somatosensory_pmap,
    ]
)
sensory_motor_roi = pmap_to_roi(sensory_motor_pmap)
sensory_motor_atlas = rois_to_atlas(
    [
        primary_visual_roi,
        primary_auditory_roi,
        primary_motor_roi,
        primary_somatosensory_roi,
    ]
)


fsaverage = nilearn.datasets.fetch_surf_fsaverage()


def plot_roi(roi):
    nilearn.plotting.plot_roi(
        roi, display_mode="x", cut_coords=np.linspace(-50, 0, 5), cmap="Dark2",
    )

plot_roi(sensory_motor_atlas)

forward_maps = pd.read_hdf(
    "examples/global_connectivity/neurosynth_forward_maps.h5", "forward_maps"
)


def build_forward_map(term):
    fmap = np.zeros(228453)
    voxel_ids = forward_maps[forward_maps.t == term].v.values.astype(np.int32)
    probabilities = forward_maps[forward_maps.t == term].probability.values
    fmap[voxel_ids] = probabilities
    masker = neurosynth.mask.Masker(
        os.path.join(
            neurosynth.__path__[0], "resources/MNI152_T1_2mm_brain.nii.gz",
        )
    )
    base_image = nibabel.load(
        os.path.join(
            neurosynth.__path__[0], "resources/MNI152_T1_2mm_brain.nii.gz",
        )
    )
    data = masker.unmask(fmap)
    return nibabel.Nifti1Image(data, affine=base_image.affine)


def neurosynth_forward_map_to_roi(pmap):
    new_data = (pmap.get_data() > 0.03).astype(int)
    new_data[new_data > 0] = np.random.randint(2, 7)
    return nibabel.Nifti1Image(new_data, pmap.affine)


dmn_pmap = build_forward_map("default mode")
cognitive_control_pmap = build_forward_map("cognitive control")
dmn_roi = neurosynth_forward_map_to_roi(dmn_pmap)
cognitive_control_roi = neurosynth_forward_map_to_roi(cognitive_control_pmap)


template = nilearn.datasets.load_mni152_template()
resampled_sensory_motor_roi = nilearn.image.resample_to_img(
    sensory_motor_roi, template
)
atlas = rois_to_atlas(
    [dmn_roi, cognitive_control_roi, resampled_sensory_motor_roi]
)
plot_roi(atlas)


# nilearn.plotting.plot_surf_roi(
# surf_mesh=fsaverage["infl_left"],
# roi_map=nilearn.surface.vol_to_surf(
# sensory_motor_roi, fsaverage["pial_left"],
# ),
# hemi="left",
# view="lateral",
# colorbar=True,
# bg_map=fsaverage["sulc_left"],
# bg_on_data=True,
# cmap=nilearn.plotting.cm.cold_white_hot,
# title="Sensory-motor cortices",
# )
