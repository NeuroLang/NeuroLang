import numpy as np
import nibabel
import nilearn.plotting
import nilearn.datasets
import nilearn.surface

sensory_motor_labels = {
    # primary visual (Brodmann area 17)
    "visual_V1",
    # primary auditory
    # "PAC_TE10",
    # "PAC_TE11",
    "PAC_TE12",
    # primary motor (Brodmann area 4)
    # "PMC_4p",
    "PMC_4a",
    # primary somatosensory (Brodmann areas 3, 2 and 1)
    # "PSC_1",
    # "PSC_2",
    # "PSC_3a",
    # "PSC_3b",
}


def load_sensory_motor_pmaps(path):
    img = nibabel.load(path)
    labels = np.array(img.header.get_volume_labels())
    label_to_pmap = dict()
    for label in sensory_motor_labels:
        label_idx = next(iter(np.argwhere(labels == label)))
        label_to_pmap[label] = nibabel.Nifti1Image(
            img.get_data()[..., label_idx] / 255, img.affine
        )
    return label_to_pmap


def combine_pmaps_exclude_overlapping(pmaps):
    any_pmap = next(iter(pmaps))
    affine = any_pmap.affine
    shape = any_pmap.get_data().shape
    counts = np.zeros(shape, dtype=np.int32)
    result_pmap = np.zeros(shape, dtype=np.float32)
    for pmap in pmaps:
        data = pmap.get_data()
        counts += (data > 0).astype(np.int32)
        result_pmap += data
    result_pmap[counts != 1] = 0
    return nibabel.Nifti1Image(result_pmap, affine)


pmaps = load_sensory_motor_pmaps(
    "examples/global_connectivity/dset_mni+tlrc.BRIK"
)

sensory_motor_pmap = combine_pmaps_exclude_overlapping(pmaps.values())

fsaverage = nilearn.datasets.fetch_surf_fsaverage()

nilearn.plotting.plot_stat_map(sensory_motor_pmap, display_mode="ortho")

# nilearn.plotting.view_img_on_surf(sensory_motor_pmap).open_in_browser()

nilearn.plotting.plot_surf_stat_map(
    surf_mesh=fsaverage["pial_left"],
    stat_map=nilearn.surface.vol_to_surf(
        sensory_motor_pmap, fsaverage["pial_left"]
    ),
    hemi="left",
    view="medial",
    colorbar=True,
    bg_map=fsaverage["sulc_left"],
    bg_on_data=True,
    cmap=nilearn.plotting.cm.cold_white_hot,
    # threshold=0.5,
    title="Sensory-motor cortices",
)
