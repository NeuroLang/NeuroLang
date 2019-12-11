import numpy as np
import nibabel
import nilearn.plotting
import nilearn.datasets
import nilearn.surface

primary_visual_labels = {"visual_V1"}
primary_auditory_labels = {"PAC_TE10", "PAC_TE11", "PAC_TE12"}
primary_motor_labels = {"PMC_4p", "PMC_4a"}
primary_somatosensory_labels = {"PSC_1", "PSC_2", "PSC_3a", "PSC_3b"}


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


pmaps = load_sensory_motor_pmaps(
    "examples/global_connectivity/dset_mni+tlrc.BRIK"
)

sensory_motor_pmap = combine_pmaps_exclude_overlapping(pmaps.values())
sensory_motor_roi = pmap_to_roi(sensory_motor_pmap)

fsaverage = nilearn.datasets.fetch_surf_fsaverage()

# nilearn.plotting.plot_stat_map(sensory_motor_pmap, display_mode="ortho")


nilearn.plotting.plot_roi(sensory_motor_roi)

# nilearn.plotting.plot_surf_roi(
# surf_mesh=fsaverage["infl_left"],
# roi_map=nilearn.surface.vol_to_surf(
# nibabel.Nifti1Image(
# (sensory_motor_pmap.get_data() > 0).astype(int).astype(float),
# sensory_motor_pmap.affine,
# ),
# fsaverage["pial_left"],
# ),
# hemi="left",
# view="lateral",
# colorbar=True,
# bg_map=fsaverage["sulc_left"],
# bg_on_data=True,
# cmap=nilearn.plotting.cm.cold_white_hot,
# title="Sensory-motor cortices",
# )
