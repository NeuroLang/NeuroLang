import nibabel as nib
import numpy as np

from neurolang_apps.utils.engines.neurosynth.init import _brain_voxels


def test_brain_mask_covers_the_whole_brain_on_the_grid_of_the_engine():
    affine = np.array(
        [[2.0, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]]
    )
    mask = nib.Nifti1Image(np.zeros((91, 109, 91), np.uint8), affine)
    voxels = {tuple(v) for v in _brain_voxels(mask)}
    inverse = np.linalg.inv(affine)

    def voxel_of(x, y, z):
        return tuple(np.round(inverse @ [x, y, z, 1])[:3].astype(int))

    # Hippocampus, thalamus and cerebellum are in the brain; a point in
    # the skull's front, outside it, is not.
    for point in [(-28, -20, -14), (10, -18, 8), (20, -60, -30)]:
        assert voxel_of(*point) in voxels, point
    assert voxel_of(0, 80, 0) not in voxels
    assert 150_000 < len(voxels) < 300_000
