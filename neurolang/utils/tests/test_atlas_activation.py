"""Tests of the studies activating the regions of an atlas."""

import nibabel as nib
import numpy as np
import pytest

from neurolang.utils.engine_registry import atlas_activation


def atlas(affine):
    """A 10 x 10 x 10 atlas: region 1 is the voxel (2, 2, 2), region 2 the
    voxels (7, 7, 7) and (7, 7, 8)."""
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2, 2, 2] = 1
    data[7, 7, 7] = data[7, 7, 8] = 2
    return nib.Nifti1Image(data, affine)


NAMES = {1: "left", 2: "right"}
GRID_2MM = np.diag([2.0, 2.0, 2.0, 1.0])


def centre(affine, ijk):
    return nib.affines.apply_affine(affine, np.array(ijk, dtype=float))


@pytest.mark.parametrize("affine", [
    GRID_2MM,
    # flipped x axis and an offset origin, as the Schaefer atlas has
    np.array([[-2.0, 0, 0, 18.0], [0, 2.0, 0, -10.0], [0, 0, 2.0, 4.0],
              [0, 0, 0, 1.0]]),
])
def test_peaks_within_the_radius_activate_the_region(affine):
    left = centre(affine, (2, 2, 2))
    right = centre(affine, (7, 7, 8))
    peaks = np.array([
        left + [3.0, 0, 0],      # exactly 3 mm: inside
        left + [0, 3.01, 0],     # just beyond 3 mm
        right + [1.0, 1.0, 1.0],  # sqrt(3) mm from (7, 7, 8)
        right + [0, 0, 40.0],    # far
    ])
    studies = np.array([10, 11, 12, 13])
    pairs = atlas_activation(atlas(affine), NAMES, peaks, studies, 3.0)
    assert pairs == [("left", 10), ("right", 12)]


def test_each_study_once_per_region_and_chunks():
    affine = GRID_2MM
    peaks = np.repeat([centre(affine, (7, 7, 7))], 5, axis=0)
    studies = np.array([1, 1, 2, 2, 3])
    pairs = atlas_activation(atlas(affine), NAMES, peaks, studies, 3.0,
                             chunk=2)
    assert pairs == [("right", 1), ("right", 2), ("right", 3)]


def test_peaks_outside_the_grid():
    peaks = np.array([[-50.0, -50.0, -50.0], [500.0, 0, 0]])
    assert atlas_activation(
        atlas(GRID_2MM), NAMES, peaks, np.array([1, 2]), 3.0
    ) == []
