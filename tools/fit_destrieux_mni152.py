"""
Fit the scales and shifts that register nilearn's Destrieux atlas to the
MNI152 template the viewer shows (``neurolang.atlases.DESTRIEUX_TO_MNI152``).

nilearn's volumetric Destrieux 2009 atlas is derived from FreeSurfer's
fsaverage: it is about 3% smaller than the MNI152 2009c template, and off
by a couple of millimetres. The fit maximises how much of its cortex lies
in the template's grey matter: the sum, over the voxels of its regions, of
the grey-matter probability less one half, so that a voxel of a region is
counted for when it is more likely grey matter than not, and against it
otherwise (shrinking the regions into the densest grey matter does not
pay). The transform is axis-aligned, a scale and a shift per axis about
the centre ``CENTRE``, so that the atlas stays on a grid of the axes.

Usage::

    python tools/fit_destrieux_mni152.py DATA_DIR

with nilearn's Destrieux atlas fetched into DATA_DIR (or fetched there).
It prints the transform, a 4×4 matrix of millimetres to millimetres, and
how much of the atlas's cortex is in the grey matter before and after.
"""
import sys

import nibabel as nib
import numpy as np
from nilearn import datasets
from scipy import ndimage, optimize

# The centre the scales are about: the middle of the MNI brain, in mm.
CENTRE = np.array([0.0, -18.0, 18.0])


def transform(params) -> np.ndarray:
    """The 4×4 matrix of the scales and shifts *params* (sx, sy, sz, tx,
    ty, tz), about CENTRE: atlas millimetres to MNI152 millimetres."""
    scale, shift = np.asarray(params[:3]), np.asarray(params[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(scale)
    matrix[:3, 3] = CENTRE + shift - scale * CENTRE
    return matrix


def regions_on(grid, labels, affine) -> np.ndarray:
    """The voxels of *grid* (an image) in a region of *labels* on
    *affine*, as a float mask."""
    to_labels = np.linalg.inv(affine) @ grid.affine
    ijk = np.indices(grid.shape[:3]).reshape(3, -1)
    points = to_labels[:3, :3] @ ijk + to_labels[:3, 3:4]
    inside = ndimage.map_coordinates(
        (labels > 0).astype(np.float32), points, order=0, cval=0.0
    )
    return inside.reshape(grid.shape[:3])


def score(params, grid, grey, labels, affine) -> float:
    mask = regions_on(grid, labels, transform(params) @ affine)
    return float((mask * (grey - 0.5)).sum())


def main(data_dir):
    atlas = datasets.fetch_atlas_destrieux_2009(
        data_dir=data_dir, lateralized=True, verbose=0
    )
    image = nib.load(atlas["maps"])
    labels = np.asanyarray(image.dataobj)
    grid = datasets.load_mni152_gm_template(resolution=2)
    grey = np.asanyarray(grid.dataobj).astype(float)
    found = optimize.minimize(
        lambda p: -score(p, grid, grey, labels, image.affine),
        x0=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], method="Powell",
        options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 20000},
    )
    matrix = transform(found.x)

    def in_grey(affine):
        mask = regions_on(grid, labels, affine)
        return (mask * grey).sum() / mask.sum()

    print("scales", np.round(found.x[:3], 4), "shifts (mm)",
          np.round(found.x[3:], 2))
    print("mean grey-matter probability of the cortex of the atlas:",
          round(in_grey(image.affine), 4), "->",
          round(in_grey(matrix @ image.affine), 4))
    print(np.array2string(np.round(matrix, 4), separator=", "))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "neurolang_data")
