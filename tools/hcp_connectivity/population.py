"""
A stop-gap structural connectivity that includes the subcortex, from one
population-average tractogram: the streamlines of a tractogram in MNI
space counted between the regions of Schaefer 2018 (400 parcels, 17
networks: 1 to 400) and Tian's subcortex (scale 2, 3T: 401 to 432), both
volume atlases of the FSL MNI152 2 mm grid.

The connectomes of :mod:`structural` end on the cortical surface, so they
have no subcortex. A streamline here is assigned the regions of its two
ends: the label of the labelled voxel nearest to each end, within
``--radius`` millimetres (an end often stops in the white matter, short of
the grey matter of a cortical parcel); a streamline with an end farther
from every region is left out. The count matrix is symmetric, the
streamlines between two regions at ``[a, b]`` and ``[b, a]``, those with
both ends in a region on its diagonal. These are the streamlines of the
tractogram, not a mean over subjects, and not of the scale of the
structural connectivity of the HCP connectomes.

``python -m hcp_connectivity population OUT TRACTOGRAM...`` (``.trk``,
``.trk.gz`` or ``.tck``, or a folder of them) writes
``OUT/parquet/schaefer400_17_tian_population.parquet`` (the matrix) and the
engine tables ``OUT/tsv/schaefer400_17_tian_population.tsv.gz``
(``region, tian_region, streamlines``) and ``OUT/tsv/tian_population.tsv.gz``
(``region1, region2, streamlines``).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .atlases import TIAN, fetch, tian_names
from .cifti import engine_name
from .tables import matrix, pairs, write_matrix, write_tsv

N_CORTEX_PARCELS = 400
RADIUS = 3.0  # mm


def label_volume(folder: Path):
    """The label volume of the 2 mm MNI grid (Schaefer 1 to 400, Tian 401
    to 432, 0 for none), its affine, the names of the regions in this
    order and the number of voxels in both atlases (Schaefer's label wins
    there)."""
    from nilearn import datasets

    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_CORTEX_PARCELS, yeo_networks=17, resolution_mm=2,
        data_dir=str(folder))
    cortex = nib.load(schaefer.maps)
    subcortex = nib.load(fetch(f"{TIAN}.nii.gz", folder))
    assert cortex.shape == subcortex.shape
    assert np.allclose(cortex.affine, subcortex.affine), "not the same grid"
    labels = np.asarray(cortex.get_fdata(), dtype=int)
    tian = np.asarray(subcortex.get_fdata(), dtype=int)
    names = [engine_name(n) for n in schaefer.labels
             if engine_name(n) != "Background"]
    assert len(names) == N_CORTEX_PARCELS and labels.max() == len(names)
    sub = tian_names(folder)
    assert tian.max() == len(sub)
    overlap = int(((labels > 0) & (tian > 0)).sum())
    labels[(labels == 0) & (tian > 0)] = tian[(labels == 0) & (tian > 0)] \
        + N_CORTEX_PARCELS
    return labels, cortex.affine, names + sub, overlap


def endpoints(path: Path) -> np.ndarray:
    """The two ends, in MNI millimetres, of each streamline of the
    tractogram *path*: an array (streamlines, 2, 3)."""
    streamlines = nib.streamlines.load(str(path)).streamlines
    return np.array([(s[0], s[-1]) for s in streamlines]).reshape(-1, 2, 3)


def assign(points: np.ndarray, labels: np.ndarray, affine: np.ndarray,
           radius: float = RADIUS) -> np.ndarray:
    """The region of each of *points* (millimetres, any shape ending in 3):
    the label of the nearest labelled voxel of *labels* within *radius*
    millimetres of it; 0 when there is none."""
    voxels = np.argwhere(labels > 0)
    centres = nib.affines.apply_affine(affine, voxels)
    distance, nearest = cKDTree(centres).query(
        points.reshape(-1, 3), distance_upper_bound=radius)
    values = labels[tuple(voxels.T)]
    found = np.isfinite(distance)
    out = np.zeros(len(distance), dtype=int)
    out[found] = values[nearest[found]]
    return out.reshape(points.shape[:-1])


def count(regions: np.ndarray, n: int) -> np.ndarray:
    """The symmetric (n, n) matrix of the streamlines between the regions
    (1 to n) of their two ends, *regions* (streamlines, 2); those with an
    end in no region (0) are not counted."""
    a, b = (regions[regions.min(axis=1) > 0] - 1).T
    both = np.bincount(a * n + b, minlength=n * n).reshape(n, n)
    return both + both.T - np.diag(np.diag(both))


def tractograms(paths) -> list:
    """The tractogram files of *paths*: files, or folders searched."""
    files = []
    for path in map(Path, paths):
        files += sorted(f for f in (path.rglob("*") if path.is_dir()
                                    else [path])
                        if f.name.endswith((".trk", ".trk.gz", ".tck")))
    return files


def run(work: Path, out: Path, paths, radius: float = RADIUS) -> None:
    work.mkdir(parents=True, exist_ok=True)
    labels, affine, names, overlap = label_volume(work)
    n = len(names)
    files = tractograms(paths)
    total = np.zeros((n, n), dtype=np.int64)
    streamlines = assigned = 0
    for file in files:
        regions = assign(endpoints(file), labels, affine, radius)
        total += count(regions, n)
        streamlines += len(regions)
        assigned += int((regions.min(axis=1) > 0).sum())
        print(f"{file.name}: {len(regions)} streamlines", flush=True)
    for folder in ("parquet", "tsv"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    frame = matrix(total, names)
    cortex, subcortex = names[:N_CORTEX_PARCELS], names[N_CORTEX_PARCELS:]
    write_matrix(frame, out / "parquet" / "schaefer400_17_tian_population"
                 ".parquet")
    write_tsv(pairs(frame, cortex, subcortex, ["region", "tian_region"],
                    "streamlines", 0),
              out / "tsv" / "schaefer400_17_tian_population.tsv.gz")
    write_tsv(pairs(frame, subcortex, subcortex, ["region1", "region2"],
                    "streamlines", 0),
              out / "tsv" / "tian_population.tsv.gz")
    print(f"{len(files)} tractograms, {streamlines} streamlines; both ends "
          f"within {radius} mm of a region: {assigned} "
          f"({assigned / max(streamlines, 1):.1%}); {overlap} voxels in "
          "both atlases (Schaefer's kept)")
