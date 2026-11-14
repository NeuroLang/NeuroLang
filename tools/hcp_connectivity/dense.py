"""
A dense whole-brain tractogram of HCP subjects in the space of the atlases
(FSL MNI152, that of the HCP's ``MNINonLinear`` images), of at most
``--megabytes`` MB, and the FreeSurfer segmentation of its subjects in that
space, so that tract_querier's queries run on it.

The streamlines are those of the subjects' tractograms (``TRK_DIR/<subject>
.trk``, e.g. Disconnectome7T's), mapped by ``--affine`` (a 4 x 4 text file,
fit-space's) when the tractograms are not quite in that space; those of at
least ``--minimum-length`` mm are kept, compressed (dipy's
``compress_streamlines``: the fewest points within ``--tolerance`` mm of
the streamline), pooled, shuffled, and as many as fit in the size are
written in ``OUT/<name>.tck`` (TCK: 12 bytes a point and one more a
streamline), at most ``--count`` of them; with ``--brain``, only those
whose points are all within the MNI152 brain mask of the viewer
(nilearn's). The segmentation, ``OUT/<name>_wmparc.nii.gz``, is the label
of each voxel the most of the subjects' ``MNINonLinear/wmparc.nii.gz``
give it (their grid, 0.7 mm; FreeSurfer's labels, so FreeSurfer's colour
LUT and the term mapping freesurfer_aseg_aparc apply).

    python -m hcp_connectivity dense TRK_DIR HCP_DIR OUT [--subjects 20]
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def streamlines(path: Path, affine: np.ndarray, minimum_length: float,
                tolerance: float) -> list:
    """The streamlines of the tractogram *path* of at least *minimum_length*
    mm, mapped by *affine* and compressed within *tolerance* mm."""
    from dipy.tracking.streamlinespeed import compress_streamlines, length

    lines = nib.streamlines.load(str(path)).streamlines
    kept = lines[np.flatnonzero(length(lines) >= minimum_length)]
    kept = [nib.affines.apply_affine(affine, s).astype(np.float32)
            for s in kept]
    return compress_streamlines(kept, tol_error=tolerance)


def segmentation(wmparcs: list) -> nib.Nifti1Image:
    """The label most of the segmentations *wmparcs* give each voxel."""
    first = nib.load(str(wmparcs[0]))
    labels = np.stack([np.asarray(nib.load(str(p)).dataobj, dtype=np.int16)
                       for p in wmparcs])
    values = np.unique(labels)
    counts = np.zeros(labels.shape[1:], dtype=np.int16)
    best = np.zeros(labels.shape[1:], dtype=np.int16)
    for value in values:
        count = (labels == value).sum(axis=0, dtype=np.int16)
        better = count > counts
        best[better], counts[better] = value, count[better]
    image = nib.Nifti1Image(best, first.affine)
    image.set_data_dtype(np.int16)
    return image


def within_brain(lines: list) -> np.ndarray:
    """Whether each streamline of *lines* has all its points within the
    MNI152 brain mask of the engines' viewer (nilearn's, 1 mm): the 7 T
    data are distorted over the orbits and under the cerebellum, where
    some streamlines leave their own subject's brain."""
    from nilearn.datasets import load_mni152_brain_mask

    mask = load_mni152_brain_mask(resolution=1)
    inside = np.asarray(mask.dataobj) > 0
    data = nib.streamlines.ArraySequence(lines)
    ijk = np.round(nib.affines.apply_affine(np.linalg.inv(mask.affine),
                                            data._data)).astype(int)
    within = np.all((ijk >= 0) & (ijk < inside.shape), axis=1)
    within[within] = inside[tuple(ijk[within].T)]
    return np.logical_and.reduceat(within, data._offsets)


def run(trk_dir: Path, hcp: Path, out: Path, name: str = "disconnectome7t",
        subjects: int = 20, affine: Path = None, minimum_length: float = 30,
        tolerance: float = 0.5, megabytes: float = 480,
        seed: int = 0, count: int = None, brain: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.loadtxt(affine) if affine else np.eye(4)
    paths = sorted(p for p in trk_dir.glob("*.trk")
                   if (hcp / p.stem / "MNINonLinear" / "wmparc.nii.gz")
                   .exists())[:subjects]
    pool = []
    for path in paths:
        lines = streamlines(path, matrix, minimum_length, tolerance)
        pool.extend(lines)
        print(f"{path.stem}: {len(lines)} streamlines, "
              f"{sum(map(len, lines))} points", flush=True)
    order = np.random.default_rng(seed).permutation(len(pool))
    if brain:
        inside = within_brain(pool)
        print(f"{inside.sum()} of {len(pool)} streamlines within the brain",
              flush=True)
        order = order[inside[order]]
    order = order[:count]
    size = 12 * (1 + np.cumsum([len(pool[i]) + 1 for i in order]))
    chosen = [pool[i] for i in order[:np.searchsorted(size, megabytes * 1e6)]]
    tractogram = nib.streamlines.Tractogram(chosen,
                                            affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(out / f"{name}.tck"))
    print(f"{len(chosen)} of {len(pool)} streamlines of {len(paths)} "
          f"subjects, {sum(map(len, chosen))} points, "
          f"{(out / f'{name}.tck').stat().st_size / 1e6:.0f} MB", flush=True)
    nib.save(segmentation([hcp / p.stem / "MNINonLinear" / "wmparc.nii.gz"
                           for p in paths]),
             str(out / f"{name}_wmparc.nii.gz"))
    print("subjects: " + " ".join(p.stem for p in paths), flush=True)
