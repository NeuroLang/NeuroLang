"""
Check that a tractogram is in the MNI space of the atlases (FSL MNI152,
that of the HCP's ``MNINonLinear`` images): its streamlines against the
white matter of its subject's own FreeSurfer segmentation in that space
(``HCP_DIR/<subject>/MNINonLinear/wmparc.nii.gz``).

The density of the streamlines' points on a grid of ``--voxel`` mm is
compared with the subject's white matter there (wmparc's labels of white
matter): the fraction of the points in it, and the correlation of the
density with it, as they are, with the left and right swapped (x -> -x,
which a tractogram of the wrong orientation would prefer), and shifted by
up to ``--shift`` mm along each axis (a tractogram in another space would
prefer a shift). The tractogram is in the space of the segmentation when
the unshifted, unswapped one is the best by a margin.

    python -m hcp_connectivity check-space TRACTOGRAM HCP_DIR [--subject S]
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def points(path: Path, most: int = 200_000, seed: int = 0) -> np.ndarray:
    """The points (millimetres) of at most *most* streamlines of the
    tractogram *path*, drawn at random."""
    streamlines = nib.streamlines.load(str(path)).streamlines
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(streamlines), min(most, len(streamlines)),
                        replace=False)
    return np.concatenate([streamlines[i] for i in chosen])


def white_matter(wmparc: Path):
    """The white matter mask of a FreeSurfer wmparc (its cerebral white
    matter, 2 and 41, and its white matter parcels, 3000 to 5002), and its
    affine."""
    image = nib.load(str(wmparc))
    labels = np.asarray(image.dataobj, dtype=int)
    mask = np.isin(labels, (2, 41)) | ((labels >= 3000) & (labels <= 5002))
    return mask, image.affine


def density(xyz: np.ndarray, mask: np.ndarray, affine: np.ndarray):
    """The number of points of *xyz* in each voxel of the grid of *mask*."""
    ijk = np.round(nib.affines.apply_affine(np.linalg.inv(affine), xyz)
                   ).astype(int)
    inside = np.all((ijk >= 0) & (ijk < mask.shape), axis=1)
    counts = np.zeros(mask.shape)
    np.add.at(counts, tuple(ijk[inside].T), 1)
    return counts


def scores(xyz, mask, affine):
    """The fraction of the points in the white matter, and the correlation
    of their density with it."""
    counts = density(xyz, mask, affine)
    inside = counts[mask].sum() / max(counts.sum(), 1)
    return inside, np.corrcoef(counts.ravel(), mask.ravel())[0, 1]


def run(tractogram: Path, hcp: Path, subject: str = None,
        shift: int = 3) -> dict:
    subject = subject or tractogram.name.split(".")[0]
    mask, affine = white_matter(hcp / subject / "MNINonLinear" /
                                "wmparc.nii.gz")
    xyz = points(tractogram)
    result = {"subject": subject, "points": len(xyz)}
    result["inside"], result["r"] = scores(xyz, mask, affine)
    swapped = xyz * np.array([-1, 1, 1])
    result["inside_swapped"], result["r_swapped"] = scores(
        swapped, mask, affine)
    best = (0, 0, 0)
    best_r = result["r"]
    for dx in range(-shift, shift + 1):
        for dy in range(-shift, shift + 1):
            for dz in range(-shift, shift + 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                _, r = scores(xyz + np.array([dx, dy, dz]), mask, affine)
                if r > best_r:
                    best, best_r = (dx, dy, dz), r
    result["best_shift_mm"], result["r_best_shift"] = best, best_r
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)
    return result


def coarse_white_matter(wmparc: Path, factor: int = 2):
    """The fraction of white matter of the blocks of *factor* voxels a
    side of *wmparc*'s grid, and the affine of the grid of the blocks."""
    mask, affine = white_matter(wmparc)
    shape = [n // factor * factor for n in mask.shape]
    mask = mask[:shape[0], :shape[1], :shape[2]].reshape(
        shape[0] // factor, factor, shape[1] // factor, factor,
        shape[2] // factor, factor).mean(axis=(1, 3, 5))
    coarse = affine @ np.diag([factor, factor, factor, 1.0])
    coarse[:3, 3] = affine[:3, 3] + affine[:3, :3] @ ([(factor - 1) / 2] * 3)
    return mask, coarse


def splat(xyz: np.ndarray, shape, affine: np.ndarray) -> np.ndarray:
    """The density of the points *xyz* on the grid of *shape* and *affine*,
    each point shared by the 8 voxels around it by trilinear weights (a
    density that moves continuously with the points)."""
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    base = np.floor(ijk).astype(int)
    frac = ijk - base
    density = np.zeros(shape)
    for corner in np.ndindex(2, 2, 2):
        at = base + corner
        weight = np.prod(np.where(corner, frac, 1 - frac), axis=1)
        inside = np.all((at >= 0) & (at < shape), axis=1)
        np.add.at(density, tuple(at[inside].T), weight[inside])
    return density


def correlation(xyz, mask, affine):
    """The correlation of the density of the points *xyz* with the white
    matter *mask* (fractions): highest when the points fill the white
    matter evenly and leave the rest empty, so that pulling them inwards,
    which leaves white matter empty, does not pay."""
    return np.corrcoef(splat(xyz, mask.shape, affine).ravel(),
                       mask.ravel())[0, 1]


def to_matrix(parameters: np.ndarray) -> np.ndarray:
    """The 4 x 4 affine of 12 parameters: the 3 x 3 linear part as the
    identity plus the first 9, the translation the last 3."""
    matrix = np.eye(4)
    matrix[:3, :3] += parameters[:9].reshape(3, 3)
    matrix[:3, 3] = parameters[9:]
    return matrix


def fit(tractograms: list, hcp: Path, most: int = 50_000) -> np.ndarray:
    """The affine that maps the points of the *tractograms* (of HCP
    subjects, named by them) onto the white matter of their subjects'
    MNINonLinear segmentations best: the one maximising the mean
    correlation of their density with the white matter (on blocks of 2
    voxels a side, 1.4 mm)."""
    from scipy.optimize import minimize

    data = []
    for path in tractograms:
        subject = Path(path).name.split(".")[0]
        mask, affine = coarse_white_matter(
            hcp / subject / "MNINonLinear" / "wmparc.nii.gz")
        xyz = points(Path(path), most)
        chosen = np.random.default_rng(0).choice(
            len(xyz), min(1_000_000, len(xyz)), replace=False)
        data.append((xyz[chosen], mask, affine))

    def loss(parameters):
        matrix = to_matrix(parameters)
        return -np.mean([correlation(nib.affines.apply_affine(matrix, xyz),
                                     mask, affine)
                         for xyz, mask, affine in data])

    result = minimize(loss, np.zeros(12), method="Powell",
                      options={"xtol": 1e-3, "ftol": 1e-6, "maxfev": 4000})
    print(f"fit: r {-loss(np.zeros(12)):.4f} -> {-result.fun:.4f}",
          flush=True)
    return to_matrix(result.x)
