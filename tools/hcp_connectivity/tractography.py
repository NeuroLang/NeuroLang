"""
The structural connectivity of an HCP subject by its own tractography,
over Schaefer's 400 cortical parcels (17 networks) and the 32 regions of
Tian's subcortex (scale 2): a task of the worker does its share of the
subjects, and sums their matrices in its part (parcels.Part).

For each subject (MRtrix3 on the HCP's preprocessed diffusion data, in the
T1w space of the subject, and its FreeSurfer segmentation):

1. ``dwi2response dhollander`` and ``dwi2fod msmt_csd`` (white matter and
   CSF; the grey matter response is estimated, its fibre distribution not
   made);
2. ``5ttgen freesurfer`` on ``aparc+aseg`` (hippocampi and amygdalae as
   sub-cortical grey matter), for the anatomically constrained
   tractography (ACT);
3. ``tckgen`` (iFOD2, ACT, backtracking, cropped at the grey/white
   interface, dynamic seeding) until *streamlines* are selected, and
   ``tcksift2`` for the weight of each;
4. the atlas in the space of the subject: the cortical parcels 1-400 and
   Tian's regions 401-432, in MNI space (:func:`combine_labels`), brought
   to the subject through the HCP's warp (:func:`warp_labels`, nearest
   neighbour), and checked against the subject's segmentation
   (:func:`dice`);
5. ``tck2connectome`` (symmetric, zero diagonal, the end points searched
   within 2 mm of a parcel): the SIFT2 weights summed between two parcels,
   times the SIFT2 proportionality coefficient (so that the subjects are
   comparable), and the number of streamlines.

The matrix, the checks and the time of each step of a subject are kept
(``OUT/matrices``, ``OUT/qc``); the subject's intermediate files, tractogram
included, are deleted. The commands of MRtrix3 are run by the command of
the environment variable ``MRTRIX_CMD`` (for a container: ``singularity
exec mrtrix3.sif``; empty: those on the PATH).
"""

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .atlases import cortical, fetch, tian_names
from .parcels import Part, done

SCHAEFER = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI/"
    "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz")
ATLAS = "schaefer400_17"
NAME = f"{ATLAS}_tian"  # the atlas of the matrices of the parts
N_CORTEX = 400
N_TIAN = 32
FILES = ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz")


# -- the atlas

def resample_nearest(labels: np.ndarray, affine: np.ndarray, shape,
                     target: np.ndarray) -> np.ndarray:
    """The image *labels* (voxels to world by *affine*) at the nearest
    voxel, on the grid of *shape* whose voxels go to world by *target*;
    0 outside."""
    grid = np.indices(shape).reshape(3, -1)
    world = target[:3, :3] @ grid + target[:3, 3:]
    voxel = np.rint(np.linalg.inv(affine)[:3, :3] @ world +
                    np.linalg.inv(affine)[:3, 3:]).astype(int)
    inside = ((voxel >= 0) & (voxel < np.array(labels.shape)[:, None])
              ).all(axis=0)
    out = np.zeros(grid.shape[1], dtype=labels.dtype)
    out[inside] = labels[tuple(voxel[:, inside])]
    return out.reshape(shape)


def combine_labels(cortex: np.ndarray, subcortex: np.ndarray) -> np.ndarray:
    """The labels of the cortical parcels (1 to 400) and of Tian's regions
    (1 to 32, numbered 401 to 432), on the same grid; Tian's where both."""
    combined = cortex.astype(np.uint16)
    inside = subcortex > 0
    combined[inside] = subcortex[inside].astype(np.uint16) + N_CORTEX
    return combined


def volume_atlas(folder: Path) -> Path:
    """The atlas in MNI space (``FOLDER/schaefer400_17_tian.nii.gz``, on
    Schaefer's 1 mm grid), made once; needs the internet the first time
    (Schaefer's volume, Tian's atlas)."""
    path = folder / f"{NAME}.nii.gz"
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    schaefer = nib.load(fetch(SCHAEFER, folder))
    from .atlases import TIAN
    subcortex = nib.load(fetch(f"{TIAN}.nii.gz", folder))
    fine = resample_nearest(np.asarray(subcortex.dataobj, dtype=int),
                            subcortex.affine, schaefer.shape,
                            schaefer.affine)
    combined = combine_labels(np.asarray(schaefer.dataobj, dtype=int), fine)
    assert len(np.unique(combined)) == N_CORTEX + N_TIAN + 1
    part = path.with_name(f"{path.stem}.part.nii.gz")
    nib.save(nib.Nifti1Image(combined, schaefer.affine), part)
    part.rename(path)
    return path


def warp_labels(labels: np.ndarray, affine: np.ndarray, warp: np.ndarray,
                warp_affine: np.ndarray, shape, target: np.ndarray,
                chunk: int = 20) -> np.ndarray:
    """The *labels* of an image in MNI space (*affine*) in the space of the
    subject, on the grid of *shape* and *target*, by nearest neighbour.

    *warp* is the HCP's ``standard2acpc_dc``, in the FSL format: on a grid
    of the subject's space (*warp_affine*), the displacement in mm, along
    the voxel axes, to the point of MNI space; it is interpolated linearly
    at each voxel of the target."""
    to_voxel = np.linalg.inv(warp_affine)
    zooms = np.linalg.norm(warp_affine[:3, :3], axis=0)
    # FSL's displacements are along the voxel axes (x flipped when the
    # image has a positive determinant), in mm
    axes = warp_affine[:3, :3] / zooms
    if np.linalg.det(warp_affine[:3, :3]) > 0:
        axes = axes * np.array([-1.0, 1.0, 1.0])
    atlas_voxel = np.linalg.inv(affine)
    out = np.zeros(shape, dtype=labels.dtype)
    for start in range(0, shape[0], chunk):
        grid = np.indices((min(chunk, shape[0] - start), *shape[1:]))
        grid[0] += start
        grid = grid.reshape(3, -1)
        world = target[:3, :3] @ grid + target[:3, 3:]
        at = to_voxel[:3, :3] @ world + to_voxel[:3, 3:]
        shift = np.stack([map_coordinates(warp[..., d], at, order=1,
                                          mode="nearest")
                          for d in range(3)])
        mni = world + axes @ shift
        voxel = np.rint(atlas_voxel[:3, :3] @ mni +
                        atlas_voxel[:3, 3:]).astype(int)
        inside = ((voxel >= 0) & (voxel < np.array(labels.shape)[:, None])
                  ).all(axis=0)
        values = np.zeros(grid.shape[1], dtype=labels.dtype)
        values[inside] = labels[tuple(voxel[:, inside])]
        out[start:start + chunk] = values.reshape(-1, *shape[1:])
    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """The Dice coefficient of the masks *a* and *b*."""
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else float("nan")


# FreeSurfer's labels of aseg, left and right
SEGMENTATION = {
    "thalamus": ((10,), (49,)),
    "caudate": ((11,), (50,)),
    "putamen": ((12,), (51,)),
    "pallidum": ((13,), (52,)),
    "hippocampus": ((17,), (53,)),
    "amygdala": ((18,), (54,)),
    "accumbens": ((26,), (58,)),
}
TIAN_STRUCTURE = {"thalamus": ("THA",), "caudate": ("CAU",),
                  "putamen": ("PUT",), "pallidum": ("GP",),
                  "hippocampus": ("HIP",), "amygdala": ("AMY",),
                  "accumbens": ("NAc",)}


def tian_region_numbers(names: list, structure: str, side: str) -> list:
    """The labels (401 to 432) of the regions of Tian's atlas of the
    *structure* in the hemisphere *side* ('l' or 'r'), from their *names*
    (``aHIP rh``: the structure in capitals after at most one letter)."""
    found = []
    for i, name in enumerate(names[N_CORTEX:]):
        region, _, hemi = name.rpartition(" ")
        stem = region.split(" ")[0]
        if hemi == f"{side}h" and any(
                stem == s or stem[1:] == s for s in
                TIAN_STRUCTURE[structure]):
            found.append(N_CORTEX + i + 1)
    return found


def atlas_check(labels: np.ndarray, segmentation: np.ndarray,
                names: list) -> dict:
    """The Dice coefficients of the structures of Tian's atlas in the space
    of the subject (*labels*) and of the subject's segmentation, by
    structure and hemisphere, and of the cortical parcels with the cortex
    of the segmentation (labels 1000 to 2999, and 3 and 42)."""
    out = {}
    for structure, (left, right) in SEGMENTATION.items():
        for side, codes in (("l", left), ("r", right)):
            regions = tian_region_numbers(names, structure, side)
            out[f"dice_{structure}_{side}"] = dice(
                np.isin(labels, regions), np.isin(segmentation, codes))
    cortex = (((segmentation >= 1000) & (segmentation < 3000)) |
              np.isin(segmentation, (3, 42)))
    out["dice_cortex"] = dice((labels > 0) & (labels <= N_CORTEX), cortex)
    out["cortex_in_labels"] = float(
        ((labels > 0) & (labels <= N_CORTEX) & cortex).sum() / cortex.sum())
    return out


# -- the matrices

def read_matrix(path: Path, n: int = N_CORTEX + N_TIAN) -> np.ndarray:
    """The connectome of tck2connectome (a CSV), checked to be square."""
    matrix = pd.read_csv(path, header=None, sep=r"[,\s]+",
                         engine="python").values.astype(float)
    assert matrix.shape == (n, n), matrix.shape
    return matrix


def assigned_fraction(path: Path, weights: np.ndarray = None) -> dict:
    """From the assignments of tck2connectome (``-out_assignments``: the
    two nodes of each streamline, 0 for none), the fraction of the
    streamlines with both ends assigned (to two different nodes: *inter*),
    and of the weights *weights* if given."""
    nodes = pd.read_csv(path, header=None, sep=r"\s+").values
    both = (nodes > 0).all(axis=1)
    different = both & (nodes[:, 0] != nodes[:, 1])
    out = {"streamlines": len(nodes),
           "assigned": float(both.mean()),
           "assigned_distinct": float(different.mean())}
    if weights is not None:
        out["assigned_weight"] = float(weights[both].sum() / weights.sum())
        out["assigned_distinct_weight"] = float(
            weights[different].sum() / weights.sum())
    return out


def labelled(names: list, *parts: str) -> np.ndarray:
    """The indices of the regions whose name has one of the *parts*."""
    return np.array([i for i, n in enumerate(names)
                     if any(p in n for p in parts)], dtype=int)


def matrix_check(matrix: np.ndarray, names: list) -> dict:
    """What the matrix of streamlines should show: the share of the
    connections of Tian's thalamus that go to the cortex; of the striatum
    (caudate, putamen, accumbens) that go to the frontal cortex
    (PFC, OFC, FEF, Fr...); of the hippocampus that go to the temporal
    (Temp, Aud); each next to the share of the cortical parcels they make
    (enrichment: their ratio); the share of the cortical connections that
    cross the midline, and of those of Tian's regions that cross it."""
    cortex = np.arange(N_CORTEX)
    left = np.array([i for i in cortex if " LH " in names[i]])
    right = np.array([i for i in cortex if " RH " in names[i]])
    frontal = labelled(names[:N_CORTEX], "PFC", "OFC", "FEF", " Fr")
    temporal = labelled(names[:N_CORTEX], "Temp", " Aud")

    def regions(structure):
        return np.array([N_CORTEX + i for i, n in enumerate(
            names[N_CORTEX:]) if any(
                n.split(" ")[0] in (s, f"a{s}", f"p{s}", f"l{s}", f"m{s}")
                for s in TIAN_STRUCTURE[structure])], dtype=int)

    def share(rows, columns):
        total = matrix[rows].sum()
        return float(matrix[np.ix_(rows, columns)].sum() / total
                     ) if total else float("nan")

    striatum = np.concatenate([regions(s) for s in
                               ("caudate", "putamen", "accumbens")])
    out = {"thalamus_to_cortex": share(regions("thalamus"), cortex),
           "striatum_to_frontal": share(striatum, frontal),
           "striatum_to_frontal_enrichment": share(striatum, frontal) /
           (len(frontal) / N_CORTEX),
           "hippocampus_to_temporal": share(regions("hippocampus"),
                                            temporal),
           "hippocampus_to_temporal_enrichment":
           share(regions("hippocampus"), temporal) /
           (len(temporal) / N_CORTEX),
           "cortex_interhemispheric": float(
               (matrix[np.ix_(left, right)].sum() * 2) /
               matrix[np.ix_(cortex, cortex)].sum()),
           "tian_to_anything": float(matrix[N_CORTEX:].sum() /
                                     matrix.sum())}
    # the thalamus keeps to its own side
    for side, own in (("l", left), ("r", right)):
        out[f"thalamus_{side}_ipsilateral"] = share(
            tian_side(names, "thalamus", side), own) / max(
                share(tian_side(names, "thalamus", side), cortex), 1e-12)
    return out


def tian_side(names: list, structure: str, side: str) -> np.ndarray:
    return np.array(tian_region_numbers(names, structure, side)) - 1


# -- the commands

def mrtrix(*command, log: Path = None, threads: int = 1) -> None:
    """Run an MRtrix3 command through ``MRTRIX_CMD``."""
    args = shlex.split(os.environ.get("MRTRIX_CMD", "")) + [
        str(c) for c in command]
    args += ["-nthreads", str(threads)]
    result = subprocess.run(args, capture_output=True, text=True)
    if log is not None:
        with open(log, "a") as handle:
            handle.write(f"$ {shlex.join(args)}\n{result.stdout}"
                         f"{result.stderr}\n")
    if result.returncode:
        raise RuntimeError(f"{shlex.join(args)}: {result.stderr[-2000:]}")


def subjects_with_diffusion(hcp: Path, limit: int = None) -> list:
    """The subjects of *hcp* with the preprocessed diffusion data, its
    segmentation and the warp, sorted; the first *limit*."""
    def complete(path):
        return (all((path / "T1w" / "Diffusion" / f).exists()
                    for f in FILES) and
                (path / "T1w" / "aparc+aseg.nii.gz").exists() and
                (path / "MNINonLinear" / "xfms" /
                 "standard2acpc_dc.nii.gz").exists())
    found = sorted(p.name for p in hcp.iterdir()
                   if p.is_dir() and complete(p))
    return found[:limit]


def timed(steps: dict, name: str, function) -> None:
    """Do *function*, and keep its time in *steps*."""
    start = time.time()
    function()
    steps[name] = round(time.time() - start, 1)


def subject_matrix(subject: str, hcp: Path, work: Path, scratch: Path,
                   streamlines: int, threads: int):
    """The matrices of *subject* (SIFT2 weights times mu, and counts), and
    its checks and timings; the intermediate files stay in
    ``SCRATCH/<subject>``."""
    home = hcp / subject
    diffusion = home / "T1w" / "Diffusion"
    folder = scratch / subject
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "log.txt"
    steps, t = {}, lambda name: folder / name

    def step(name, function):
        """Run *function* unless it was done before (its marker)."""
        marker = t(f"{name}.done")
        if not marker.exists():
            timed(steps, name, function)
            marker.write_text("")

    def run(*command):
        mrtrix(*command, log=log, threads=threads)

    step("convert", lambda: (
        run("mrconvert", diffusion / "data.nii.gz", t("dwi.mif"),
            "-fslgrad", diffusion / "bvecs", diffusion / "bvals",
            "-force"),
        run("mrconvert", diffusion / "nodif_brain_mask.nii.gz",
            t("mask.mif"), "-force")))
    step("response", lambda: run(
        "dwi2response", "dhollander", t("dwi.mif"), t("wm.txt"),
        t("gm.txt"), t("csf.txt"), "-mask", t("mask.mif"), "-force"))
    step("fod", lambda: run(
        "dwi2fod", "msmt_csd", t("dwi.mif"), t("wm.txt"), t("wmfod.mif"),
        t("csf.txt"), t("csffod.mif"), "-mask", t("mask.mif"), "-force"))
    step("5tt", lambda: run(
        "5ttgen", "freesurfer", home / "T1w" / "aparc+aseg.nii.gz",
        t("5tt.mif"), "-sgm_amyg_hipp", "-nocrop", "-force"))
    step("tckgen", lambda: run(
        "tckgen", t("wmfod.mif"), t("tracks.tck"), "-act", t("5tt.mif"),
        "-backtrack", "-crop_at_gmwmi", "-seed_dynamic", t("wmfod.mif"),
        "-select", streamlines, "-force"))
    step("sift2", lambda: run(
        "tcksift2", t("tracks.tck"), t("wmfod.mif"), t("weights.csv"),
        "-act", t("5tt.mif"), "-out_mu", t("mu.txt"), "-force"))

    names = cortical(work)[ATLAS][1] + tian_names(work / "atlases")
    mni = nib.load(volume_atlas(work / "atlases"))
    aseg = nib.load(home / "T1w" / "aparc+aseg.nii.gz")

    def atlas():
        warp = nib.load(home / "MNINonLinear" / "xfms" /
                        "standard2acpc_dc.nii.gz")
        labels = warp_labels(np.asarray(mni.dataobj), mni.affine,
                             np.asarray(warp.dataobj, dtype=np.float32),
                             warp.affine, aseg.shape, aseg.affine)
        nib.save(nib.Nifti1Image(labels, aseg.affine), t("labels.nii.gz"))
    step("atlas", atlas)

    def connectome():
        for weighted, name in ((True, "weighted"), (False, "count")):
            run("tck2connectome", t("tracks.tck"), t("labels.nii.gz"),
                t(f"{name}.csv"), "-symmetric", "-zero_diagonal",
                "-assignment_radial_search", 2,
                *(["-tck_weights_in", t("weights.csv"),
                   "-out_assignments", t("assignments.txt")]
                  if weighted else []), "-force")
    step("connectome", connectome)

    mu = float(t("mu.txt").read_text())
    weights = np.loadtxt(t("weights.csv"))
    counts = read_matrix(t("count.csv"))
    weighted = read_matrix(t("weighted.csv")) * mu
    labels = np.asarray(nib.load(t("labels.nii.gz")).dataobj)
    qc = {"subject": subject, "mu": mu, "streamlines_requested": streamlines,
          "nonzero_weights": int((weights > 0).sum()),
          **assigned_fraction(t("assignments.txt"), weights),
          **atlas_check(labels, np.asarray(aseg.dataobj), names),
          **matrix_check(weighted, names), "seconds": steps}
    return {NAME: weighted, f"{NAME}_count": counts}, qc


def run(work: Path, hcp: Path, out: Path, task: int, tasks: int,
        streamlines: int = 5_000_000, limit: int = None,
        scratch: Path = None, keep: bool = False) -> None:
    """The matrices of the share of the subjects of the task, into the
    part ``OUT/part-<task>.npz`` (``OUT/matrices/<subject>.npz``, the
    checks in ``OUT/qc/<subject>.json``)."""
    scratch = scratch or work / "tractography"
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK",
                                 os.cpu_count() or 1))
    for folder in ("matrices", "qc"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    part = Part(out / f"part-{task}.npz", every=1)
    finished = done(out)
    for subject in subjects_with_diffusion(hcp, limit)[task::tasks]:
        if subject in finished:
            continue
        start = time.time()
        try:
            matrices, qc = subject_matrix(subject, hcp, work, scratch,
                                          streamlines, threads)
        except Exception as error:  # a broken subject: told, left out
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            continue
        qc["seconds"]["total"] = round(time.time() - start, 1)
        np.savez_compressed(out / "matrices" / f"{subject}.npz", **matrices)
        (out / "qc" / f"{subject}.json").write_text(
            json.dumps(qc, indent=1))
        part.add(subject, matrices, total=float(matrices[NAME].sum()),
                 assigned=qc["assigned"], mu=qc["mu"])
        if not keep:
            shutil.rmtree(scratch / subject, ignore_errors=True)
        print(f"{subject}: {qc['seconds']['total']:.0f} s, "
              f"{qc['assigned']:.3f} assigned", flush=True)
    part.save()
