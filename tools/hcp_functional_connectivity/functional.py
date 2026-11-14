"""
The resting-state functional connectivity of the HCP subjects, by atlas:
a task of a Slurm array, which sums, over its share of the subjects, the
Fisher z of the correlations of the mean time series of the regions.

For each subject with the four runs of rest (rfMRI_REST{1,2}_{LR,RL},
ICA-FIX cleaned, MSMAll, 91,282 grayordinates), each run's time series
are averaged within each region: the cortical parcels of each atlas of
``WORK/atlases.npz`` (hcp_structural_connectivity/atlases.py, on the
59,412 cortical vertices of the grayordinates), then the 32 regions of
Tian's subcortex atlas, scale 2 (its 3T image, on the grid of the
subcortical voxels of the grayordinates). Their Pearson correlations,
Fisher-z transformed, are averaged over the four runs: the subject's
connectivity. The task writes, in OUT, ``part-<task>.npz``: for each
atlas the sum of the subjects' connectivity and of its square, and the
subjects it summed (combine.py makes the group's).

    python functional.py WORK HCP_DIR OUT --task I --tasks N
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.sparse as sp
from nibabel import analyze

RUNS = ("rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR",
        "rfMRI_REST2_RL")
N_CORTEX = 59412
TIAN = "Tian_Subcortex_S2_3T"

# Some of the HCP's CIFTI files have voxel sizes of 0 in their NIfTI-2
# header, which nibabel refuses; they mean nothing to a CIFTI file, read
# here as 1 (the files are left as they are).
_set_zooms = analyze.AnalyzeHeader.set_zooms
analyze.AnalyzeHeader.set_zooms = lambda self, zooms: _set_zooms(
    self, tuple(z if z > 0 else 1.0 for z in zooms))


def run_path(hcp: Path, subject: str, run: str) -> Path:
    return (hcp / subject / "MNINonLinear" / "Results" / run /
            f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii")


def subjects_with_all_runs(hcp: Path) -> list:
    return sorted(s.name for s in hcp.iterdir() if s.is_dir() and all(
        run_path(hcp, s.name, run).exists() for run in RUNS))


def averaging(parcels: np.ndarray, n: int) -> sp.csr_matrix:
    """The matrix that averages the rows of each of the *n* parcels
    (``parcels``: the parcel of each row, from 1; 0 for none)."""
    inside = np.flatnonzero(parcels)
    counts = np.bincount(parcels[inside] - 1, minlength=n)
    return sp.csr_matrix(
        (1.0 / counts[parcels[inside] - 1], (inside, parcels[inside] - 1)),
        shape=(len(parcels), n))


def tian_averaging(work: Path, axis) -> sp.csr_matrix:
    """The averaging of the subcortical grayordinates of *axis* (a
    BrainModelAxis) by the regions of Tian's atlas."""
    image = nib.load(work / "atlases" / f"{TIAN}.nii.gz")
    assert np.allclose(image.affine, axis.affine), "not the grid of the data"
    labels = np.asarray(image.get_fdata(), dtype=int)
    parcels = np.zeros(axis.size - N_CORTEX, dtype=int)
    for name, rows, model in axis.iter_structures():
        if "CORTEX" not in name:
            start, stop, _ = rows.indices(axis.size)
            i, j, k = model.voxel.T
            parcels[start - N_CORTEX:stop - N_CORTEX] = labels[i, j, k]
    assert (np.bincount(parcels, minlength=labels.max() + 1)[1:] > 0).all()
    return averaging(parcels, labels.max())


def connectivity(series: np.ndarray) -> np.ndarray:
    """The Fisher z of the correlations of the columns of *series*, its
    diagonal 0."""
    r = np.corrcoef(series, rowvar=False)
    np.fill_diagonal(r, 0)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("work", type=Path)
    parser.add_argument("hcp", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    atlases = np.load(args.work / "atlases.npz")
    names = [a for a in atlases.files if not a.endswith(":names")]
    cortex = {a: averaging(atlases[a].astype(int),
                           len(atlases[f"{a}:names"])) for a in names}
    subjects = subjects_with_all_runs(args.hcp)[args.task::args.tasks]
    sums = {a: 0.0 for a in names}
    squares = {a: 0.0 for a in names}
    tian = None
    done = []
    for subject in subjects:
        start = time.time()
        z = {a: 0.0 for a in names}
        try:
            for run in RUNS:
                image = nib.load(run_path(args.hcp, subject, run))
                if tian is None:
                    tian = tian_averaging(args.work,
                                          image.header.get_axis(1))
                data = np.asarray(image.get_fdata(dtype=np.float32))
                sub = (tian.T @ data[:, N_CORTEX:].T).T
                for a in names:
                    regions = (cortex[a].T @ data[:, :N_CORTEX].T).T
                    z[a] = z[a] + connectivity(np.hstack([regions, sub]))
        except Exception as error:  # a broken run: the subject left out
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            continue
        for a in names:
            mean = z[a] / len(RUNS)
            sums[a] = sums[a] + mean
            squares[a] = squares[a] + mean ** 2
        done.append(subject)
        print(f"{subject}: {time.time() - start:.0f} s", flush=True)
    np.savez(args.out / f"part-{args.task}.npz", subjects=np.array(done),
             **{f"{a}:sum": sums[a] for a in names},
             **{f"{a}:squares": squares[a] for a in names})


if __name__ == "__main__":
    main()
