"""
The resting-state functional connectivity of the HCP subjects, by atlas:
a task of the worker sums its share of the subjects.

For each subject with the four runs of rest (rfMRI_REST{1,2}_{LR,RL},
ICA-FIX cleaned, MSMAll, 91,282 grayordinates), each run's time series
are averaged within each region: the cortical parcels of each atlas of
``WORK/atlases.npz``, then the 32 regions of Tian's subcortex atlas
(atlases.tian). Their Pearson correlations, Fisher-z transformed, are
averaged over the four runs, the subject's connectivity, summed into the
task's part (parcels.Part); a region without grayordinates (two of
Schaefer's 1000 parcels lie under the medial wall) has NaN.
"""

import time
from pathlib import Path

import numpy as np

from .atlases import cortical, tian
from .cifti import N_CORTEX, load
from .parcels import Part, done, indicator

RUNS = ("rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR",
        "rfMRI_REST2_RL")


def run_path(hcp: Path, subject: str, run: str) -> Path:
    return (hcp / subject / "MNINonLinear" / "Results" / run /
            f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii")


def subjects_with_all_runs(hcp: Path) -> list:
    return sorted(s.name for s in hcp.iterdir() if s.is_dir() and all(
        run_path(hcp, s.name, run).exists() for run in RUNS))


def connectivity(series: np.ndarray) -> np.ndarray:
    """The Fisher z of the correlations of the columns of *series*, its
    diagonal 0; NaN those of a constant column (a region without
    grayordinates)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(series, rowvar=False)
    np.fill_diagonal(r, 0)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def run(work: Path, hcp: Path, out: Path, task: int, tasks: int) -> None:
    atlases = {a: indicator(parcels, len(names), mean=True)
               for a, (parcels, names) in cortical(work).items()}
    out.mkdir(parents=True, exist_ok=True)
    part = Part(out / f"part-{task}.npz")
    finished = done(out)
    subcortex = None
    for subject in subjects_with_all_runs(hcp)[task::tasks]:
        if subject in finished:
            continue
        start = time.time()
        z = {a: 0.0 for a in atlases}
        try:
            for name in RUNS:
                image = load(run_path(hcp, subject, name))
                if subcortex is None:
                    parcels, names = tian(work / "atlases",
                                          image.header.get_axis(1))
                    subcortex = indicator(parcels, len(names), mean=True)
                data = np.asarray(image.get_fdata(dtype=np.float32))
                deep = (subcortex.T @ data[:, N_CORTEX:].T).T
                for a, P in atlases.items():
                    regions = (P.T @ data[:, :N_CORTEX].T).T
                    z[a] = z[a] + connectivity(np.hstack([regions, deep]))
        except Exception as error:  # a broken run: the subject left out
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            continue
        part.add(subject, {a: m / len(RUNS) for a, m in z.items()})
        print(f"{subject}: {time.time() - start:.0f} s", flush=True)
    part.save()
