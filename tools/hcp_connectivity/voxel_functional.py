"""
The resting-state functional connectivity of the cortical voxels of the
HCP subjects with the parcels of Schaefer's atlas (400 parcels, 17
networks), as partial correlations: a task of the worker sums its share
of the subjects.

For each subject with the four runs of rest (rfMRI_REST{1,2}_{LR,RL},
ICA-FIX cleaned, volumetric: ``MNINonLinear/Results/<run>/<run>_hp2000_
clean.nii.gz``, on the 2 mm FSL MNI152 grid of the atlas's volume) and its
age, sex and family (``HCP_DIR/hcp_restricted.csv``, ``hcp_behavioral.
csv``), the partial correlation of each voxel of the atlas with each
parcel, controlling for the other parcels (:func:`partial_correlations`;
a voxel's own parcel is the mean of its voxels without it), its Fisher z,
averaged over the four runs (a run with a parcel without signal is left
out, a subject with no such run too); NaN for a voxel without signal in a
run.

The group (:class:`GroupSums`): for each voxel the z of the subjects with
signal in it, regressed on their age and sex, each family counting once
(a subject weighs 1 / the subjects of its family), read at the mean age
and an equal mix of the sexes. A task keeps the sums of that regression's
normal equations, in float32 where they are matrices.
"""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

import pandas as pd

from .cifti import engine_name
from .functional import RUNS
from .parcels import indicator

KEY = "schaefer400_17_voxel_partial"
N_PARCELS = 400
CHUNK = 8192  # voxels at a time
MIN_SUBJECTS = 4  # to regress on the age and the sex
AGE_ORIGIN = 30.0  # the ages are centred on it in the sums


def run_path(hcp: Path, subject: str, run: str) -> Path:
    return (hcp / subject / "MNINonLinear" / "Results" / run /
            f"{run}_hp2000_clean.nii.gz")


def subjects_with_all_runs(hcp: Path) -> list:
    return sorted(s.name for s in hcp.iterdir() if s.is_dir() and all(
        run_path(hcp, s.name, run).exists() for run in RUNS))


def schaefer_volume(folder: Path):
    """The label volume of Schaefer's atlas (400 parcels, 17 networks) on
    the 2 mm FSL MNI152 grid (1 to 400, 0 for none), its affine and the
    names of the parcels, in the engines' form (needs the internet)."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=17, resolution_mm=2,
        data_dir=str(folder))
    image = nib.load(atlas.maps)
    names = [engine_name(n) for n in atlas.labels
             if engine_name(n) != "Background"]
    labels = np.asarray(image.get_fdata(), dtype=int)
    assert len(names) == N_PARCELS and labels.max() == N_PARCELS
    return labels, image.affine, names


def centres(labels: np.ndarray, affine: np.ndarray):
    """The voxels of the label volume *labels* that have a parcel, in the
    order of the array (``labels > 0`` as a mask), and their centres in
    millimetres: arrays (voxels, 3)."""
    voxels = np.argwhere(labels > 0)
    return voxels, apply_affine(affine, voxels)


def _means(series: np.ndarray, parcels: np.ndarray, n: int) -> np.ndarray:
    """The mean time series of each parcel: an array (time, n)."""
    return np.asarray(indicator(parcels, n, mean=True).T @ series.T).T


def ledoit_wolf_alpha(series: np.ndarray, parcels: np.ndarray,
                      n: int) -> float:
    """The Ledoit-Wolf shrinkage coefficient of the correlations of the
    parcels' mean time series (*series*: time by voxels; *parcels*: the
    parcel of each voxel, from 1)."""
    from sklearn.covariance import ledoit_wolf_shrinkage

    means = _means(series, parcels, n)
    return float(ledoit_wolf_shrinkage(
        (means - means.mean(0)) / means.std(0)))


def partial_correlations(series: np.ndarray, parcels: np.ndarray, n: int,
                         alpha: float, chunk: int = CHUNK) -> np.ndarray:
    """The partial correlation of each voxel with each parcel, controlling
    for the other parcels: an array (voxels, n).

    A parcel's signal is the mean of its voxels; a voxel's own parcel's,
    the mean of the others (the voxel left out). The correlations of the
    voxel and the parcels are shrunk to the identity by *alpha*
    (``(1 - alpha) R + alpha I``), and the partial correlations are those
    of the inverse. All the voxels share the inverse of the parcels' own
    shrunk covariance, ``Theta``; leaving a voxel out of its parcel
    changes that parcel's row and column of it, a change of rank 2 whose
    inverse comes by Woodbury's identity, so that no matrix is inverted
    per voxel. (Scaling variables does not change partial correlations,
    so the shrinkage is on the covariance: ``(1 - alpha) S + alpha diag(S)``.)

    *series*: time by voxels; *parcels*: the parcel of each voxel, from 1.
    """
    parcels = np.asarray(parcels)
    p = parcels - 1
    counts = np.bincount(p, minlength=n)
    if (counts < 2).any():
        raise ValueError("a parcel with fewer than two voxels")
    T = series.shape[0]
    X = series - series.mean(axis=0)
    means = _means(X, parcels, n)
    G = means.T @ means / T
    keep = 1 - alpha
    Theta = np.linalg.inv(keep * G + alpha * np.diag(np.diag(G)))
    KT = (Theta @ G).T
    theta_diagonal = np.diag(Theta)
    out = np.empty((series.shape[1], n))
    for start in range(0, series.shape[1], chunk):
        rows = slice(start, start + chunk)
        x = X[:, rows].astype(np.float64)
        pc = p[rows]
        m = len(pc)
        at = np.arange(m)
        B = x.T @ means / T
        q = (x * x).sum(axis=0) / T
        c = (counts[pc] / (counts[pc] - 1.0))[:, None]
        d = (1.0 / (counts[pc] - 1.0))[:, None]
        Bp = B[at, pc][:, None]
        Gp = G[pc]
        Gpp = G[pc, pc][:, None]
        q = q[:, None]
        TB = B @ Theta
        tp = Theta[pc]
        # the change of row p of the covariance of the parcels, r (its
        # p-th entry half the change of the variance), and Theta r
        r = keep * ((c - 1) * Gp - d * B)
        r_formula = r[at, pc][:, None]
        r_p = (c ** 2 * Gpp - 2 * c * d * Bp + d ** 2 * q - Gpp) / 2
        r[at, pc] = r_p[:, 0]
        tr = keep * ((c - 1) * KT[pc] - d * TB) + tp * (r_p - r_formula)
        # the voxel's covariances with the parcels, and Theta times them
        b = keep * B
        b[at, pc] = (keep * (c * Bp - d * q))[:, 0]
        Tb = keep * (TB + tp * ((c - 1) * Bp - d * q))
        # Woodbury: (I + V' Theta U)^-1, U = [e_p, r], V = [r, e_p]
        a1 = tr[at, pc][:, None]
        a2 = (r * tr).sum(axis=1)[:, None]
        a3 = Theta[pc, pc][:, None]
        with np.errstate(all="ignore"):
            det = (1 + a1) ** 2 - a2 * a3
            w11 = w22 = (1 + a1) / det
            w12, w21 = -a2 / det, -a3 / det
            h1 = (r * Tb).sum(axis=1)[:, None]
            h2 = Tb[at, pc][:, None]
            Tb = Tb - tp * (w11 * h1 + w12 * h2) - tr * (w21 * h1 + w22 * h2)
            diagonal = theta_diagonal - (
                w11 * tp * tr + w12 * tp ** 2 + w21 * tr ** 2
                + w22 * tr * tp)
            s = q - (b * Tb).sum(axis=1)[:, None]
            out[rows] = Tb / np.sqrt(s * diagonal + Tb ** 2)
    return out


def connectivity(series: np.ndarray, parcels: np.ndarray,
                 n: int) -> np.ndarray:
    """The Fisher z of the partial correlations of the voxels of *series*
    with the *n* parcels, shrunk by Ledoit-Wolf's coefficient of the
    parcels' means; float32; NaN for a constant voxel."""
    alpha = ledoit_wolf_alpha(series, parcels, n)
    r = partial_correlations(series, parcels, n, alpha)
    return np.arctanh(np.clip(r, -0.999999, 0.999999)).astype(np.float32)


def load_phenotypes(hcp: Path) -> pd.DataFrame:
    """The age (years), sex (``male``, 1 or 0) and family of the subjects
    of both ``hcp_restricted.csv`` (Family_ID, Age_in_Yrs) and
    ``hcp_behavioral.csv`` (Gender) of the folder *hcp*, by subject (a
    string)."""
    restricted = pd.read_csv(hcp / "hcp_restricted.csv",
                             dtype={"Subject": str, "Family_ID": str},
                             usecols=["Subject", "Family_ID", "Age_in_Yrs"])
    behavioral = pd.read_csv(hcp / "hcp_behavioral.csv",
                             dtype={"Subject": str},
                             usecols=["Subject", "Gender"])
    both = restricted.merge(behavioral, on="Subject").set_index("Subject")
    return pd.DataFrame({
        "age": both["Age_in_Yrs"].astype(float),
        "male": (both["Gender"] == "M").astype(float),
        "family": both["Family_ID"]}).sort_index()


def family_weights(family: pd.Series) -> pd.Series:
    """The weight of each subject, 1 / the subjects of its *family*, so
    that each family counts once (same index)."""
    return 1.0 / family.map(family.value_counts())


def eligible(hcp: Path) -> list:
    """The subjects with the four runs and an age, a sex and a family."""
    known = set(load_phenotypes(hcp).index)
    return [s for s in subjects_with_all_runs(hcp) if s in known]


class GroupSums:
    """The sums of the subjects' z for the group's regression on age and
    sex, voxel by voxel, over the subjects with signal in it.

    The weighted normal equations of z on 1, the age (centred) and the sex
    of each voxel: its six sums of w (1, a, m, a a, a m, m m), and three
    matrices (voxels, parcels) of the sums of w z, w a z and w m z, in
    float32, over the subjects with signal (``n_valid``). Subjects, their
    weights, ages, sexes and families are kept: the mean age and the
    weights are those of all of them.
    """

    MATRICES = 3

    def __init__(self, voxels: int, parcels: int):
        self.moments = np.zeros((6, voxels))
        self.n_valid = np.zeros(voxels, dtype=np.int64)
        self.sums = np.zeros((self.MATRICES, voxels, parcels),
                             dtype=np.float32)
        self.subjects, self.families = [], []
        self.weights, self.ages, self.male = [], [], []

    def add(self, subject: str, z: np.ndarray, weight: float, age: float,
            male: float, family: str = "") -> None:
        """Add the z of *subject* (voxels, parcels; NaN for a voxel
        without signal)."""
        valid = ~np.isnan(z).any(axis=1)
        a = age - AGE_ORIGIN
        factors = weight * np.array(
            [1.0, a, male, a * a, a * male, male * male])
        self.moments[:, valid] += factors[:, None]
        self.n_valid += valid
        z = np.where(valid[:, None], z, 0).astype(np.float32)
        for i, covariate in enumerate((1.0, a, male)):
            self.sums[i] += np.float32(weight * covariate) * z
        self.subjects.append(subject)
        self.families.append(family)
        self.weights.append(float(weight))
        self.ages.append(float(age))
        self.male.append(float(male))

    def merge(self, other: "GroupSums") -> None:
        self.moments += other.moments
        self.n_valid += other.n_valid
        self.sums += other.sums
        for name in ("subjects", "families", "weights", "ages", "male"):
            getattr(self, name).extend(getattr(other, name))

    def adjusted(self) -> np.ndarray:
        """The correlations (voxels, parcels), float32: the tanh of the
        regression of z on the age and the sex, read at the weighted mean
        age and an equal mix of the sexes; NaN for a voxel with signal in
        fewer than MIN_SUBJECTS subjects, or whose subjects do not vary in
        age and sex."""
        m = self.moments
        A = np.stack([np.stack([m[0], m[1], m[2]], axis=-1),
                      np.stack([m[1], m[3], m[4]], axis=-1),
                      np.stack([m[2], m[4], m[5]], axis=-1)], axis=1)
        eye = np.eye(3)
        ok = self.n_valid >= MIN_SUBJECTS
        A[~ok] = eye
        ok &= np.linalg.cond(A) < 1e8
        A[~ok] = eye
        reference = np.average(self.ages, weights=self.weights) - AGE_ORIGIN
        beta = np.linalg.solve(A, self.sums.transpose(1, 0, 2).astype(
            np.float64))
        z = beta[:, 0] + reference * beta[:, 1] + 0.5 * beta[:, 2]
        out = np.tanh(z).astype(np.float32)
        out[~ok] = np.nan
        return out

    def save(self, path: Path) -> None:
        """Write the sums as *path* (``.npz``), through a partial file, and
        the subjects done beside it (``.subjects``, for a shell to read)."""
        path = Path(path)
        partial = path.with_name(path.stem + ".partial.npz")
        np.savez(partial, subjects=np.array(self.subjects),
                 families=np.array(self.families),
                 weights=np.array(self.weights), ages=np.array(self.ages),
                 male=np.array(self.male), n_valid=self.n_valid,
                 **{f"moment:{i}": x for i, x in enumerate(self.moments)},
                 **{f"sum:{i}": x for i, x in enumerate(self.sums)})
        partial.rename(path)
        path.with_suffix(".subjects").write_text(
            "".join(f"{s}\n" for s in self.subjects))

    @classmethod
    def load(cls, path: Path) -> "GroupSums":
        stored = np.load(path)
        sums = [stored[f"sum:{i}"] for i in range(cls.MATRICES)]
        result = cls(*sums[0].shape)
        result.sums = np.stack(sums)
        result.moments = np.stack([stored[f"moment:{i}"] for i in range(6)])
        result.n_valid = stored["n_valid"].astype(np.int64)
        result.subjects = [str(s) for s in stored["subjects"]]
        result.families = [str(s) for s in stored["families"]]
        result.weights = [float(x) for x in stored["weights"]]
        result.ages = [float(x) for x in stored["ages"]]
        result.male = [float(x) for x in stored["male"]]
        return result

    @classmethod
    def of(cls, folder: Path) -> "GroupSums":
        """The sums of all the parts of *folder*, one at a time in memory."""
        found = sorted(p for p in Path(folder).glob("part-*.npz")
                       if not p.name.endswith(".partial.npz"))
        total = None
        for path in found:
            part = cls.load(path)
            if total is None:
                total = part
            else:
                total.merge(part)
        if total is None:
            raise FileNotFoundError(f"no part-*.npz in {folder}")
        return total


def finished(folder: Path) -> set:
    """The subjects of the parts of *folder*."""
    return {line for path in Path(folder).glob("part-*.subjects")
            for line in path.read_text().split()}


def run(work: Path, hcp: Path, out: Path, task: int, tasks: int) -> None:
    labels, affine, names = schaefer_volume(work / "atlases")
    mask = labels > 0
    parcels = labels[mask]
    everyone = eligible(hcp)
    phenotypes = load_phenotypes(hcp).loc[everyone]
    # fixed by all the eligible subjects, so that the tasks are independent
    weights = family_weights(phenotypes["family"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"part-{task}.npz"
    sums = (GroupSums.load(path) if path.exists()
            else GroupSums(int(mask.sum()), len(names)))
    done = finished(out)
    added = 0
    for subject in everyone[task::tasks]:
        if subject in done:
            continue
        start = time.time()
        z, used = 0.0, 0
        try:
            for name in RUNS:
                image = nib.load(str(run_path(hcp, subject, name)))
                assert image.shape[:3] == labels.shape and np.allclose(
                    image.affine, affine), "not the grid of the atlas"
                series = image.get_fdata(dtype=np.float32)[mask].T
                valid = series.std(axis=0) > 0  # the others: no signal
                if (np.bincount(parcels[valid], minlength=len(names) + 1)[1:]
                        < 2).any():  # a parcel outside the run's brain mask
                    print(f"{subject}: {name} left out: a parcel without "
                          "signal", flush=True)
                    continue
                r = np.full((series.shape[1], len(names)), np.nan,
                            dtype=np.float32)
                r[valid] = connectivity(series[:, valid], parcels[valid],
                                        len(names))
                z, used = z + r, used + 1
        except Exception as error:  # a broken run: the subject left out
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            continue
        if not used:
            print(f"{subject}: left out, no usable run", flush=True)
            continue
        phenotype = phenotypes.loc[subject]
        sums.add(subject, z / used, weights[subject], phenotype["age"],
                 phenotype["male"], phenotype["family"])
        added += 1
        sums.save(path)  # a task killed loses at most the subject at work
        print(f"{subject}: {time.time() - start:.0f} s", flush=True)
