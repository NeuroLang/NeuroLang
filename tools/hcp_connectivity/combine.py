"""
The group connectivity of the parts of the workers (structural.py,
functional.py), its tables and its gradients, in DATASET.

The structural connectivity, by atlas ``a``:

- ``parquet/<a>_mean.parquet`` and ``parquet/<a>_sd.parquet``: the mean
  over the subjects of the streamlines between two parcels, and their
  standard deviation, square matrices whose columns and index
  (``region``) are the parcels (the diagonal: the streamlines within a
  parcel, counted twice); ``parquet/subjects.parquet``: the subjects and
  their streamlines (counted twice);
- ``tsv/<a>.tsv.gz``: ``region1  region2  streamlines``, between two
  distinct parcels, both ways round, to the thousandth (the pairs under it
  left out);
- ``validation.tsv``: for each subject the authors also parcellated with
  Glasser's atlas, the correlation of the streamlines between distinct
  parcels of theirs and of ours (``r``), and the fraction on which they
  differ (``differ``).

The functional connectivity, over the cortical parcels of ``a`` then the
32 regions of Tian's subcortex (Tian's names: ``aHIP rh``):

- ``parquet/<a>_functional_mean.parquet`` (r, the tanh of the mean of the
  subjects' Fisher z) and ``parquet/<a>_functional_z_sd.parquet``;
  ``parquet/functional_subjects.parquet``;
- ``tsv/<a>_functional.tsv.gz`` (``region1  region2  r``, two parcels),
  ``tsv/<a>_tian_functional.tsv.gz`` (``region  tian_region  r``),
  ``tsv/tian_functional.tsv.gz`` (two of Tian's regions), to 6 decimals.

And the linear gradients of the cortex of each (gradients.py):
``parquet/<a>[_functional]_gradients.parquet``, ``...
_gradient_variance.parquet``, and their ``tsv/...tsv.gz``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .atlases import cortical
from .gradients import gradients
from .parcels import group
from .tables import matrix, pairs, write_matrix, write_tsv

DIMENSIONS = 10


def write_gradients(dataset: Path, name: str, profiles: pd.DataFrame,
                    transform) -> pd.DataFrame:
    """Write the gradients of the connectivity *profiles* (a matrix of the
    cortical parcels) as ``<name>_gradients`` and
    ``<name>_gradient_variance``; return the variance."""
    scores, variance, _ = gradients(profiles, DIMENSIONS, transform)
    scores.to_parquet(dataset / "parquet" / f"{name}_gradients.parquet")
    variance.to_parquet(
        dataset / "parquet" / f"{name}_gradient_variance.parquet",
        index=False)
    long = scores.reset_index().melt(
        id_vars="region", var_name="dimension", value_name="value")
    long["dimension"] = long["dimension"].astype(int)
    write_tsv(long.dropna(), dataset / "tsv" / f"{name}_gradients.tsv.gz",
              float_format="%.6g")
    write_tsv(variance, dataset / "tsv" / f"{name}_gradient_variance.tsv.gz",
              float_format="%.6g")
    return variance


def structural(work: Path, parts: Path, dataset: Path) -> None:
    mean, sd, subjects, extra = group(parts)
    for folder in ("parquet", "tsv"):
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    for a, (_, regions) in cortical(work).items():
        write_matrix(matrix(mean[a], regions),
                     dataset / "parquet" / f"{a}_mean.parquet")
        write_matrix(matrix(sd[a], regions),
                     dataset / "parquet" / f"{a}_sd.parquet")
        write_tsv(pairs(matrix(mean[a], regions), regions, regions,
                        ["region1", "region2"], "streamlines", 3),
                  dataset / "tsv" / f"{a}.tsv.gz")
        write_gradients(dataset, a, matrix(mean[a], regions), np.log1p)
    pd.DataFrame({"subject": subjects, "streamlines": extra["total"]}
                 ).to_parquet(dataset / "parquet" / "subjects.parquet",
                              index=False)
    validation = pd.DataFrame({"subject": subjects,
                               "r": extra["glasser_r"],
                               "differ": extra["glasser_differ"]}).dropna()
    validation.to_csv(dataset / "validation.tsv", sep="\t", index=False)
    print(f"{len(subjects)} subjects; Glasser, {len(validation)} subjects: "
          f"r median {validation.r.median():.4f} "
          f"(min {validation.r.min():.4f})")


def functional(work: Path, tian_names: list, parts: Path,
               dataset: Path) -> None:
    mean_z, sd_z, subjects, _ = group(parts)
    for folder in ("parquet", "tsv"):
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": subjects}).to_parquet(
        dataset / "parquet" / "functional_subjects.parquet", index=False)
    for a, (_, regions) in cortical(work).items():
        names = regions + tian_names
        values = np.tanh(mean_z[a])
        np.fill_diagonal(values, 1.0)
        r = matrix(values, names)
        write_matrix(r, dataset / "parquet" / f"{a}_functional_mean.parquet")
        write_matrix(matrix(sd_z[a], names),
                     dataset / "parquet" / f"{a}_functional_z_sd.parquet")
        write_tsv(pairs(r, regions, regions, ["region1", "region2"], "r", 6),
                  dataset / "tsv" / f"{a}_functional.tsv.gz")
        write_tsv(pairs(r, regions, tian_names, ["region", "tian_region"],
                        "r", 6),
                  dataset / "tsv" / f"{a}_tian_functional.tsv.gz")
        if a == "schaefer100_17":  # Tian's own, the same for every atlas
            write_tsv(pairs(r, tian_names, tian_names,
                            ["region1", "region2"], "r", 6),
                      dataset / "tsv" / "tian_functional.tsv.gz")
        z = matrix(mean_z[a], names).loc[regions, regions]
        variance = write_gradients(dataset, f"{a}_functional", z,
                                   lambda values: values)
        upper = np.triu_indices(len(regions), 1)
        print(f"{a}: mean r "
              f"{np.nanmean(r.loc[regions, regions].values[upper]):.3f}; "
              "gradients explain " + ", ".join(
                  f"{x:.3f}" for x in variance[variance.hemisphere == "L"]
                  .explained[:3]), flush=True)
    print(f"{len(subjects)} subjects")


def regradient(work: Path, dataset: Path) -> None:
    """The gradients again, of the matrices of DATASET."""
    for a in cortical(work):
        path = dataset / "parquet" / f"{a}_mean.parquet"
        if path.exists():
            write_gradients(dataset, a, pd.read_parquet(path), np.log1p)
        path = dataset / "parquet" / f"{a}_functional_mean.parquet"
        if path.exists():
            r = pd.read_parquet(path)
            regions = [n for n in r.index if n in cortical(work)[a][1]]
            write_gradients(dataset, f"{a}_functional",
                            np.arctanh(r.loc[regions, regions].clip(
                                -0.999999, 0.999999)), lambda values: values)
