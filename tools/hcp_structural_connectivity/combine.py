"""
The group structural connectivity of the parcellated HCP connectomes
(parcellate.py): for each atlas, the mean over the subjects of the number
of streamlines between two parcels.

Writes, in OUT_DIR:

- ``<a>_mean.parquet`` and ``<a>_sd.parquet`` for each atlas ``a``: the
  mean over the subjects of the streamlines between two parcels, and their
  standard deviation, square matrices whose columns and index (``region``)
  are the parcels, named as those of the NeuroLang engines; the diagonal
  is the streamlines within a parcel, counted twice;
- ``subjects.parquet``: the subjects and their streamlines (counted twice);
- ``<a>.tsv.gz`` for each atlas: ``region1  region2  streamlines``, the
  mean between two distinct parcels, both ways round, to the thousandth,
  the pairs no streamline joins (under a thousandth) left out;
- ``validation.tsv``: for each subject the authors also parcellated with
  Glasser's atlas, the correlation of the streamlines between distinct
  parcels of theirs and of ours, and the fraction of the streamlines on
  which they differ.

    python combine.py WORK OUT GLASSER_DIR OUT_DIR

``python combine.py --from-npz NPZ OUT_DIR`` writes the parquet files of
the ``hcp_structural_connectivity.npz`` an earlier version wrote.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    """Write *matrix* in the parquet file *path*, its 64-bit values whole:
    zstd on their bytes split by significance (byte_stream_split), which
    compresses averages, whose last bits are noise, better than pandas'
    default (snappy, dictionaries): 20% smaller, as small as numpy's
    compressed npz."""
    matrix.to_parquet(path, compression="zstd", use_dictionary=False,
                      use_byte_stream_split=list(matrix.columns))


def write_parquet(arrays, atlases: list, target: Path) -> None:
    """Write the mean and the standard deviation matrices of each atlas,
    and the subjects, in parquet files of *target*."""
    for a in atlases:
        regions = pd.Index([str(x) for x in arrays[f"{a}:names"]],
                           name="region")
        for statistic, key in (("mean", a), ("sd", f"{a}:sd")):
            write_matrix(pd.DataFrame(arrays[key], index=regions,
                                      columns=list(regions)),
                         target / f"{a}_{statistic}.parquet")
    pd.DataFrame({"subject": [str(s) for s in arrays["subjects"]],
                  "streamlines": arrays["totals"]}).to_parquet(
        target / "subjects.parquet", index=False)


def from_npz(npz: Path, target: Path) -> None:
    arrays = np.load(npz)
    atlases = [k for k in arrays.files
               if ":" not in k and k not in ("subjects", "totals")]
    write_parquet(arrays, atlases, target)


def main(work: Path, out: Path, glasser_dir: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    atlases = np.load(work / "atlases.npz")
    names = [n for n in atlases.files if not n.endswith(":names")]
    files = sorted(out.glob("[0-9]*.npz"))
    subjects = [f.stem for f in files if not f.stem.endswith(".part")]
    sums = {a: 0.0 for a in names}
    squares = {a: 0.0 for a in names}
    totals, rows = [], []
    for subject in subjects:
        result = np.load(out / f"{subject}.npz")
        totals.append(float(result["total"]))
        for a in names:
            sums[a] = sums[a] + result[a]
            squares[a] = squares[a] + result[a] ** 2
        theirs = glasser_dir / \
            f"{subject}_test_structural_connectivity_glasser.npy"
        if theirs.exists():
            g, m = np.load(theirs), result["glasser"]
            upper = np.triu_indices_from(g, 1)
            rows.append({
                "subject": subject,
                "r": np.corrcoef(m[upper], g[upper])[0, 1],
                "differ": np.abs(m - g)[upper].sum() / g[upper].sum(),
            })
    n = len(subjects)
    arrays = {"subjects": np.array(subjects), "totals": np.array(totals)}
    for a in names:
        mean = sums[a] / n
        arrays[a] = mean
        arrays[f"{a}:sd"] = np.sqrt(np.maximum(squares[a] / n - mean ** 2,
                                               0))
        arrays[f"{a}:names"] = atlases[f"{a}:names"]
        labels = atlases[f"{a}:names"]
        # to the thousandth of a streamline: the smoothing of the
        # connectomes leaves less to pairs no streamline joins
        i, j = np.nonzero(np.round(mean, 3))
        kept = i != j
        pd.DataFrame({
            "region1": labels[i[kept]], "region2": labels[j[kept]],
            "streamlines": np.round(mean[i[kept], j[kept]], 3),
        }).to_csv(target / f"{a}.tsv.gz", sep="\t", index=False)
    write_parquet(arrays, names, target)
    validation = pd.DataFrame(rows)
    validation.to_csv(target / "validation.tsv", sep="\t", index=False)
    print(f"{n} subjects; streamlines per subject (counted twice): "
          f"{np.min(totals):.0f} to {np.max(totals):.0f}")
    if len(validation):
        print(f"Glasser, {len(validation)} subjects: r median "
              f"{validation.r.median():.4f} (min {validation.r.min():.4f}), "
              f"differ median {validation.differ.median():.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "--from-npz":
        from_npz(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(*map(Path, sys.argv[1:5]))
