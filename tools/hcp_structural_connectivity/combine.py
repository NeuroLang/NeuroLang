"""
The group structural connectivity of the parcellated HCP connectomes
(parcellate.py): for each atlas, the mean over the subjects of the number
of streamlines between two parcels.

Writes, in OUT_DIR:

- ``hcp_structural_connectivity.npz``: for each atlas ``a``, ``a`` (the
  mean matrix, its diagonal the streamlines within a parcel, counted
  twice), ``a:sd`` (their standard deviation across subjects),
  ``a:names`` and ``subjects``;
- ``<a>.tsv.gz`` for each atlas: ``region1  region2  streamlines``, the
  mean between two distinct parcels, both ways round, the pairs no
  streamline joins left out; the regions are named as the parcels of the
  NeuroLang engines;
- ``validation.tsv``: for each subject the authors also parcellated with
  Glasser's atlas, the correlation of the streamlines between distinct
  parcels of theirs and of ours, and the fraction of the streamlines on
  which they differ.

    python combine.py WORK OUT GLASSER_DIR OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
        i, j = np.nonzero(mean)
        kept = i != j
        pd.DataFrame({
            "region1": labels[i[kept]], "region2": labels[j[kept]],
            "streamlines": np.round(mean[i[kept], j[kept]], 3),
        }).to_csv(target / f"{a}.tsv.gz", sep="\t", index=False)
    np.savez_compressed(target / "hcp_structural_connectivity.npz", **arrays)
    validation = pd.DataFrame(rows)
    validation.to_csv(target / "validation.tsv", sep="\t", index=False)
    print(f"{n} subjects; streamlines per subject (counted twice): "
          f"{np.min(totals):.0f} to {np.max(totals):.0f}")
    if len(validation):
        print(f"Glasser, {len(validation)} subjects: r median "
              f"{validation.r.median():.4f} (min {validation.r.min():.4f}), "
              f"differ median {validation.differ.median():.3f}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
