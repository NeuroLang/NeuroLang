"""
The group resting-state functional connectivity of the HCP subjects, from
the parts of functional.py: for each atlas, over its cortical parcels and
the 32 regions of Tian's subcortex (scale 2), the correlation of the
mean of the subjects' Fisher z (r = tanh of the mean z), the standard
deviation of the subjects' z, and the linear gradients of the cortex (as
those of the structural connectivity, hcp_structural_connectivity/
gradients.py, of the z of the parcels within each hemisphere).

Writes, in DATASET:

- ``parquet/<a>_functional_mean.parquet`` (r) and
  ``parquet/<a>_functional_z_sd.parquet``: square matrices whose columns
  and index (``region``) are the parcels then Tian's regions, named as the
  NeuroLang engines name them (Tian's ``aHIP rh``); NaN for a parcel
  without grayordinates (two of Schaefer's 1000);
- ``parquet/functional_subjects.parquet``: the subjects;
- ``parquet/<a>_functional_gradients.parquet``,
  ``parquet/<a>_functional_gradient_variance.parquet``;
- ``tsv/<a>_functional.tsv.gz`` (``region1  region2  r``, two parcels),
  ``tsv/<a>_tian_functional.tsv.gz`` (``region  tian_region  r``),
  ``tsv/tian_functional.tsv.gz`` (two of Tian's regions),
  ``tsv/<a>_functional_gradients.tsv.gz`` and
  ``tsv/<a>_functional_gradient_variance.tsv.gz``: the NeuroLang
  relations, each pair both ways round, to 6 decimals.

    python combine.py WORK PARTS DATASET
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] /
                       "hcp_structural_connectivity"))
from gradients import gradients  # noqa: E402

# gzip without the time, so that a file made again is the same
GZIP = {"method": "gzip", "mtime": 0}


def pairs(matrix: pd.DataFrame, rows, columns, names) -> pd.DataFrame:
    """The long table of *matrix* between *rows* and *columns*, the
    diagonal and the NaN left out, the columns named *names*."""
    block = matrix.loc[rows, columns].stack().reset_index()
    block.columns = [*names, "r"]
    block = block[block[names[0]] != block[names[1]]]
    block["r"] = block["r"].round(6)
    return block


def tsv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, compression=GZIP)


def main(work: Path, parts: Path, dataset: Path) -> None:
    atlases = np.load(work / "atlases.npz")
    names = [a for a in atlases.files if not a.endswith(":names")]
    tian = [line.strip().replace("-", " ") for line in
            (work / "atlases" / "Tian_Subcortex_S2_3T_label.txt")
            .read_text().splitlines() if line.strip()]
    files = sorted(parts.glob("part-*.npz"))
    subjects = np.concatenate([np.load(f)["subjects"] for f in files])
    n = len(subjects)
    (dataset / "parquet").mkdir(parents=True, exist_ok=True)
    (dataset / "tsv").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": [str(s) for s in subjects]}).to_parquet(
        dataset / "parquet" / "functional_subjects.parquet", index=False)
    for a in names:
        sums = sum(np.load(f)[f"{a}:sum"] for f in files)
        squares = sum(np.load(f)[f"{a}:squares"] for f in files)
        mean_z = sums / n
        sd_z = np.sqrt(np.maximum(squares / n - mean_z ** 2, 0))
        regions = [str(x) for x in atlases[f"{a}:names"]]
        index = pd.Index(regions + tian, name="region")
        r = pd.DataFrame(np.tanh(mean_z), index=index, columns=list(index))
        np.fill_diagonal(r.values, 1.0)
        z = pd.DataFrame(mean_z, index=index, columns=list(index))
        for frame, kind in ((r, "functional_mean"), (
                pd.DataFrame(sd_z, index=index, columns=list(index)),
                "functional_z_sd")):
            frame.to_parquet(dataset / "parquet" / f"{a}_{kind}.parquet",
                             compression="zstd")
        tsv(pairs(r, regions, regions, ["region1", "region2"]),
            dataset / "tsv" / f"{a}_functional.tsv.gz")
        tsv(pairs(r, regions, tian, ["region", "tian_region"]),
            dataset / "tsv" / f"{a}_tian_functional.tsv.gz")
        if a == "schaefer100_17":  # Tian's own, the same for every atlas
            tsv(pairs(r, tian, tian, ["region1", "region2"]),
                dataset / "tsv" / "tian_functional.tsv.gz")
        scores, variance, matched = gradients(
            z.loc[regions, regions], 10, transform=lambda values: values)
        scores.to_parquet(dataset / "parquet" /
                          f"{a}_functional_gradients.parquet")
        variance.to_parquet(dataset / "parquet" /
                            f"{a}_functional_gradient_variance.parquet",
                            index=False)
        long = scores.reset_index().melt(
            id_vars="region", var_name="dimension", value_name="value")
        long["dimension"] = long["dimension"].astype(int)
        tsv(long.dropna(), dataset / "tsv" / f"{a}_functional_gradients.tsv.gz")
        tsv(variance, dataset / "tsv" /
            f"{a}_functional_gradient_variance.tsv.gz")
        upper = np.triu_indices(len(regions), 1)
        print(f"{a}: mean r {np.nanmean(r.loc[regions, regions].values[upper]):.3f}; "
              f"gradients explain " + ", ".join(
                  f"{x:.3f}" for x in variance[variance.hemisphere == "L"]
                  .explained[:3]), flush=True)
    print(f"{n} subjects")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
