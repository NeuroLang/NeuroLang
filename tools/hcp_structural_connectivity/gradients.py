"""
The linear gradients of the structural connectivity: a principal component
analysis of the connectivity profiles of the regions of each atlas, within
each hemisphere.

The profile of a region is its row of the group connectivity
(``<atlas>_mean.parquet``, combine.py) towards the regions of its own
hemisphere, log(1 + streamlines), its diagonal 0; the columns are
centred, and the principal components of the regions, their scores, are
the gradients: the first orders the regions of the hemisphere along the
axis of their connectivity that varies most, the second along the next,
orthogonal to it, and so on. A principal component analysis of the whole
brain would find first the two hemispheres, which few streamlines join
(its first component correlates at |r| = 0.92 with the hemisphere): each
hemisphere has its own, and the right one's are rotated onto the left
one's (orthogonal Procrustes over the homologous regions, those of the
same name), so that a dimension is the same axis in both. The sign of
each makes the left hemisphere's largest score positive. These are linear
(PCA), not the diffusion embedding of the functional gradients of
Margulies et al. (2016).

Writes, in DATASET, for each atlas ``a`` of DATASET/parquet:

- ``parquet/<a>_gradients.parquet``: the regions (index ``region``) by
  their first DIMENSIONS gradients (columns ``1``, ``2``...);
- ``parquet/<a>_gradient_variance.parquet``: the fraction of the variance
  of the profiles of each hemisphere each explains (``dimension``,
  ``hemisphere``, ``explained``);
- ``tsv/<a>_gradients.tsv.gz`` (``region  dimension  value``) and
  ``tsv/<a>_gradient_variance.tsv.gz`` (``dimension  hemisphere
  explained``), the relations of the NeuroLang engines.

    python gradients.py DATASET [--dimensions 10]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# The hemisphere in a region's name: Schaefer's ' LH ', Destrieux's 'L ',
# Glasser's 'L_'.
HEMISPHERE = re.compile(r"(^|\s)(L|R)H?(\s|_)|^(L|R)_")


def hemisphere(name: str) -> str:
    match = HEMISPHERE.search(name)
    return (match.group(2) or match.group(4)) if match else ""


def homologue(name: str) -> str:
    """The name of a region, its hemisphere's letter made X."""
    match = HEMISPHERE.search(name)
    start = match.start(2) if match.group(2) else match.start(4)
    return name[:start] + "X" + name[start + 1:]


def components(profiles: np.ndarray, dimensions: int):
    """The scores of the rows of *profiles* on their first principal
    components, and the total variance of the profiles."""
    centred = profiles - profiles.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :dimensions] * s[:dimensions], (s ** 2).sum()


def gradients(mean: pd.DataFrame, dimensions: int):
    """The gradients of the regions of the connectivity *mean*, and the
    fraction of the variance of each hemisphere each explains."""
    names = mean.index.to_numpy()
    sides = np.array([hemisphere(n) for n in names])
    assert set(sides) == {"L", "R"}, set(sides)
    connectivity = np.log1p(mean.to_numpy(dtype=float))
    np.fill_diagonal(connectivity, 0)
    scores, totals = {}, {}
    for side in "LR":
        rows = np.flatnonzero(sides == side)
        scores[side], totals[side] = components(
            connectivity[np.ix_(rows, rows)], dimensions)
        scores[side] = pd.DataFrame(scores[side], index=names[rows])
    left, right = scores["L"], scores["R"]
    # the right hemisphere's axes onto the left's, over the homologues
    pairs = pd.Series(left.index, index=[homologue(n) for n in left.index])
    matched = [(pairs[homologue(n)], n) for n in right.index
               if homologue(n) in pairs.index]
    a = right.loc[[r for _, r in matched]].to_numpy()
    b = left.loc[[l for l, _ in matched]].to_numpy()
    u, _, vt = np.linalg.svd(a.T @ b)
    right = pd.DataFrame(right.to_numpy() @ (u @ vt), index=right.index)
    # the left hemisphere's largest score positive
    flip = np.sign(left.to_numpy()[np.abs(left.to_numpy()).argmax(axis=0),
                                   np.arange(left.shape[1])])
    left, right = left * flip, right * flip
    columns = [str(d) for d in range(1, left.shape[1] + 1)]
    result = pd.concat([left, right]).loc[mean.index]
    result.columns = columns
    result.index.name = "region"
    variance = pd.DataFrame([
        {"dimension": d, "hemisphere": side,
         "explained": float((frame.iloc[:, d - 1] ** 2).sum() / totals[side])}
        for side, frame in (("L", left), ("R", right))
        for d in range(1, left.shape[1] + 1)
    ])
    return result, variance, len(matched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dimensions", type=int, default=10)
    args = parser.parse_args()
    for path in sorted((args.dataset / "parquet").glob("*_mean.parquet")):
        atlas = path.name[:-len("_mean.parquet")]
        scores, variance, matched = gradients(pd.read_parquet(path),
                                              args.dimensions)
        scores.to_parquet(args.dataset / "parquet" /
                          f"{atlas}_gradients.parquet")
        variance.to_parquet(args.dataset / "parquet" /
                            f"{atlas}_gradient_variance.parquet",
                            index=False)
        long = scores.reset_index().melt(
            id_vars="region", var_name="dimension", value_name="value")
        long["dimension"] = long["dimension"].astype(int)
        long.to_csv(args.dataset / "tsv" / f"{atlas}_gradients.tsv.gz",
                    sep="\t", index=False, float_format="%.6g")
        variance.to_csv(
            args.dataset / "tsv" / f"{atlas}_gradient_variance.tsv.gz",
            sep="\t", index=False, float_format="%.6g")
        left = variance[variance.hemisphere == "L"].explained
        print(f"{atlas}: {matched} homologous pairs; left explained "
              + ", ".join(f"{x:.3f}" for x in left[:4]))


if __name__ == "__main__":
    main()
