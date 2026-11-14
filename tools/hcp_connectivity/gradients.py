"""
The linear gradients of a connectivity: a principal component analysis of
the connectivity profiles of the regions of an atlas, within each
hemisphere.

The profile of a region is its row of the group connectivity (combine.py)
towards the regions of its own hemisphere, transformed (log(1 +
streamlines) for the structural connectivity, the Fisher z itself for the
functional one), its diagonal 0; the columns are centred, and the
principal components of the regions, their scores, are
the gradients: the first orders the regions of the hemisphere along the
axis of their connectivity that varies most, the second along the next,
orthogonal to it, and so on. A principal component analysis of the whole
brain would find first the two hemispheres, which few streamlines join
(for the structural connectivity, its first component correlates at
|r| = 0.92 with the hemisphere): each
hemisphere has its own, and the right one's are rotated onto the left
one's (orthogonal Procrustes over the homologous regions, those of the
same name), so that a dimension is the same axis in both. The sign of
each makes the left hemisphere's largest score positive. These are linear
(PCA), not the diffusion embedding of the functional gradients of
Margulies et al. (2016).
"""

import numpy as np
import pandas as pd

from .cifti import hemisphere, homologue


def components(profiles: np.ndarray, dimensions: int):
    """The scores of the rows of *profiles* on their first principal
    components, and the total variance of the profiles."""
    centred = profiles - profiles.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :dimensions] * s[:dimensions], (s ** 2).sum()


def gradients(mean: pd.DataFrame, dimensions: int, transform=np.log1p):
    """The gradients of the regions of the connectivity *mean*, *transform*
    of its values the profiles (log(1 + streamlines); the functional
    connectivity's own Fisher z), and the fraction of the variance of each
    hemisphere each explains. A region of NaN values (without data) has
    NaN gradients."""
    defined = ~mean.isna().all(axis=1)
    result_index = mean.index
    mean = mean.loc[defined, defined]
    names = mean.index.to_numpy()
    sides = np.array([hemisphere(n) for n in names])
    assert set(sides) == {"L", "R"}, set(sides)
    connectivity = transform(mean.to_numpy(dtype=float))
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
    result = pd.concat([left, right]).reindex(result_index)
    result.columns = columns
    result.index.name = "region"
    variance = pd.DataFrame([
        {"dimension": d, "hemisphere": side,
         "explained": float((frame.iloc[:, d - 1] ** 2).sum() / totals[side])}
        for side, frame in (("L", left), ("R", right))
        for d in range(1, left.shape[1] + 1)
    ])
    return result, variance, len(matched)
