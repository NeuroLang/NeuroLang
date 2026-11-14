"""
Parcellating, and the group statistics of the subjects' matrices.

A task of a worker keeps, for each atlas, the sum of its subjects'
matrices and of their squares, and its subjects, in a part
(``OUT/part-<task>.npz``), saved every few subjects and at its end, so
that a task started again goes on where it stopped (the subjects since
the last save are done again); :func:`group` makes the mean and the
standard deviation of all the parts.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def indicator(parcels: np.ndarray, n: int, mean: bool = False):
    """The matrix that sums (or, with *mean*, averages) the rows of each of
    the *n* parcels (*parcels*: the parcel of each row, from 1; 0 for
    none)."""
    inside = np.flatnonzero(parcels)
    weights = np.ones(len(inside))
    if mean:
        counts = np.bincount(parcels[inside] - 1, minlength=n)
        weights = 1.0 / counts[parcels[inside] - 1]
    return sp.csr_matrix((weights, (inside, parcels[inside] - 1)),
                         shape=(len(parcels), n))


class Part:
    """The sums of the matrices of the subjects of a task, by atlas."""

    def __init__(self, path: Path, every: int = 10):
        self.path = Path(path)
        self.every = every
        self.sums, self.squares, self.subjects, self.extra = {}, {}, [], {}
        if self.path.exists():
            stored = np.load(self.path)
            self.subjects = [str(s) for s in stored["subjects"]]
            for key in stored.files:
                kind, _, atlas = key.partition(":")
                if key.endswith((":sum", ":squares")):  # the first format
                    atlas, _, kind = key.rpartition(":")
                if kind == "sum":
                    self.sums[atlas] = stored[key]
                elif kind == "squares":
                    self.squares[atlas] = stored[key]
                elif kind == "extra":
                    self.extra[atlas] = list(stored[key])

    def add(self, subject: str, matrices: dict, **extra) -> None:
        """Add the *matrices* of *subject*, by atlas, and its *extra*
        values (one per subject, by name); save every *every* subjects."""
        for atlas, matrix in matrices.items():
            self.sums[atlas] = self.sums.get(atlas, 0.0) + matrix
            self.squares[atlas] = self.squares.get(atlas, 0.0) + matrix ** 2
        for name, value in extra.items():
            self.extra.setdefault(name, []).append(value)
        self.subjects.append(subject)
        if len(self.subjects) % self.every == 0:
            self.save()

    def save(self) -> None:
        if not self.subjects:
            return
        partial = self.path.with_name(self.path.stem + ".partial.npz")
        np.savez(partial, subjects=np.array(self.subjects),
                 **{f"sum:{a}": m for a, m in self.sums.items()},
                 **{f"squares:{a}": m for a, m in self.squares.items()},
                 **{f"extra:{k}": np.array(v) for k, v in self.extra.items()})
        partial.rename(self.path)
        # the subjects done, for a shell to read (relay.sh)
        self.path.with_suffix(".subjects").write_text(
            "".join(f"{s}\n" for s in self.subjects))


def parts(folder: Path) -> list:
    return [Part(p) for p in sorted(Path(folder).glob("part-*.npz"))
            if not p.name.endswith(".partial.npz")]


def done(folder: Path) -> set:
    """The subjects of the parts of *folder*."""
    return {s for part in parts(folder) for s in part.subjects}


def group(folder: Path):
    """The mean and the standard deviation over the subjects of the parts
    of *folder*, by atlas, their subjects, and their extra values."""
    found = parts(folder)
    subjects = [s for part in found for s in part.subjects]
    n = len(subjects)
    mean, sd = {}, {}
    for atlas in found[0].sums:
        total = sum(part.sums[atlas] for part in found)
        squares = sum(part.squares[atlas] for part in found)
        mean[atlas] = total / n
        sd[atlas] = np.sqrt(np.maximum(squares / n - mean[atlas] ** 2, 0))
    extra = {k: [v for part in found for v in part.extra.get(k, [])]
             for k in found[0].extra}
    return mean, sd, subjects, extra
