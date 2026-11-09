"""Tractograms: streamlines in flat arrays (DESIGN.md, 2 and 7)."""

import hashlib
import os
import warnings
from typing import Any, Union

import nibabel as nib
import numpy as np

from neurolang.atlases import Space

__all__ = ["Tractogram"]


class Tractogram:
    """The streamlines of one subject or run, in one space.

    The streamlines are stored as nibabel's ``ArraySequence`` does: one
    ``(P, 3)`` array of all the points and an ``(S + 1,)`` array of
    offsets, so that streamline ``s`` is ``points[offsets[s]:offsets[s+1]]``.
    A streamline is an integer id ``s`` (its position in the file), never
    an object.

    Parameters
    ----------
    id : str
        The tractogram's identifier, the value of the ``t`` column of the
        relations (a subject or run id).
    points : numpy.ndarray
        ``(P, 3)`` float, in mm, in the space's world coordinates (RAS+);
        kept as float32.
    offsets : numpy.ndarray
        ``(S + 1,)`` int64, increasing, ``offsets[0] == 0``,
        ``offsets[-1] == P``.
    space : Space
        The space tag.
    drop_short : bool
        If True, the streamlines of one point are kept in the arrays, so
        that the positions ``s`` of the others are still those of the file,
        but are in no relation (:func:`nl_tracts.indexing`), with a
        warning that says how many. If False (the default), they are an
        error (DESIGN.md, 15.37).

    Raises
    ------
    ValueError
        If the arrays are inconsistent, or a streamline has no point, or
        one point and not ``drop_short`` (tract_querier fails on it too);
        the message gives its id.
    """

    def __init__(
        self, id: str, points: np.ndarray, offsets: np.ndarray, space: Space,
        drop_short: bool = False,
    ) -> None:
        points = np.asarray(points)
        offsets = np.asarray(offsets, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points are (P, 3), not {points.shape}"
            )
        if (
            offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0
            or offsets[-1] != len(points) or (np.diff(offsets) < 0).any()
        ):
            raise ValueError(
                "offsets start at 0, do not decrease and end at the number "
                f"of points ({len(points)})"
            )
        short = np.flatnonzero(np.diff(offsets) < (1 if drop_short else 2))
        if len(short):
            s = int(short[0])
            raise ValueError(
                f"streamline {s} has {int(offsets[s + 1] - offsets[s])} "
                "points: a streamline needs at least 2 points (prune the "
                "short ones first"
                + ("" if drop_short else ", or drop_short=True") + ")"
            )
        short = int((np.diff(offsets) < 2).sum())
        if short:
            warnings.warn(
                f"tractogram {id!r}: {short} streamlines of one point are "
                "in no relation; the others keep their positions in the "
                "file as ids s"
            )
        self.id = id
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.offsets = offsets
        self.space = space

    @classmethod
    def from_file(
        cls, id: str, path: Union[str, os.PathLike], space: Space,
        drop_short: bool = False,
    ) -> "Tractogram":
        """Read a ``.trk``, ``.tck`` or ``.vtk``/``.vtp`` with nibabel.

        The points are taken to RAS+ mm (nibabel's ``to_world``); the file's
        reference image is not otherwise used.
        """
        tractogram = nib.streamlines.load(str(path)).tractogram.to_world()
        return cls.from_streamlines(
            id, tractogram.streamlines, space, drop_short)

    @classmethod
    def from_streamlines(
        cls, id: str, streamlines: Any, space: Space,
        drop_short: bool = False,
    ) -> "Tractogram":
        """From a nibabel ``ArraySequence`` or a list of ``(n_i, 3)``
        arrays."""
        lines = [np.asarray(line) for line in streamlines]
        lengths = np.array([len(line) for line in lines], dtype=np.int64)
        points = np.concatenate(lines) if lines else np.zeros((0, 3))
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return cls(id, points, offsets, space, drop_short)

    @property
    def n_streamlines(self) -> int:
        return len(self.offsets) - 1

    def lengths_in_points(self) -> np.ndarray:
        """``(S,)`` number of points of each streamline."""
        return np.diff(self.offsets)

    def lengths_in_mm(self) -> np.ndarray:
        """``(S,)`` polyline length of each streamline, in mm (float64)."""
        if not len(self.points):
            return np.zeros(0)
        steps = np.zeros(len(self.points))
        steps[:-1] = np.linalg.norm(
            np.diff(self.points.astype(np.float64), axis=0), axis=1
        )
        # no step from a streamline to the next
        steps[self.offsets[1:] - 1] = 0
        return np.add.reduceat(steps, self.offsets[:-1])

    def content_hash(self) -> str:
        """Hash of the points, offsets and space (not of the id)."""
        h = hashlib.sha256(b"nl_tracts.Tractogram.v1")
        h.update(self.points.astype("<f4").tobytes())
        h.update(self.offsets.astype("<i8").tobytes())
        h.update(f"\nspace={self.space}".encode())
        return h.hexdigest()

    def __repr__(self) -> str:
        return (
            f"Tractogram({self.id!r}, space={self.space!r}, "
            f"streamlines={self.n_streamlines})"
        )
