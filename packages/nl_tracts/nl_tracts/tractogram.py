"""Tractograms: streamlines in flat arrays (DESIGN.md, 2 and 7)."""

import os
from typing import Any, Union

import numpy as np

from .spaces import Space

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
        ``(P, 3)`` float, in mm, in the space's world coordinates (RAS+).
    offsets : numpy.ndarray
        ``(S + 1,)`` int64, non-decreasing, ``offsets[0] == 0``,
        ``offsets[-1] == P``.
    space : Space
        The space tag.
    """

    def __init__(
        self, id: str, points: np.ndarray, offsets: np.ndarray, space: Space
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_file(
        cls, id: str, path: Union[str, os.PathLike], space: Space
    ) -> "Tractogram":
        """Read a ``.trk``, ``.tck`` or ``.vtk``/``.vtp`` with nibabel.

        The points are taken to RAS+ mm (nibabel's ``to_rasmm``); the file's
        reference image is not otherwise used.
        """
        raise NotImplementedError

    @classmethod
    def from_streamlines(
        cls, id: str, streamlines: Any, space: Space
    ) -> "Tractogram":
        """From a nibabel ``ArraySequence`` or a list of (n_i, 3) arrays."""
        raise NotImplementedError

    @property
    def n_streamlines(self) -> int:
        raise NotImplementedError

    def lengths_in_points(self) -> np.ndarray:
        """``(S,)`` number of points of each streamline."""
        raise NotImplementedError

    def lengths_in_mm(self) -> np.ndarray:
        """``(S,)`` polyline length of each streamline, in mm."""
        raise NotImplementedError

    def content_hash(self) -> str:
        """Hash of the points, offsets and space (not of the id)."""
        raise NotImplementedError
