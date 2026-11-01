"""Bulk spatial indexing of a tractogram by an atlas (DESIGN.md, 6 and 7).

Every function is vectorised over all the points of the tractogram; no
function loops over streamlines in Python. The semantics are those of
``tract_querier.tract_label_indices`` (DESIGN.md, 6.1).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import os

import numpy as np
import pandas as pd

from .atlas import Atlas
from .tractogram import Tractogram

__all__ = [
    "IndexingParams",
    "RelationStore",
    "build_relations",
    "endpoint_coord_relation",
    "endpoint_relation",
    "streamline_extent_relation",
    "streamline_length_relation",
    "traverses_relation",
    "points_to_voxels",
]


@dataclass(frozen=True)
class IndexingParams:
    """The parameters that determine the atlas-dependent relations.

    Attributes
    ----------
    crossing_threshold : float
        Percent of its points a streamline must have in a label to
        traverse it: ``count / n_points >= crossing_threshold / 100``.
        The default, 2.0, is tract_querier's.
    length_threshold : float
        Streamlines shorter than this, in mm, are left out of *every*
        relation (their ids are not reused). 0 keeps all.
    emit_background : bool
        If True, ``traverses`` and ``endpoint`` also have rows for the
        atlas's background label. Default False (DESIGN.md, 7.3).
    """

    crossing_threshold: float = 2.0
    length_threshold: float = 0.0
    emit_background: bool = False


def points_to_voxels(
    points: np.ndarray, affine: np.ndarray, shape: tuple
) -> np.ndarray:
    """Voxel indices of points: inverse affine, ``np.round``, then clip.

    ``np.round`` rounds half to even, as tract_querier does. Points that
    fall outside the volume are clipped to its border, and one warning
    reports how many.

    Returns
    -------
    numpy.ndarray
        ``(P, 3)`` int64.
    """
    raise NotImplementedError


def traverses_relation(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> pd.DataFrame:
    """``traverses(a, t, s, l)``: columns ``a, t, s, l``.

    ``(a, t, s, l)`` is a row iff at least ``crossing_threshold`` percent of
    the points of streamline ``s`` are in label ``l`` (the denominator
    counts all the points, including those in the background).
    """
    raise NotImplementedError


def endpoint_relation(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> pd.DataFrame:
    """``endpoint(a, t, s, e, l)``: columns ``a, t, s, e, l``.

    ``e`` is 0 for the first and 1 for the last point of the streamline,
    ``l`` the label of that point.
    """
    raise NotImplementedError


def streamline_extent_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``streamline_extent(t, s, axis, lo, hi)``: atlas-independent.

    The bounding box of the streamline's points in world mm; ``axis`` is
    0, 1, 2 for x, y, z.
    """
    raise NotImplementedError


def endpoint_coord_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``endpoint_coord(t, s, e, axis, v)``: atlas-independent."""
    raise NotImplementedError


def streamline_length_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``streamline_length(t, s, mm)``: atlas-independent (an addition to
    the schema, DESIGN.md, 3.3)."""
    raise NotImplementedError


@dataclass
class RelationStore:
    """The relations of any number of atlases and tractograms.

    A NeuroLang engine's ``add_tuple_set`` *replaces* a relation, so the
    rows of several atlases and tractograms are accumulated here, and
    handed over once, by :func:`nl_tracts.frontend.register`.

    Attributes
    ----------
    relations : dict of str to pandas.DataFrame
        The relation name to its rows.
    """

    relations: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_atlas(self, atlas: Atlas) -> None:
        """Add ``atlas``, ``label``, ``label_term`` and ``label_extent``."""
        raise NotImplementedError

    def add_tractogram(
        self, tractogram: Tractogram, params: IndexingParams
    ) -> None:
        """Add ``tractogram``, ``streamline`` and the atlas-independent
        relations."""
        raise NotImplementedError

    def add_index(
        self,
        tractogram: Tractogram,
        atlas: Atlas,
        params: IndexingParams,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        allow_space_mismatch: bool = False,
    ) -> None:
        """Add ``traverses`` and ``endpoint`` of ``tractogram`` by ``atlas``.

        Checks the spaces (:class:`~nl_tracts.spaces.SpaceMismatchError`),
        and reads or writes the parquet cache if ``cache_dir`` is given.
        """
        raise NotImplementedError


def build_relations(
    tractogram: Tractogram,
    atlas: Atlas,
    params: IndexingParams = IndexingParams(),
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    allow_space_mismatch: bool = False,
) -> RelationStore:
    """All the relations of one tractogram and one atlas.

    The one-call form of ``RelationStore.add_atlas``, ``add_tractogram``
    and ``add_index``.
    """
    raise NotImplementedError
