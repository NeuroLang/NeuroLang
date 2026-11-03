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
        If True, ``traverses`` also has rows for the atlas's background
        label. Default False (DESIGN.md, 6.3). ``endpoint`` never has rows
        for the background label, whatever this says.
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


def _kept_streamlines(
    tractogram: Tractogram, params: IndexingParams
) -> np.ndarray:
    """The ids of the streamlines the relations have: all of them, or those
    at least ``length_threshold`` mm long. The ids are never renumbered
    (DESIGN.md, 6.1, parity 7)."""
    ids = np.arange(tractogram.n_streamlines)
    if params.length_threshold > 0:
        ids = ids[tractogram.lengths_in_mm() >= params.length_threshold]
    return ids


def _column_t(tractogram: Tractogram, n: int) -> np.ndarray:
    return np.full(n, tractogram.id, dtype=object)


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
    ``l`` the label of that point. An end in the background has no row
    (DESIGN.md, 6.3).
    """
    raise NotImplementedError


def streamline_extent_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``streamline_extent(t, s, axis, lo, hi)``: atlas-independent.

    The bounding box of the streamline's points in world mm; ``axis`` is
    0, 1, 2 for x, y, z. One ``reduceat`` per bound over all the points.
    """
    ids = _kept_streamlines(tractogram, params)
    start = tractogram.offsets[:-1]
    low = np.minimum.reduceat(tractogram.points, start, axis=0)[ids]
    high = np.maximum.reduceat(tractogram.points, start, axis=0)[ids]
    n = len(ids)
    return pd.DataFrame(
        {
            "t": _column_t(tractogram, 3 * n),
            "s": np.repeat(ids, 3).astype(np.int32),
            "axis": np.tile(np.arange(3), n).astype(np.int8),
            "lo": low.reshape(-1),
            "hi": high.reshape(-1),
        }
    )


def endpoint_coord_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``endpoint_coord(t, s, e, axis, v)``: atlas-independent.

    The coordinates of the first (``e`` 0) and the last (``e`` 1) point of
    each streamline, in world mm.
    """
    ids = _kept_streamlines(tractogram, params)
    ends = np.stack(
        [
            tractogram.points[tractogram.offsets[:-1]],
            tractogram.points[tractogram.offsets[1:] - 1],
        ],
        axis=1,
    )[ids]  # (K, 2, 3)
    n = len(ids)
    return pd.DataFrame(
        {
            "t": _column_t(tractogram, 6 * n),
            "s": np.repeat(ids, 6).astype(np.int32),
            "e": np.tile(np.repeat(np.arange(2), 3), n).astype(np.int8),
            "axis": np.tile(np.arange(3), 2 * n).astype(np.int8),
            "v": ends.reshape(-1),
        }
    )


def streamline_length_relation(
    tractogram: Tractogram, params: IndexingParams
) -> pd.DataFrame:
    """``streamline_length(t, s, mm)``: atlas-independent (an addition to
    the schema, DESIGN.md, 3.1)."""
    ids = _kept_streamlines(tractogram, params)
    return pd.DataFrame(
        {
            "t": _column_t(tractogram, len(ids)),
            "s": ids.astype(np.int32),
            "mm": tractogram.lengths_in_mm()[ids].astype(np.float32),
        }
    )


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

    def _append(self, name: str, frame: pd.DataFrame) -> None:
        if name in self.relations:
            frame = pd.concat(
                [self.relations[name], frame], ignore_index=True
            )
        self.relations[name] = frame

    def add_atlas(self, atlas: Atlas) -> None:
        """Add ``atlas``, ``label``, ``label_term`` and ``label_extent``.

        Raises
        ------
        ValueError
            If an atlas of that name was added: it would have its rows twice.
        """
        known = self.relations.get("atlas")
        if known is not None and atlas.name in set(known["a"]):
            raise ValueError(f"atlas {atlas.name!r} was already added")
        self._append(
            "atlas", pd.DataFrame({"a": np.array([atlas.name], dtype=object)})
        )
        self._append("label", atlas.label_relation())
        self._append("label_term", atlas.label_term_relation())
        self._append("label_extent", atlas.label_extent_relation())

    def add_tractogram(
        self, tractogram: Tractogram, params: IndexingParams
    ) -> None:
        """Add ``tractogram``, ``streamline`` and the atlas-independent
        relations (``streamline_extent``, ``endpoint_coord``,
        ``streamline_length``).

        Raises
        ------
        ValueError
            If a tractogram of that id was added.
        """
        known = self.relations.get("tractogram")
        if known is not None and tractogram.id in set(known["t"]):
            raise ValueError(
                f"tractogram {tractogram.id!r} was already added"
            )
        ids = _kept_streamlines(tractogram, params)
        self._append(
            "tractogram",
            pd.DataFrame({"t": np.array([tractogram.id], dtype=object)}),
        )
        self._append(
            "streamline",
            pd.DataFrame(
                {
                    "t": _column_t(tractogram, len(ids)),
                    "s": ids.astype(np.int32),
                }
            ),
        )
        self._append(
            "streamline_extent", streamline_extent_relation(tractogram, params)
        )
        self._append(
            "endpoint_coord", endpoint_coord_relation(tractogram, params)
        )
        self._append(
            "streamline_length", streamline_length_relation(tractogram, params)
        )

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
