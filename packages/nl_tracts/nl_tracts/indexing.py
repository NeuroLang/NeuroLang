"""Bulk spatial indexing of a tractogram by an atlas (DESIGN.md, 6 and 7).

Every function is vectorised over all the points of the tractogram; no
function loops over streamlines in Python. The semantics are those of
``tract_querier.tract_label_indices`` (DESIGN.md, 6.1).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import os

import warnings

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


#: The points of a chunk, at most (a streamline is never split): the
#: temporaries of an indexing are of this size.
CHUNK_POINTS = 1 << 22


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

    def __post_init__(self):
        if not 0 <= self.crossing_threshold <= 100:
            raise ValueError(
                "crossing_threshold is a percentage, between 0 and 100, "
                f"not {self.crossing_threshold}"
            )
        if self.length_threshold < 0:
            raise ValueError(
                f"length_threshold is in mm, not {self.length_threshold}"
            )


def _voxels(points: np.ndarray, inverse: np.ndarray, shape) -> tuple:
    """Voxel indices of points by the inverse affine, and the number of
    points that were outside the volume (and are clipped to its border).
    The expression is the reference's, so that the rounding agrees to the
    last bit."""
    ijk = np.round(
        np.dot(inverse[:3, :3], np.asarray(points, dtype=float).T).T
        + inverse[:3, 3]
    ).astype(np.int64)
    limit = np.asarray(shape)
    outside = int(((ijk < 0) | (ijk >= limit)).any(axis=1).sum())
    return np.clip(ijk, 0, limit - 1), outside


def _warn_outside(outside: int, total: int) -> None:
    if outside:
        warnings.warn(
            f"{outside} of {total} tract points fall outside the image; "
            "they are taken at its border",
            stacklevel=3,
        )


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
    ijk, outside = _voxels(points, np.linalg.inv(affine), shape)
    _warn_outside(outside, len(ijk))
    return ijk


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


def _chunks(tractogram: Tractogram, ids: np.ndarray):
    """``(ids, points, offsets)`` of groups of the streamlines ``ids`` of
    at most ``CHUNK_POINTS`` points (more for one longer streamline): the
    points of each group are gathered with no loop over the streamlines."""
    lengths = tractogram.lengths_in_points()[ids]
    ends = np.cumsum(lengths)
    group = (ends - lengths) // CHUNK_POINTS
    cuts = np.r_[0, np.flatnonzero(np.diff(group)) + 1, len(ids)]
    all_points = tractogram.points
    first = tractogram.offsets[ids]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        size = lengths[lo:hi]
        local = np.r_[0, np.cumsum(size)]
        # the ragged arange: the index of each point of each streamline
        index = np.repeat(first[lo:hi] - local[:-1], size) + np.arange(
            local[-1]
        )
        yield ids[lo:hi], all_points[index], local


def _index(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> tuple:
    """The ``traverses`` and ``endpoint`` relations, in one pass."""
    ids = _kept_streamlines(tractogram, params)
    inverse = np.linalg.inv(atlas.affine)
    present = atlas.present_labels
    n_labels = len(present)
    threshold = params.crossing_threshold / 100.0
    out = {"s": [], "l": [], "es": [], "e": [], "el": []}
    outside = 0
    for chunk_ids, points, offsets in _chunks(tractogram, ids):
        ijk, n_out = _voxels(points, inverse, atlas.labels.shape)
        outside += n_out
        labels = atlas.labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        n_points = np.diff(offsets)
        k = len(chunk_ids)

        # traverses: the points of each streamline by label, as a count of
        # the key (streamline, rank of the label); bincount if it is small
        key = (
            np.repeat(np.arange(k, dtype=np.int64), n_points) * n_labels
            + np.searchsorted(present, labels)
        )
        if k * n_labels <= 1 << 26:
            counts = np.bincount(key, minlength=k * n_labels)
            unique = np.flatnonzero(counts)
            count = counts[unique]
        else:
            unique, count = np.unique(key, return_counts=True)
        line, rank = unique // n_labels, unique % n_labels
        # the denominator is all the points, the background's too; the
        # expression is the reference's (count * 1. / total >= threshold)
        keep = count / n_points[line] >= threshold
        if not params.emit_background:
            keep &= present[rank] != atlas.background
        out["s"].append(chunk_ids[line[keep]])
        out["l"].append(present[rank[keep]])

        # endpoint: the label of the first and of the last point, none for
        # the background
        ends = np.stack(
            [labels[offsets[:-1]], labels[offsets[1:] - 1]], axis=1
        ).reshape(-1)
        in_a_region = ends != atlas.background
        out["es"].append(np.repeat(chunk_ids, 2)[in_a_region])
        out["e"].append(np.tile(np.arange(2), k)[in_a_region])
        out["el"].append(ends[in_a_region])
    _warn_outside(outside, int(tractogram.lengths_in_points()[ids].sum()))

    def column(name, dtype):
        parts = out[name]
        return (np.concatenate(parts) if parts else np.zeros(0)).astype(dtype)

    traverses = pd.DataFrame(
        {
            "a": np.full(len(column("s", np.int32)), atlas.name, dtype=object),
            "t": np.full(len(column("s", np.int32)), tractogram.id,
                         dtype=object),
            "s": column("s", np.int32),
            "l": column("l", np.int32),
        }
    )
    n_ends = len(column("es", np.int32))
    endpoint = pd.DataFrame(
        {
            "a": np.full(n_ends, atlas.name, dtype=object),
            "t": np.full(n_ends, tractogram.id, dtype=object),
            "s": column("es", np.int32),
            "e": column("e", np.int8),
            "l": column("el", np.int32),
        }
    )
    return traverses, endpoint


def traverses_relation(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> pd.DataFrame:
    """``traverses(a, t, s, l)``: columns ``a, t, s, l``.

    ``(a, t, s, l)`` is a row iff at least ``crossing_threshold`` percent of
    the points of streamline ``s`` are in label ``l`` (the denominator
    counts all the points, including those in the background), and at least
    one is. The background has a row only with ``emit_background``.
    """
    return _index(tractogram, atlas, params)[0]


def endpoint_relation(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> pd.DataFrame:
    """``endpoint(a, t, s, e, l)``: columns ``a, t, s, e, l``.

    ``e`` is 0 for the first and 1 for the last point of the streamline,
    ``l`` the label of that point. An end in the background has no row
    (DESIGN.md, 6.3).
    """
    return _index(tractogram, atlas, params)[1]


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
    _indexed: set = field(default_factory=set, repr=False)

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

        Raises
        ------
        ValueError
            If this atlas and this tractogram were indexed already.
        """
        atlas.check_same_space(tractogram, allow_space_mismatch)
        pair = (atlas.name, tractogram.id)
        if pair in self._indexed:
            raise ValueError(
                f"tractogram {tractogram.id!r} was already indexed by atlas "
                f"{atlas.name!r}"
            )
        frames = None
        if cache_dir is not None:
            from .cache import cache_key, read_cached, write_cached

            key = cache_key(tractogram, atlas, params)
            cached = [
                read_cached(cache_dir, key, name)
                for name in ("traverses", "endpoint")
            ]
            if all(c is not None for c in cached):
                frames = [
                    _labelled(c, tractogram, atlas) for c in cached
                ]
        if frames is None:
            frames = list(_index(tractogram, atlas, params))
            if cache_dir is not None:
                for name, frame in zip(("traverses", "endpoint"), frames):
                    write_cached(
                        cache_dir, key, name, frame.drop(columns=["a", "t"])
                    )
        self._append("traverses", frames[0])
        self._append("endpoint", frames[1])
        self._indexed.add(pair)


def _labelled(
    frame: pd.DataFrame, tractogram: Tractogram, atlas: Atlas
) -> pd.DataFrame:
    """A cached relation with its ``a`` and ``t`` columns."""
    frame = frame.copy()
    frame.insert(0, "t", np.full(len(frame), tractogram.id, dtype=object))
    frame.insert(0, "a", np.full(len(frame), atlas.name, dtype=object))
    return frame


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
    store = RelationStore()
    store.add_atlas(atlas)
    store.add_tractogram(tractogram, params)
    store.add_index(
        tractogram, atlas, params, cache_dir=cache_dir,
        allow_space_mismatch=allow_space_mismatch,
    )
    return store
