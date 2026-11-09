"""What tractography makes of an atlas (DESIGN.md, 3 and 4).

The atlas is NeuroLang's :class:`neurolang.atlases.Atlas`, the one
``.load_atlas``, ``add_atlas_set`` and the ``atlases:`` of an engine load.
nl_tracts adds what is specific to tracts: the term mapping of its labels
(:mod:`nl_tracts.terms`) and the catalogue relations ``label``,
``label_term`` and ``label_extent``.
"""

import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import ndimage

from neurolang.atlases import Atlas, label_name

from .terms import COLUMNS as TERM_COLUMNS
from .terms import load_term_mapping

__all__ = [
    "DEFAULT_TERM_MAPPINGS",
    "ProbabilisticAtlas",
    "atlas_term_mapping",
    "label_extent_relation",
    "label_relation",
    "label_term_relation",
    "unmapped_labels",
]

#: The shipped term mapping of an atlas by its kind (``Atlas.kind``), the
#: one ``.index_tracts`` attaches unless given another: FreeSurfer's
#: ``aseg``/``aparc`` (and ``wmparc``) names for a colour LUT, a2009s for
#: Destrieux, Tian's scale 2 for Tian. Schaefer has none.
DEFAULT_TERM_MAPPINGS = {
    "freesurfer": "freesurfer_aseg_aparc",
    "destrieux": "freesurfer_destrieux",
    "nilearn:destrieux_2009": "freesurfer_destrieux",
    "tian": "tian_subcortex_s2",
}


def atlas_term_mapping(atlas: Atlas, term_mapping: Any = None) -> pd.DataFrame:
    """The term mapping of ``atlas``: ``term_mapping`` (a DataFrame, a path
    or the key of a shipped one) if given, else that of its kind in
    :data:`DEFAULT_TERM_MAPPINGS`, else an empty one."""
    if term_mapping is None:
        term_mapping = DEFAULT_TERM_MAPPINGS.get(atlas.kind)
    return _mapping(term_mapping)


def _mapping(term_mapping: Any) -> pd.DataFrame:
    if term_mapping is None:
        return pd.DataFrame({c: pd.Series(dtype=str) for c in TERM_COLUMNS})
    return load_term_mapping(term_mapping)


def _column_a(atlas: Atlas, n: int) -> np.ndarray:
    """The atlas's name, ``n`` times. A DataFrame holds it in pandas'
    string dtype (Arrow-backed from pandas 3, DESIGN.md, 7.2)."""
    return np.full(n, atlas.name, dtype=object)


def _matched(atlas: Atlas, mapping: pd.DataFrame) -> pd.DataFrame:
    """The rows of the mapping of each label of the atlas: names match as
    :func:`neurolang.atlases.label_name` writes them, so that a mapping of
    raw names serves an atlas whose names were normalised, as those an
    engine fetches."""
    table = atlas.table[["id", "name"]].assign(
        key=atlas.table["name"].map(label_name)
    )
    return table.merge(
        mapping.assign(key=mapping["atlas_label_name"].map(label_name)),
        on="key",
    )


def _catalogue(atlas: Atlas, mapping: pd.DataFrame) -> pd.DataFrame:
    """The labels of ``streamline_label``: those of the table, with their
    hemisphere (the mapping's ``hemi`` a hint, :meth:`Atlas.hemispheres`),
    and the ids of the volume the table does not name, as ``label_<id>``
    and ``none``."""
    matched = _matched(atlas, mapping[mapping["hemi"] != ""])
    hints = dict(zip(matched["name"], matched["hemi"]))
    frame = atlas.table.assign(hemi=atlas.hemispheres(hints).to_numpy())
    unlisted = atlas.unlisted_labels
    if unlisted:
        frame = pd.concat([frame, pd.DataFrame({
            "id": np.asarray(unlisted, np.int32),
            "name": [f"label_{i}" for i in unlisted],
            "hemi": "none",
        })]).sort_values("id").reset_index(drop=True)
    return frame


def label_relation(
    atlas: Atlas, term_mapping: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """``label(a, l, name, hemi)``: columns ``a, l, name, hemi``.

    The labels of the table and those of the volume it does not name
    (``label_<id>``, ``none``), whose streamlines are in
    ``streamline_label``; the hemisphere is that of the table, else that
    of the term mapping, else the inference (:meth:`Atlas.hemispheres`).
    """
    t = _catalogue(atlas, _mapping(term_mapping))
    return pd.DataFrame(
        {
            "a": _column_a(atlas, len(t)),
            "l": t["id"].to_numpy(np.int32),
            "name": t["name"].to_numpy(object),
            "hemi": t["hemi"].to_numpy(object),
        }
    )


def label_term_relation(
    atlas: Atlas, term_mapping: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """``label_term(a, l, term)``: columns ``a, l, term``."""
    joined = _matched(atlas, _mapping(term_mapping))
    return pd.DataFrame(
        {
            "a": _column_a(atlas, len(joined)),
            "l": joined["id"].to_numpy(np.int32),
            "term": joined["term"].to_numpy(object),
        }
    )


def unmapped_labels(
    atlas: Atlas, term_mapping: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """The labels of the atlas that have no term.

    The background is not a region and is not reported. Use it to see
    what a term mapping leaves out: a label without a term cannot be
    reached from a rule over ``label_term``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``name``, ``hemi``, by id.
    """
    mapping = _mapping(term_mapping)
    t = _catalogue(atlas, mapping)
    mapped = set(_matched(atlas, mapping)["id"])
    unmapped = ~t["id"].isin(mapped)
    return t[unmapped & (t["id"] != atlas.background)].reset_index(drop=True)


def label_extent_relation(atlas: Atlas) -> pd.DataFrame:
    """``label_extent(a, l, axis, lo, hi)``.

    The extent of each label along each world axis (0, 1, 2 = x, y, z
    of the space), in mm, **by voxel edges**: a label whose first and
    last voxel are ``start`` and ``stop - 1`` along an axis spans
    ``[start - 0.5, stop - 0.5]`` in voxel units, the physical extent
    of its voxels. The 8 corners of that box are mapped by the whole
    affine, and the extent is their minimum and maximum, so axes that
    are flipped or permuted need no care and ``lo < hi`` always.

    It is **exactly** the extent of the label's voxels if each world
    axis depends on one voxel axis (axis-aligned, flipped or permuted
    volumes), and the box of the label's *box* under an oblique affine,
    which contains the extent of its voxels. This differs from
    ``tract_querier``'s ``compute_label_bounding_boxes``, which maps
    two corners ``[start, stop]`` (half a voxel off at both ends) and
    is not a box under an oblique affine (DESIGN.md, 6.1).

    The background has no row, nor has an id absent from the volume.
    All the labels are done at once, from one ``find_objects``.
    """
    volume = atlas.labels
    if atlas.background != 0:
        # find_objects skips 0: swap it with the background
        volume = volume.copy()
        volume[atlas.labels == atlas.background] = 0
        volume[atlas.labels == 0] = atlas.background
    found = [
        (i, o) for i, o in enumerate(ndimage.find_objects(volume))
        if o is not None
    ]
    ids = np.array([i + 1 for i, _ in found], dtype=np.int64)
    ids[ids == atlas.background] = 0
    start = np.array([[s.start for s in o] for _, o in found], float)
    stop = np.array([[s.stop for s in o] for _, o in found], float)
    start, stop = start.reshape(-1, 3), stop.reshape(-1, 3)
    lo, hi = start - 0.5, stop - 0.5  # voxel edges
    bits = np.array(list(itertools.product((0, 1), repeat=3)), bool)
    corners = np.where(bits[None], hi[:, None, :], lo[:, None, :])
    world = corners @ atlas.affine[:3, :3].T + atlas.affine[:3, 3]
    n = len(ids)
    return pd.DataFrame(
        {
            "a": _column_a(atlas, 3 * n),
            "l": np.repeat(ids, 3).astype(np.int32),
            "axis": np.tile(np.arange(3), n).astype(np.int8),
            "lo": world.min(axis=1).reshape(-1).astype(np.float32),
            "hi": world.max(axis=1).reshape(-1).astype(np.float32),
        }
    )


class ProbabilisticAtlas:
    """Extension point for 4D (probability map) atlases. Not implemented.

    A probabilistic atlas holds, per voxel, a probability for each label.
    It will produce *probabilistic* facts ``streamline_label_p(a, x, l)``
    with the probability of the streamline traversing the label, instead
    of the boolean ``streamline_label``; it shares ``label``,
    ``label_term`` and the loaders of :class:`~neurolang.atlases.Atlas`,
    and the engine side is ``add_probabilistic_facts_from_tuples``
    (DESIGN.md, 4.4).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "probabilistic atlases are future work (DESIGN.md, 4.4)"
        )
