"""Atlases: a label volume, its affine and a label table (DESIGN.md, 4)."""

import hashlib
import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .spaces import Space, check_same_space
from .terms import COLUMNS as TERM_COLUMNS
from .terms import HEMISPHERES, load_term_mapping

__all__ = [
    "Atlas",
    "HemisphereRule",
    "HEMISPHERES",
    "HEMISPHERE_RULES",
    "ProbabilisticAtlas",
    "infer_hemispheres",
    "load_label_table_bids_dseg",
    "load_label_table_freesurfer_lut",
    "load_label_table_tsv",
]


@dataclass(frozen=True)
class HemisphereRule:
    """Infer the hemisphere of a raw label name.

    Attributes
    ----------
    pattern : str
        A regular expression that must match the *whole* name.
    hemi : str
        ``"left"`` or ``"right"``.
    """

    pattern: str
    hemi: str

    def __post_init__(self):
        if self.hemi not in ("left", "right"):
            raise ValueError(
                f"hemi of a rule is left or right, not {self.hemi!r}"
            )


def _rules(*pairs):
    return tuple(
        HemisphereRule(pattern, hemi)
        for left, right in pairs
        for pattern, hemi in ((left, "left"), (right, "right"))
    )


#: The default, ordered rules; the first that matches wins (DESIGN.md, 4.3):
#: FreeSurfer ``Left-``/``Right-``, ``ctx-lh-``/``ctx-rh-`` and a2009s
#: ``ctx_lh_``/``ctx_rh_``, the suffixes ``_L``/``_R`` (AAL), ``lh.``/
#: ``rh.``, ``.left``/``.right``, and the prefixes ``Left ``/``Right ``
#: (Harvard-Oxford) and ``L ``/``R `` (nilearn's Destrieux).
HEMISPHERE_RULES: Sequence[HemisphereRule] = _rules(
    (r"Left[- ].+", r"Right[- ].+"),
    (r"ctx[-_]lh[-_].+", r"ctx[-_]rh[-_].+"),
    (r".+_L", r".+_R"),
    (r"lh\..+", r"rh\..+"),
    (r".+\.left", r".+\.right"),
    (r"L .+", r"R .+"),
)


def infer_hemispheres(
    names: Iterable[str],
    rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
) -> pd.Series:
    """The hemisphere of raw label names, by the first rule that matches.

    Returns
    -------
    pandas.Series
        ``"left"``, ``"right"`` or ``"none"`` for each name, in order.
    """
    names = pd.Series(list(names), dtype=object)
    hemi = pd.Series("none", index=names.index, dtype=object)
    unset = pd.Series(True, index=names.index)
    for rule in rules:
        matched = (
            unset
            & names.str.fullmatch(rule.pattern).fillna(False).astype(bool)
        )
        hemi[matched] = rule.hemi
        unset &= ~matched
    return hemi


def load_label_table_freesurfer_lut(
    path: Union[str, os.PathLike]
) -> pd.DataFrame:
    """Read a FreeSurfer colour LUT (``FreeSurferColorLUT.txt``).

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (int64) and ``name``; the colours are ignored.
    """
    return pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, usecols=[0, 1],
        names=["id", "name"], dtype={"id": "int64", "name": str},
    )


def load_label_table_bids_dseg(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read a BIDS ``*_dseg.tsv`` (columns ``index`` and ``name``)."""
    frame = pd.read_csv(path, sep="\t", dtype={"name": str})
    if not {"index", "name"} <= set(frame.columns):
        raise ValueError(
            "a dseg.tsv needs the columns index and name, "
            f"has {list(frame.columns)}"
        )
    frame = frame[["index", "name"]].rename(columns={"index": "id"})
    return frame.astype({"id": "int64"})


def load_label_table_tsv(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read a TSV with columns ``id``, ``name`` and optionally ``hemi``.

    Empty ``hemi`` cells are ``""`` (the hemisphere is then inferred).
    """
    frame = pd.read_csv(
        path, sep="\t", dtype={"name": str}, keep_default_na=False,
    )
    if not {"id", "name"} <= set(frame.columns):
        raise ValueError(
            f"a label TSV needs the columns id and name, has "
            f"{list(frame.columns)}"
        )
    columns = ["id", "name"] + (["hemi"] if "hemi" in frame.columns else [])
    return frame[columns].astype({"id": "int64"})


def _label_volume(data: Any) -> np.ndarray:
    """A 3D int32 label volume from integer or integral-float data."""
    data = np.asanyarray(data)
    if data.ndim != 3:
        raise ValueError(f"a label volume is 3D, not {data.ndim}D")
    if data.dtype == bool:
        data = data.astype(np.int32)
    elif not np.issubdtype(data.dtype, np.integer):
        if not (
            np.issubdtype(data.dtype, np.floating)
            and np.isfinite(data).all()
            and np.array_equal(data, np.rint(data))
        ):
            raise ValueError(
                "label volumes hold integer labels; this one has "
                "fractions or non-finite values"
            )
    if data.size and data.min() < 0:
        raise ValueError("negative labels are not supported")
    if data.size and data.max() > np.iinfo(np.int32).max:
        raise ValueError("labels above 2**31 - 1 are not supported")
    return np.ascontiguousarray(data, dtype=np.int32)


def _read_image(image: Union[str, os.PathLike, Any]):
    """Label volume and affine of a path or a nibabel-like image."""
    if isinstance(image, (str, os.PathLike)):
        image = nib.load(image)
    # the data object, not get_fdata: label images may hold integers
    data = np.asanyarray(image.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim > 3:
        raise NotImplementedError(
            "4D images are probabilistic atlases, which are future work "
            "(DESIGN.md, 4.4)"
        )
    return data, np.asarray(image.affine, dtype=float)


def _strings(values: Iterable[Any]) -> list:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


class Atlas:
    """A deterministic atlas: label volume + affine + label table + space.

    Parameters
    ----------
    name : str
        The atlas's identifier, the value of the ``a`` column of every
        label-dependent relation. Unique in an engine.
    labels : numpy.ndarray
        3D volume of non-negative integer labels (floats with integral
        values are accepted); ``background`` marks voxels outside any
        region.
    affine : numpy.ndarray
        4x4, voxel indices (ijk) to the space's world coordinates in mm
        (RAS+, as nibabel): the affine maps an index to the *centre* of
        its voxel.
    table : pandas.DataFrame
        Columns ``id``, ``name`` and optionally ``hemi`` (``left``,
        ``right``, ``none`` or empty). Ids of the table absent from the
        volume are dropped, with a warning; ids of the volume absent from
        the table (other than the background) are named ``label_<id>``,
        with a warning.
    space : Space
        The space tag.
    background : int
        The label that means "no region" (default 0, DESIGN.md, 6.3). It
        has a row in ``label`` only if the table has one, and none in
        ``label_extent``.
    hemisphere_rules : Sequence[HemisphereRule]
        Rules to infer the hemisphere from the name, where nothing else
        says it. Defaults to :data:`HEMISPHERE_RULES`; ``()`` infers
        nothing.
    term_mapping : DataFrame, path or str, optional
        A term mapping (:func:`nl_tracts.terms.load_term_mapping`: columns
        ``atlas_label_name``, ``term``, ``hemi``), or the key of a shipped
        one, such as ``"freesurfer_aseg_aparc"``. Without it ``label_term``
        is empty and every label is reported by :meth:`unmapped_labels`.

    Notes
    -----
    The hemisphere of a label is, in order of precedence: the ``hemi`` of
    the table, if not empty; the ``hemi`` of the term mapping, if the name
    is in it and not empty; the inference by ``hemisphere_rules``. A warning
    reports the labels for which these disagree.

    ``self.table`` is the resolved label table: columns ``id`` (int32),
    ``name``, ``hemi``, for the ids of the volume.
    """

    def __init__(
        self,
        name: str,
        labels: np.ndarray,
        affine: np.ndarray,
        table: pd.DataFrame,
        space: Space,
        background: int = 0,
        hemisphere_rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
        term_mapping: Optional[
            Union[pd.DataFrame, str, os.PathLike]
        ] = None,
    ) -> None:
        self.name = name
        self.space = space
        self.background = int(background)
        self.labels = _label_volume(labels)
        #: the sorted ids of the labels of the volume (with the background)
        self.present_labels = np.unique(self.labels)
        self.affine = self._checked_affine(affine)
        self._mapping = (
            pd.DataFrame({c: pd.Series(dtype=str) for c in TERM_COLUMNS})
            if term_mapping is None
            else load_term_mapping(term_mapping)
        )
        self.table = self._resolved_table(table, hemisphere_rules)
        self._extent = None

    @staticmethod
    def _checked_affine(affine) -> np.ndarray:
        affine = np.asarray(affine, dtype=float)
        if affine.shape != (4, 4):
            raise ValueError(f"the affine is 4x4, not {affine.shape}")
        if np.linalg.matrix_rank(affine[:3, :3]) < 3:
            raise ValueError("the affine is singular")
        return affine

    def _resolved_table(self, table, rules) -> pd.DataFrame:
        if not {"id", "name"} <= set(table.columns):
            raise ValueError(
                f"the label table needs the columns id and name, has "
                f"{list(table.columns)}"
            )
        table = table.reset_index(drop=True)
        ids = table["id"].astype("int64")
        if ids.duplicated().any():
            raise ValueError(
                f"duplicate ids in the label table: "
                f"{sorted(ids[ids.duplicated()].unique().tolist())}"
            )
        names = pd.Series(_strings(table["name"]), dtype=object)
        given = (
            table["hemi"].fillna("").astype(str)
            if "hemi" in table.columns
            else pd.Series("", index=table.index)
        )
        bad = sorted(set(given) - set(HEMISPHERES) - {""})
        if bad:
            raise ValueError(f"hemi must be left, right or none, not {bad}")

        present = set(self.present_labels.tolist())
        absent = ~ids.isin(present).to_numpy()
        if absent.any():
            dropped = dict(zip(ids[absent], names[absent]))
            warnings.warn(
                f"atlas {self.name!r}: labels {dropped} are in the label "
                "table but not in the volume; dropped",
                stacklevel=4,
            )
        extra = sorted(present - set(ids) - {self.background})
        if extra:
            warnings.warn(
                f"atlas {self.name!r}: labels {extra} are in the volume but "
                "not in the label table; named label_<id>",
                stacklevel=4,
            )

        mapped = (
            self._mapping[self._mapping["hemi"] != ""]
            .drop_duplicates("atlas_label_name")
            .set_index("atlas_label_name")["hemi"]
        )
        from_mapping = names.map(mapped).fillna("")
        inferred = infer_hemispheres(names, rules)
        hemi = given.where(
            given != "", from_mapping.where(from_mapping != "", inferred)
        )
        self._warn_on_disagreement(names, given, from_mapping, inferred, hemi)

        frame = pd.DataFrame({"id": ids, "name": names, "hemi": hemi})[
            ~absent
        ]
        if extra:
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame(
                        {
                            "id": extra,
                            "name": [f"label_{i}" for i in extra],
                            "hemi": "none",
                        }
                    ),
                ]
            )
        frame = frame.sort_values("id").reset_index(drop=True)
        return frame.astype({"id": np.int32})

    def _warn_on_disagreement(self, names, given, mapped, inferred, final):
        sources = pd.DataFrame(
            {
                "table": given,
                "mapping": mapped,
                "name": inferred.where(inferred != "none", ""),
            }
        )
        disagree = sources.apply(lambda r: len(set(r) - {""}) > 1, axis=1)
        for i in np.flatnonzero(disagree):
            warnings.warn(
                f"atlas {self.name!r}, label {names[i]!r}: hemi sources "
                f"disagree ({sources.iloc[i].to_dict()}); using "
                f"{final[i]!r}",
                stacklevel=4,
            )

    # Constructors for the loaders of DESIGN.md, 4.2.

    @classmethod
    def from_freesurfer(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        lut: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image (path or nibabel image) and a FreeSurfer LUT.

        Keyword arguments are those of the constructor.
        """
        volume, affine = _read_image(image)
        return cls(
            name, volume, affine, load_label_table_freesurfer_lut(lut),
            space, **kwargs,
        )

    @classmethod
    def from_bids_dseg(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        dseg_tsv: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and its BIDS ``dseg.tsv``."""
        volume, affine = _read_image(image)
        return cls(
            name, volume, affine, load_label_table_bids_dseg(dseg_tsv),
            space, **kwargs,
        )

    @classmethod
    def from_tsv(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        tsv: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and a TSV (``id``, ``name``[, ``hemi``])."""
        volume, affine = _read_image(image)
        return cls(
            name, volume, affine, load_label_table_tsv(tsv), space,
            **kwargs,
        )

    @classmethod
    def from_nilearn(
        cls, name: str, bunch: Any, space: Space, **kwargs: Any
    ) -> "Atlas":
        """From the object a nilearn atlas fetcher returns.

        Duck-typed, nilearn is not imported: ``bunch.maps`` is a path or an
        image; ``bunch.lut``, if present, is a DataFrame with ``index`` and
        ``name``; otherwise ``bunch.labels`` are the names and the id of a
        name is its position. Byte strings are decoded.
        """
        volume, affine = _read_image(bunch.maps)
        lut = getattr(bunch, "lut", None)
        if lut is not None:
            table = pd.DataFrame(
                {
                    "id": np.asarray(lut["index"], dtype="int64"),
                    "name": _strings(lut["name"]),
                }
            )
        else:
            labels = _strings(bunch.labels)
            table = pd.DataFrame({"id": range(len(labels)), "name": labels})
        return cls(name, volume, affine, table, space, **kwargs)

    # The relations of DESIGN.md, 3.

    def _column_a(self, n: int) -> np.ndarray:
        """The atlas's name, ``n`` times. A DataFrame holds it in pandas'
        string dtype (Arrow-backed from pandas 3, DESIGN.md, 7.2)."""
        return np.full(n, self.name, dtype=object)

    def label_relation(self) -> pd.DataFrame:
        """``label(a, l, name, hemi)``: columns ``a, l, name, hemi``."""
        t = self.table
        return pd.DataFrame(
            {
                "a": self._column_a(len(t)),
                "l": t["id"].to_numpy(np.int32),
                "name": t["name"].to_numpy(object),
                "hemi": t["hemi"].to_numpy(object),
            }
        )

    def label_term_relation(self) -> pd.DataFrame:
        """``label_term(a, l, term)``: columns ``a, l, term``."""
        joined = self.table[["id", "name"]].merge(
            self._mapping[["atlas_label_name", "term"]],
            left_on="name", right_on="atlas_label_name",
        )
        return pd.DataFrame(
            {
                "a": self._column_a(len(joined)),
                "l": joined["id"].to_numpy(np.int32),
                "term": joined["term"].to_numpy(object),
            }
        )

    def unmapped_labels(self) -> pd.DataFrame:
        """The labels of the atlas that have no term.

        The background is not a region and is not reported. Use it to see
        what a term mapping leaves out: a label without a term cannot be
        reached from a rule over ``label_term``.

        Returns
        -------
        pandas.DataFrame
            Columns ``id``, ``name``, ``hemi``, by id.
        """
        t = self.table
        unmapped = ~t["name"].isin(self._mapping["atlas_label_name"])
        return t[unmapped & (t["id"] != self.background)].reset_index(
            drop=True
        )

    def label_extent_relation(self) -> pd.DataFrame:
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
        if self._extent is None:
            self._extent = self._compute_label_extent()
        return self._extent.copy()

    def _compute_label_extent(self) -> pd.DataFrame:
        volume = self.labels
        if self.background != 0:
            # find_objects skips 0: swap it with the background
            volume = volume.copy()
            volume[self.labels == self.background] = 0
            volume[self.labels == 0] = self.background
        found = [
            (i, o) for i, o in enumerate(ndimage.find_objects(volume))
            if o is not None
        ]
        ids = np.array([i + 1 for i, _ in found], dtype=np.int64)
        ids[ids == self.background] = 0
        start = np.array([[s.start for s in o] for _, o in found], float)
        stop = np.array([[s.stop for s in o] for _, o in found], float)
        start, stop = start.reshape(-1, 3), stop.reshape(-1, 3)
        lo, hi = start - 0.5, stop - 0.5  # voxel edges
        bits = np.array(list(itertools.product((0, 1), repeat=3)), bool)
        corners = np.where(bits[None], hi[:, None, :], lo[:, None, :])
        world = corners @ self.affine[:3, :3].T + self.affine[:3, 3]
        n = len(ids)
        return pd.DataFrame(
            {
                "a": self._column_a(3 * n),
                "l": np.repeat(ids, 3).astype(np.int32),
                "axis": np.tile(np.arange(3), n).astype(np.int8),
                "lo": world.min(axis=1).reshape(-1).astype(np.float32),
                "hi": world.max(axis=1).reshape(-1).astype(np.float32),
            }
        )

    def check_same_space(
        self, other: Any, allow_mismatch: bool = False
    ) -> None:
        """Raise :class:`~nl_tracts.spaces.SpaceMismatchError` unless
        ``other`` (an atlas, or anything with a ``space``) is in the same
        space as this atlas; with ``allow_mismatch``, warn."""
        check_same_space(self.space, other.space, allow_mismatch)

    def content_hash(self) -> str:
        """Hash of the volume, affine, label table, background and space.

        Not of the name, nor of the term mapping, which do not change what
        a tractogram is indexed to (DESIGN.md, 7.3).
        """
        h = hashlib.sha256(b"nl_tracts.Atlas.v1")
        h.update(np.asarray(self.labels.shape, np.int64).tobytes())
        h.update(self.labels.tobytes())
        h.update(self.affine.astype("<f8").tobytes())
        for row in self.table.itertuples(index=False):
            h.update(f"\n{row.id}\t{row.name}\t{row.hemi}".encode())
        h.update(f"\nbackground={self.background}".encode())
        h.update(f"\nspace={self.space}".encode())
        return h.hexdigest()

    def __repr__(self) -> str:
        return (
            f"Atlas({self.name!r}, space={self.space!r}, "
            f"shape={self.labels.shape}, labels={len(self.table)})"
        )


class ProbabilisticAtlas:
    """Extension point for 4D (probability map) atlases. Not implemented.

    A probabilistic atlas holds, per voxel, a probability for each label.
    It will produce *probabilistic* facts ``streamline_label_p(a, x, l)``
    with the probability of the streamline traversing the label, instead
    of the boolean ``streamline_label``; it shares ``label``,
    ``label_term`` and the loaders of :class:`Atlas`, and the engine side is
    ``add_probabilistic_facts_from_tuples`` (DESIGN.md, 4.4).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "probabilistic atlases are future work (DESIGN.md, 4.4)"
        )
