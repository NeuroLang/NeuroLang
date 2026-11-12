"""
Atlases: a label image, its affine, a label table and a space.

An :class:`Atlas` is what ``.load_atlas``, ``add_atlas_set`` and the
``atlases:`` of an engine load: the regions of the relation
``name(label, region)`` of the engine, the centres of their voxels, and
what the engine ``tracts`` indexes streamlines by. This module owns the loaders
(a FreeSurfer colour LUT, a BIDS ``dseg.tsv``, a TSV, a CSV, a nilearn
fetcher, the fetchers of the atlases of the engines), the normalisation of
label names, the inference of the hemisphere of a label, the space tag and
the content hash.

It needs numpy, pandas and nibabel only, and nilearn when an atlas is
fetched, so that the core of the engine ``tracts`` may import it: it imports no
other module of NeuroLang but ``neurolang.utils.nilearn_compat``, lazily,
to fetch (``.importlinter``).
"""

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, NewType, Optional, Sequence
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd

__all__ = [
    "Atlas",
    "FETCHERS",
    "HEMISPHERES",
    "HEMISPHERE_RULES",
    "HemisphereRule",
    "SpaceMismatchError",
    "check_same_space",
    "infer_hemispheres",
    "label_colours",
    "label_name",
    "label_table",
    "label_voxel_indices",
    "load_label_table_bids_dseg",
    "load_label_table_csv",
    "load_label_table_freesurfer_lut",
    "load_label_table_tsv",
]

Path_ = Union[str, os.PathLike]

# --- spaces ----------------------------------------------------------------

#: A free-form tag naming a coordinate space, e.g. ``"MNI152NLin2009cAsym"``
#: or ``"subject:100307"``. Two tags are the same space iff they are equal:
#: NeuroLang does no registration.
Space = NewType("Space", str)


class SpaceMismatchError(ValueError):
    """Two objects, e.g. an atlas and a tractogram, are not in one space."""


def check_same_space(
    atlas_space: Space, other_space: Space, allow_mismatch: bool = False
) -> None:
    """Raise :class:`SpaceMismatchError` unless the spaces are equal.

    Parameters
    ----------
    atlas_space, other_space : Space
        The tags of the atlas and of the other object, e.g. a tractogram.
    allow_mismatch : bool
        The explicit override: if True, warn instead of raising. The caller
        asserts that the two are aligned by other means.
    """
    if atlas_space == other_space:
        return
    message = (
        f"spaces differ: {atlas_space!r} and {other_space!r}; "
        "NeuroLang does no registration"
    )
    if not allow_mismatch:
        raise SpaceMismatchError(message)
    warnings.warn(message + " (allow_mismatch=True)", stacklevel=2)


# --- label names and hemispheres ------------------------------------------

#: The hemisphere of a label.
HEMISPHERES = ("left", "right", "none")


def label_name(value: Any) -> str:
    """A raw label name as the fetched atlases name their regions: a
    string, decoded if bytes, whose ``-`` and ``_`` are spaces
    (``"L G_and_S_frontomargin"`` is ``"L G and S frontomargin"``)."""
    if isinstance(value, bytes):
        value = value.decode("utf8")
    return str(value).replace("-", " ").replace("_", " ")


@dataclass(frozen=True)
class HemisphereRule:
    """Infer the hemisphere of a label name.

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


#: The default, ordered rules; the first that matches wins: FreeSurfer
#: ``Left-``/``Right-``, ``ctx-lh-``/``ctx-rh-`` and a2009s
#: ``ctx_lh_``/``ctx_rh_`` (``ctx lh `` as :func:`label_name` writes
#: them), the suffixes ``_L``/``_R`` (AAL), ``lh.``/``rh.``,
#: ``.left``/``.right``, the prefixes ``Left ``/``Right ``
#: (Harvard-Oxford) and ``L ``/``R `` (nilearn's Destrieux), and the
#: suffixes ``-lh``/``-rh`` (Tian's subcortex), `` lh``/`` rh`` as
#: :func:`label_name` writes them, and Schaefer's ``17Networks_LH_``/
#: ``_RH_`` (``17Networks LH ``).
HEMISPHERE_RULES: Sequence[HemisphereRule] = _rules(
    (r"Left[- ].+", r"Right[- ].+"),
    (r"ctx[-_ ]lh[-_ ].+", r"ctx[-_ ]rh[-_ ].+"),
    (r".+_L", r".+_R"),
    (r"lh\..+", r"rh\..+"),
    (r".+\.left", r".+\.right"),
    (r"L .+", r"R .+"),
    (r".+[- ]lh", r".+[- ]rh"),
    (r"\d+Networks[_ ]LH[_ ].+", r"\d+Networks[_ ]RH[_ ].+"),
)


def infer_hemispheres(
    names: Iterable[str],
    rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
) -> pd.Series:
    """The hemisphere of label names, by the first rule that matches.

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


# --- label tables ---------------------------------------------------------

def load_label_table_freesurfer_lut(path: Path_) -> pd.DataFrame:
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


def load_label_table_bids_dseg(path: Path_) -> pd.DataFrame:
    """Read a BIDS ``*_dseg.tsv`` (columns ``index`` and ``name``)."""
    frame = pd.read_csv(path, sep="\t", dtype={"name": str})
    if not {"index", "name"} <= set(frame.columns):
        raise ValueError(
            "a dseg.tsv needs the columns index and name, "
            f"has {list(frame.columns)}"
        )
    frame = frame[["index", "name"]].rename(columns={"index": "id"})
    return frame.astype({"id": "int64"})


def load_label_table_tsv(path: Path_) -> pd.DataFrame:
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


def load_label_table_csv(path: Path_, **read_csv: Any) -> pd.DataFrame:
    """Read a table whose first two columns are the id and the name of
    each label, whatever their headers, with ``pandas.read_csv(path,
    **read_csv)`` (what ``.load_atlas`` reads from a URL)."""
    frame = pd.read_csv(path, **read_csv)
    if frame.shape[1] < 2:
        raise ValueError(
            "a label table needs two columns, the id and the name of each "
            f"label; has {list(frame.columns)}"
        )
    return pd.DataFrame({
        "id": frame.iloc[:, 0].to_numpy("int64"),
        "name": _strings(frame.iloc[:, 1]),
    })


def label_table(labels: Any) -> pd.DataFrame:
    """The ``id``, ``name`` table of labels as nilearn's fetchers (and
    those of :data:`FETCHERS`) give them:
    a DataFrame with ``index`` and ``name`` (a look-up table), a dict id
    -> name, a list whose position is the id, or pairs (id, name)."""
    if isinstance(labels, pd.DataFrame):
        ids, names = labels["index"], labels["name"]
    elif isinstance(labels, dict):
        ids, names = list(labels), list(labels.values())
    elif isinstance(labels, (list, tuple)) and not (
        labels and isinstance(labels[0], (list, tuple))
    ):
        ids, names = range(len(labels)), labels
    else:  # pairs, or a structured array of two columns
        pairs = [tuple(p) for p in labels]
        ids, names = [p[0] for p in pairs], [p[1] for p in pairs]
    return pd.DataFrame({
        "id": np.asarray(ids, dtype="int64"), "name": _strings(names),
    })


def label_colours(labels: Any) -> Dict[int, str]:
    """The colour, ``#rrggbb``, of each label id of a look-up table with a
    ``color`` column, as nilearn's fetchers (and those of :data:`FETCHERS`)
    give it in ``lut``; ``{}`` for labels without colours."""
    if not (isinstance(labels, pd.DataFrame) and "color" in labels.columns):
        return {}
    return {
        int(index): colour
        for index, colour in zip(labels["index"], labels["color"])
        if isinstance(colour, str) and colour.startswith("#")
    }


def _hex(red, green, blue) -> str:
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def _strings(values: Iterable[Any]) -> list:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


# --- volumes ---------------------------------------------------------------

def label_voxel_indices(data, labels):
    """
    Voxel coordinates of each label of a labelled array, in one pass.

    Parameters
    ----------
    data : numpy.ndarray
        labelled array.
    labels : iterable
        label values to look up.

    Returns
    -------
    dict
        mapping each label to an ``(n, data.ndim)`` integer array with the
        coordinates of the voxels equal to it, in the same (C) order as
        ``numpy.nonzero(data == label)``.

    """
    flat = data.ravel()
    order = np.argsort(flat, kind='stable')
    sorted_values = flat[order]
    labels = list(labels)
    # all the bounds at once, in the dtype of the data: a label of another
    # dtype would cast the whole array, at each search
    values = np.asarray(labels).astype(sorted_values.dtype, copy=False)
    starts = np.searchsorted(sorted_values, values, side='left')
    ends = np.searchsorted(sorted_values, values, side='right')
    result = {}
    for label, value, start, end in zip(labels, values, starts, ends):
        indices = order[start:end] if value == label else order[:0]
        result[label] = np.column_stack(
            np.unravel_index(indices, data.shape)
        ).reshape(-1, data.ndim)
    return result


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


def _read_image(image: Union[Path_, Any]):
    """The image (loaded if a path), its label volume and its affine."""
    if isinstance(image, (str, os.PathLike)):
        image = nib.load(image)
    # the data object, not get_fdata: label images may hold integers
    data = np.asanyarray(image.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim > 3:
        raise NotImplementedError(
            "a 4D image is a probabilistic atlas, which an Atlas is not"
        )
    return image, data, np.asarray(image.affine, dtype=float)


# --- the fetchers of the atlases of the engines ---------------------------

#: FreeSurfer's colours of its labels, those of the Destrieux atlas
#: (``aparc.a2009s``, ``ctx_lh_G_and_S_frontomargin``...) among them.
FREESURFER_LUT_URL = (
    "https://raw.githubusercontent.com/freesurfer/freesurfer/v7.4.1/"
    "distribution/FreeSurferColorLUT.txt"
)


def freesurfer_colours(data_dir: Path_) -> Dict[str, str]:
    """The colour of each label of FreeSurfer's look-up table, by name,
    fetched into *data_dir*/``freesurfer``."""
    from .utils.nilearn_compat import fetch_files

    [path] = fetch_files(
        str(Path(data_dir) / "freesurfer"),
        [("FreeSurferColorLUT.txt", FREESURFER_LUT_URL, {})], verbose=0,
    )
    table = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, usecols=range(5),
        names=["id", "name", "r", "g", "b"],
    )
    return {
        name: _hex(r, g, b)
        for name, r, g, b in zip(table["name"], table["r"], table["g"],
                                 table["b"])
    }


def _freesurfer_destrieux_name(name: str) -> str:
    """FreeSurfer's name of the Destrieux region *name* of nilearn:
    ``L G_and_S_frontomargin`` is ``ctx_lh_G_and_S_frontomargin``."""
    side, _, region = name.partition(" ")
    if not region:
        side, region = "L", name
    return f"ctx_{'rh' if side == 'R' else 'lh'}_{region}"


#: The millimetres of nilearn's Destrieux atlas to those of the MNI152
#: template: derived from FreeSurfer's fsaverage, it is 1.3 to 2.6% smaller
#: than the template and off by about a millimetre. A scale and a shift per
#: axis, fitted to the grey matter of the template by
#: ``tools/fit_destrieux_mni152.py`` (the grey-matter probability of its
#: cortex goes from 0.627 to 0.649, that of Schaefer's being 0.636).
DESTRIEUX_TO_MNI152 = np.array([
    [1.0179, 0.0, 0.0, -0.6292],
    [0.0, 1.0256, 0.0, 1.2415],
    [0.0, 0.0, 1.013, -0.8922],
    [0.0, 0.0, 0.0, 1.0],
])


def registered_image(path: Path_, transform: np.ndarray) -> str:
    """
    The image *path* with its millimetres moved by *transform*, a 4×4
    matrix, in a file of its own next to it (``x_mni152.nii.gz``), written
    once: its voxels are those of *path*, only its affine changes.
    """
    path = Path(path)
    name = path.name.split(".")[0]
    registered = path.with_name(f"{name}_mni152.nii.gz")
    if not registered.exists():
        image = nib.load(str(path))
        moved = nib.Nifti1Image(
            np.asanyarray(image.dataobj), transform @ image.affine
        )
        temporary = registered.with_name(f".{os.getpid()}.{registered.name}")
        nib.save(moved, str(temporary))
        os.replace(temporary, registered)
    return str(registered)


def fetch_destrieux(**kwargs):
    """
    The Destrieux 2009 atlas of nilearn, on the MNI152 template
    (:data:`DESTRIEUX_TO_MNI152`): ``maps``, ``labels`` and ``lut``, with
    FreeSurfer's colours of its regions in its ``color`` column when they
    can be fetched (:func:`freesurfer_colours`).
    """
    from .utils.nilearn_compat import nilearn_datasets

    atlas = nilearn_datasets().fetch_atlas_destrieux_2009(**kwargs)
    atlas.maps = registered_image(atlas.maps, DESTRIEUX_TO_MNI152)
    lut = getattr(atlas, "lut", None)
    if lut is None or not kwargs.get("data_dir"):
        return atlas
    try:
        colours = freesurfer_colours(kwargs["data_dir"])
    except Exception:  # the atlas goes without its colours
        return atlas
    atlas.lut = lut.assign(color=[
        colours.get(_freesurfer_destrieux_name(str(name)))
        for name in lut["name"]
    ])
    return atlas


def fetch_schaefer(**kwargs):
    """The Schaefer 2018 atlas of nilearn: ``maps`` and ``labels``."""
    from .utils.nilearn_compat import nilearn_datasets

    return nilearn_datasets().fetch_atlas_schaefer_2018(**kwargs)


TIAN_URL = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
    "Group-Parcellation/3T/Subcortex-Only/"
)


def fetch_tian(scale: int = 2, data_dir: str = "", **kwargs):
    """
    Fetch the Melbourne subcortex atlas (Tian et al., 2020, Nat Neurosci,
    doi:10.1038/s41593-020-00711-6), 3T, at *scale* 1 to 4 (16, 32, 50 or
    54 regions), on the 2 mm grid of the MNI152 2009c asymmetric template,
    from the authors' repository. Its label file names the region of value
    *n* on its line *n*; region names end with the hemisphere (``-lh``,
    ``-rh``).
    """
    from .utils.nilearn_compat import fetch_files

    base = f"Tian_Subcortex_S{int(scale)}_3T"
    image, labels = fetch_files(
        str(Path(data_dir) / "tian"),
        [
            (f"{base}_2009cAsym.nii.gz",
             f"{TIAN_URL}{base}_2009cAsym.nii.gz", {}),
            (f"{base}_label.txt", f"{TIAN_URL}{base}_label.txt", {}),
        ],
        verbose=0,
    )
    names = Path(labels).read_text().split()
    fetched = {"maps": image, "labels": ["Background", *names]}
    try:
        colours = _tian_colours(scale, data_dir)
    except Exception:  # the atlas goes without its colours
        return fetched
    fetched["lut"] = pd.DataFrame({
        "index": range(len(names) + 1), "name": ["Background", *names],
        "color": [None] + [colours.get(name) for name in names],
    })
    return fetched


#: The label files of the authors with the colours of the regions: those of
#: the atlases of the cortex and the subcortex together.
TIAN_COLOURS_URL = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
    "Group-Parcellation/3T/Cortex-Subcortex/"
)


def _tian_colours(scale: int, data_dir: str) -> Dict[str, str]:
    """The colours of the regions of the Melbourne subcortex atlas at
    *scale*, by name, from the label file of the authors (in Workbench's
    format: a name, then its value and its red, green, blue and alpha)."""
    from .utils.nilearn_compat import fetch_files

    base = (
        f"Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_"
        f"S{int(scale)}_label.txt"
    )
    [path] = fetch_files(
        str(Path(data_dir) / "tian"), [(base, TIAN_COLOURS_URL + base, {})],
        verbose=0,
    )
    lines = Path(path).read_text().split("\n")
    return {
        name.strip(): _hex(*values.split()[1:4])
        for name, values in zip(lines[::2], lines[1::2])
        if len(values.split()) >= 4
    }


#: The fetchers of the deterministic atlases of the engines, by the kind of
#: ``.load_atlas``: each returns ``maps`` (a path) and ``labels``, as
#: nilearn's fetchers.
FETCHERS = {
    "destrieux": fetch_destrieux,
    "schaefer": fetch_schaefer,
    "tian": fetch_tian,
}


# --- the atlas -------------------------------------------------------------

class Atlas:
    """A deterministic atlas: label volume + affine + label table + space.

    Parameters
    ----------
    name : str
        The atlas's identifier, unique in an engine.
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
        the table (other than the background) are no region, with a
        warning (``unlisted_labels``).
    space : Space
        The space tag, ``"unspecified"`` by default.
    background : int
        The label that means "no region" (default 0): it has no region
        and no point, but keeps its row of the table if it has one.
    hemisphere_rules : Sequence[HemisphereRule]
        Rules to infer the hemisphere from the name, where the table does
        not say it. Defaults to :data:`HEMISPHERE_RULES`; ``()`` infers
        nothing.
    kind : str, optional
        How it was loaded: the ``kind`` of ``.load_atlas`` (``freesurfer``,
        ``destrieux``...; ``nilearn:<fetcher>`` for a nilearn fetcher).
    image : optional
        The nibabel image the volume was read from, kept as
        :attr:`image`.

    Notes
    -----
    ``self.table`` is the resolved label table: columns ``id`` (int32),
    ``name``, ``hemi``, by id, for the ids of the table in the volume. The
    hemisphere of a label is the ``hemi`` of the table, if not empty, else
    the inference by ``hemisphere_rules`` (:meth:`hemispheres`).
    """

    def __init__(
        self,
        name: str,
        labels: np.ndarray,
        affine: np.ndarray,
        table: pd.DataFrame,
        space: Space = "unspecified",
        background: int = 0,
        hemisphere_rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
        kind: Optional[str] = None,
        image: Any = None,
    ) -> None:
        self.name = name
        self.space = space
        self.kind = kind
        self.background = int(background)
        self.labels = _label_volume(labels)
        #: the sorted ids of the labels of the volume (with the background)
        self.present_labels = np.unique(self.labels)
        self.affine = self._checked_affine(affine)
        self._image = image
        self.table = self._resolved_table(table, hemisphere_rules)

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
        #: the ids of the volume that the table does not name, but the
        #: background: they are no region
        self.unlisted_labels = sorted(
            present - set(ids) - {self.background}
        )
        if self.unlisted_labels:
            warnings.warn(
                f"atlas {self.name!r}: labels {self.unlisted_labels} are in "
                "the volume but not in the label table; they are no region",
                stacklevel=4,
            )
        frame = pd.DataFrame({
            "id": ids.to_numpy(np.int32), "name": names, "given": given,
            "inferred": infer_hemispheres(names, rules),
        })[~absent].sort_values("id", kind="stable").reset_index(drop=True)
        self._given = frame.pop("given")
        self._inferred = frame.pop("inferred")
        self.table = frame
        frame["hemi"] = self.hemispheres()
        return frame

    def hemispheres(
        self, hints: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """The hemisphere of each label of :attr:`table`, in its order.

        It is, in order of precedence: the ``hemi`` of the label table, if
        not empty; the hint for its name, if any and not empty; the
        inference by the hemisphere rules. A warning names each label for
        which two of these disagree (the inference counting only when it
        is not ``none``); with ``hints``, only those where a hint is one of
        them.

        Parameters
        ----------
        hints : dict, optional
            A label name -> ``left``, ``right``, ``none`` or ``""``, e.g.
            the ``hemi`` of a term mapping of the engine ``tracts``.
        """
        names = self.table["name"]
        from_hints = (
            names.map(hints).fillna("").astype(str)
            if hints else pd.Series("", index=names.index, dtype=object)
        )
        given, inferred = self._given, self._inferred
        hemi = given.where(
            given != "", from_hints.where(from_hints != "", inferred)
        )
        sources = pd.DataFrame({
            "table": given,
            "hints": from_hints,
            "name": inferred.where(inferred != "none", ""),
        })
        disagree = sources.apply(lambda r: len(set(r) - {""}) > 1, axis=1)
        if hints:
            disagree &= from_hints != ""
        for i in np.flatnonzero(disagree):
            warnings.warn(
                f"atlas {self.name!r}, label {names.iloc[i]!r}: hemi "
                f"sources disagree ({sources.iloc[i].to_dict()}); using "
                f"{hemi.iloc[i]!r}",
                stacklevel=3,
            )
        return hemi

    # The loaders.

    @classmethod
    def from_freesurfer(
        cls, name: str, image: Union[Path_, Any], lut: Path_,
        space: Space = "unspecified", **kwargs: Any,
    ) -> "Atlas":
        """From a label image (path or nibabel image) and a FreeSurfer LUT.

        Keyword arguments are those of the constructor.
        """
        return cls._from_image(
            name, image, load_label_table_freesurfer_lut(lut), space,
            kind="freesurfer", **kwargs,
        )

    @classmethod
    def from_bids_dseg(
        cls, name: str, image: Union[Path_, Any], dseg_tsv: Path_,
        space: Space = "unspecified", **kwargs: Any,
    ) -> "Atlas":
        """From a label image and its BIDS ``dseg.tsv``."""
        return cls._from_image(
            name, image, load_label_table_bids_dseg(dseg_tsv), space,
            kind="bids", **kwargs,
        )

    @classmethod
    def from_tsv(
        cls, name: str, image: Union[Path_, Any], tsv: Path_,
        space: Space = "unspecified", **kwargs: Any,
    ) -> "Atlas":
        """From a label image and a TSV (``id``, ``name``[, ``hemi``])."""
        return cls._from_image(
            name, image, load_label_table_tsv(tsv), space, kind="tsv",
            **kwargs,
        )

    @classmethod
    def from_csv(
        cls, name: str, image: Union[Path_, Any], labels_csv: Path_,
        space: Space = "unspecified", read_csv: Optional[dict] = None,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and a table whose first two columns are the
        id and the name of each label, read by ``pandas.read_csv`` with
        the options ``read_csv`` (:func:`load_label_table_csv`)."""
        return cls._from_image(
            name, image, load_label_table_csv(labels_csv, **(read_csv or {})),
            space, kind="csv", **kwargs,
        )

    @classmethod
    def from_labels(
        cls, name: str, image: Union[Path_, Any], labels: Any,
        space: Space = "unspecified", normalise: bool = False,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and its labels as nilearn's fetchers give
        them: a look-up table (DataFrame with ``index`` and ``name``), a
        dict id -> name, a list whose position is the id (with a leading
        ``Background``), or pairs (id, name). The row of the background,
        if any, is dropped. With ``normalise``, the names are those of
        :func:`label_name`, as the engines name the regions of the atlases
        they fetch."""
        table = label_table(labels)
        table = table[table["id"] != kwargs.get("background", 0)]
        if normalise:
            table["name"] = table["name"].map(label_name)
        return cls._from_image(name, image, table, space, **kwargs)

    @classmethod
    def from_nilearn(
        cls, name: str, fetcher: str, space: Space = "MNI152",
        background: int = 0, hemisphere_rules=HEMISPHERE_RULES,
        **options: Any,
    ) -> "Atlas":
        """The atlas ``nilearn.datasets.fetch_atlas_<fetcher>(**options)``
        fetches (``destrieux_2009``, ``harvard_oxford`` with
        ``atlas_name=...``, ``schaefer_2018`` with ``n_rois=...``), its
        label names normalised (:func:`label_name`), as the engines name
        those of the atlases they fetch. Its kind is ``nilearn:<fetcher>``.
        """
        from .utils.nilearn_compat import nilearn_datasets

        try:
            datasets = nilearn_datasets()
        except ImportError as e:
            raise ImportError(
                'an atlas of kind="nilearn" needs nilearn (pip install '
                'nilearn)'
            ) from e
        fetch = getattr(datasets, f"fetch_atlas_{fetcher}", None)
        if fetch is None:
            raise ValueError(f"nilearn has no fetch_atlas_{fetcher}")
        bunch = fetch(**options)
        lut = getattr(bunch, "lut", None)
        return cls.from_labels(
            name, bunch.maps, bunch.labels if lut is None else lut, space,
            normalise=True, background=background,
            hemisphere_rules=hemisphere_rules, kind=f"nilearn:{fetcher}",
        )

    @classmethod
    def fetch(
        cls, name: str, kind: str, data_dir: Path_,
        space: Space = "MNI152", **options: Any,
    ) -> "Atlas":
        """The atlas of an engine of kind ``destrieux``, ``schaefer`` or
        ``tian`` (:data:`FETCHERS`), fetched into ``data_dir`` with the
        options of its fetcher (``n_rois=100`` of ``schaefer``, ``scale=2``
        of ``tian``), its label names normalised (:func:`label_name`)."""
        if kind not in FETCHERS:
            raise ValueError(
                f"no fetcher of atlases {kind!r}; they are "
                f"{', '.join(FETCHERS)}"
            )
        fetched = FETCHERS[kind](data_dir=str(data_dir), **options)
        return cls.from_labels(
            name, fetched["maps"], fetched["labels"], space,
            normalise=True, kind=kind,
        )

    @classmethod
    def _from_image(cls, name, image, table, space, **kwargs) -> "Atlas":
        image, volume, affine = _read_image(image)
        return cls(name, volume, affine, table, space, image=image, **kwargs)

    # What the engine makes of it.

    @property
    def shape(self):
        """The shape of the volume."""
        return self.labels.shape

    @property
    def image(self):
        """The nibabel image of the atlas: the one it was read from, if
        any, else one of its volume and affine."""
        if self._image is None:
            self._image = nib.Nifti1Image(self.labels, self.affine)
        return self._image

    def region_names(self) -> Dict[int, str]:
        """The name of each region, by id: the labels of the table but the
        background."""
        regions = self.table[self.table["id"] != self.background]
        return dict(zip(regions["id"].tolist(), regions["name"].tolist()))

    def label_values(self) -> Dict[str, List[int]]:
        """The ids of each region, by name (a name may have several)."""
        values = {}
        for value, name in self.region_names().items():
            values.setdefault(name, []).append(value)
        return values

    def label_voxels(self) -> Dict[int, np.ndarray]:
        """The voxel indices (ijk) of each region, by id, found in one
        pass (:func:`label_voxel_indices`)."""
        return label_voxel_indices(self.labels, list(self.region_names()))

    def points(self) -> pd.DataFrame:
        """The centre of every voxel of a region, in mm through the
        affine, and its name: columns ``x``, ``y``, ``z``, ``label``, in
        the (C) order of the voxels."""
        names = self.region_names()
        voxels = np.argwhere(np.isin(self.labels, list(names)))
        points = nib.affines.apply_affine(self.affine, voxels)
        return pd.DataFrame({
            "x": points[:, 0], "y": points[:, 1], "z": points[:, 2],
            "label": [
                names[v] for v in self.labels[tuple(voxels.T)].tolist()
            ],
        })

    def check_same_space(
        self, other: Any, allow_mismatch: bool = False
    ) -> None:
        """Raise :class:`SpaceMismatchError` unless ``other`` (an atlas, or
        anything with a ``space``) is in the same space as this atlas; with
        ``allow_mismatch``, warn."""
        check_same_space(self.space, other.space, allow_mismatch)

    def content_hash(self) -> str:
        """Hash of the volume, affine, label table, background and space,
        not of the name nor of the kind."""
        h = hashlib.sha256(b"neurolang.Atlas.v1")
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
