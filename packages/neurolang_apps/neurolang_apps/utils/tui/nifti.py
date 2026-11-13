"""
Saving and viewing of query results in the interactive REPL.

``.save PATH`` writes the last result as a table, or as a NIfTI image
when PATH ends with ``.nii`` or ``.nii.gz``; ``.view PATH`` shows a NIfTI
image, or the last result, in a browser: an HTML page of
:mod:`~neurolang_apps.utils.tui.brain_viewer`, whose controls change the
threshold, the colormap and its symmetry live (see :data:`VIEW_HELP`).

A NIfTI image is written on a grid of the engine, with its shape and
affine: that of the engine's template, the MNI image its voxel indices
refer to (the ``template`` of the engine
:func:`~neurolang_apps.utils.engine_registry.build_engine` returns), or that
of an atlas whose voxel indices a sort names with ``grid`` (the engine's
``grids``). The sorts of the columns of the result, from the query that
gave it (:mod:`~.provenance`), else their names, tell which columns are
the voxels (:func:`point_columns`) and :func:`value_column` which column
holds their value; see :data:`SAVE_HELP`.

``.view`` of a result of streamline ids draws the streamlines, with an
engine that has them: one with the method ``streamline_points(ids)``,
returning a list of (n_i, 3) float arrays, the points of each id in
millimetres (RAS+), and optionally ``streamline_background()``, a
nibabel image in their space (a subject's anatomy or labels) to draw
them on, the MNI152 T1 otherwise. The column of the ids is the one of
the sort :data:`STREAMLINE_SORT` in the query, else the one column of
integers of unknown sort (:func:`streamline_column`), when the result
has no voxel columns: ``ans(x, y, z)`` stays millimetres.
"""

import gzip
import os
import shlex
import tempfile
import warnings
import webbrowser
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang_apps.utils import engine_registry

from . import brain_viewer
from .commands import HELP_TOPICS, command
from .provenance import COMPUTED

SAVE_USAGE = ".save PATH [--value COLUMN] [--voxel I J K | --mm X Y Z]"

SAVE_HELP = """\
[bold cyan].save PATH[/bold cyan] writes the last result (a file per obtain block).

  out.csv, out.tsv      a table, comma- or tab-separated, gzipped for a
  out.csv.gz            .gz path; any other path is CSV
  out.nii, out.nii.gz   a NIfTI image on a grid of the engine (its shape
                        and affine), that of its template by default

The voxels of a NIfTI image are the columns of the result of a sort of
the engine declaring a space (engines.yaml):
  space: voxel          grid indices (i, j, k), written as they are on
                        the grid the sort names (e.g. grid: schaefer, an
                        atlas's), by default the template's
  space: mm             millimetres (x, y, z), converted to the nearest
                        voxel with the inverse of the template's affine
A column has the sort of the argument of the predicate its variable is
in, in the last query: ans(x, y, z) :- voxel(x, y, z) holds indices. A
column of unknown sort, of a relation without declared arguments, is of
the sort it is named as (i, j, k or x, y, z). --voxel I J K (indices of
the template's grid) or --mm X Y Z (millimetres) give the columns.

The value of a voxel is that of the column given with --value COLUMN,
else of the probability column, else of the one numeric column that is
neither a voxel column nor an entity's (a column of a sort that is not
of type float, such as study_id); otherwise the image is a mask, 1 at
each voxel of the result. Rows of the same voxel keep their maximum.
A mask is written as uint8, integers as int32 and other values as
float32; rows outside the grid are skipped. A path may hold spaces.

A result on the regions of an atlas, a column of a sort naming its
atlas (atlas: destrieux), such as ans(l, p) :- PROB[...], paints each
region with the value of its rows, on the atlas's grid; labels are
looked up in the other atlases of the engine too, and regions of several
atlases are written on the template's grid. A result of pairs of regions
or points is a network, which .view shows; .save writes it as CSV."""


HELP_TOPICS["save"] = SAVE_HELP


class SaveError(ValueError):
    """Raised when a result cannot be saved as asked."""


# The space of the columns of region labels of an atlas.
LABEL_SPACE = "label"


class PointColumns(NamedTuple):
    """
    The columns of a result locating its voxels, their space and grid.

    The *space* is ``voxel`` for the indices of the *grid*, the name of a
    grid of the engine (the template's by default), ``mm`` for
    millimetres in the template's space, written on its grid, and
    :data:`LABEL_SPACE` for the labels of the regions of the atlas of the
    *grid*, whose voxels are those of each region.
    """

    columns: Tuple[str, ...]
    space: str
    grid: str = engine_registry.TEMPLATE_GRID


class SaveRequest(NamedTuple):
    """
    What ``.save`` writes: its *path* and, for NIfTI, the columns to use.

    *value* names the value column and *points* the voxel columns; when
    None, they follow from the result and the engine's sorts. *fill* is
    the value of the voxels without one: not in the result, or NaN.
    """

    path: Path
    value: Optional[str] = None
    points: Optional[PointColumns] = None
    fill: float = 0.0


class SavedImage(NamedTuple):
    """
    A result converted to a NIfTI *image*, and how.

    *value* is the column of the voxel values, None for a mask; *voxels*
    the number of voxels written and *outside* the number of rows outside
    the *grid*, skipped. For a result on the regions of an atlas,
    *regions* is the number of regions painted and *outside* that of the
    labels of no region, skipped; None otherwise.
    """

    image: nib.Nifti1Image
    value: Optional[str]
    voxels: int
    outside: int
    grid: str = engine_registry.TEMPLATE_GRID
    regions: Optional[int] = None


# The options of .save and the number of columns each takes.
_SAVE_OPTIONS = {"--value": 1, "--voxel": 3, "--mm": 3}


# How each option is written, for the messages.
_OPTION_FORMS = {
    "--value": "--value COLUMN", "--voxel": "--voxel I J K",
    "--mm": "--mm X Y Z", "--threshold": "--threshold T",
    "--cmap": "--cmap NAME", "--symmetric": "--symmetric",
    "--one-sided": "--one-sided", "--fill": "--fill V",
}


def split_args(arg: str, widths: dict, usage: str) -> tuple:
    """
    Return the paths and the options of the command arguments *arg*.

    *widths* gives the number of values each option takes; the options
    are returned as a dict of their values. Consecutive words that are
    not options are one path, joined by a space, so that a path with
    spaces need not be quoted. A :class:`SaveError` tells why *arg*
    does not follow *usage*.
    """
    try:
        tokens = shlex.split(arg)
    except ValueError as exc:
        raise SaveError(f"{exc}. Usage: {usage}") from exc
    paths, options, joining = [], {}, False
    while tokens:
        token = tokens.pop(0)
        if token.startswith("--"):
            options[token] = _option_values(token, tokens, widths)
        elif joining:
            paths[-1] = f"{paths[-1]} {token}"
        else:
            paths.append(token)
        joining = not token.startswith("--")
    return paths, options


def parse_save_args(arg: str) -> SaveRequest:
    """
    Return the request of the arguments *arg* of ``.save``.

    They are a path, quoted or not if it holds spaces, and the options
    of :data:`SAVE_USAGE`; a :class:`SaveError` tells why they are not.
    """
    paths, options = split_args(arg, _SAVE_OPTIONS, SAVE_USAGE)
    if len(paths) != 1:
        raise SaveError(f"Usage: {SAVE_USAGE}")
    return _save_request(Path(paths[0]).expanduser(), options)


def _option_values(option: str, tokens: list, widths: dict) -> list:
    """Remove the values of *option* from the head of *tokens*."""
    width = widths.get(option)
    if width is None:
        forms = [_OPTION_FORMS[name] for name in widths]
        raise SaveError(
            f"Unknown option {option}; the options are "
            f"{', '.join(forms[:-1])} and {forms[-1]}."
        )
    values = tokens[:width]
    if len(values) < width or any(v.startswith("--") for v in values):
        unit = "column" if option in _SAVE_OPTIONS else "value"
        count = f"1 {unit}" if width == 1 else f"{width} {unit}s"
        raise SaveError(f"{option} takes {count}.")
    del tokens[:width]
    return values


def _save_request(path: Path, options: dict) -> SaveRequest:
    """Return the request to save *path* with the parsed *options*."""
    if options and not is_nifti_path(path):
        raise SaveError(
            f"{', '.join(options)}: options of NIfTI files (.nii, .nii.gz)."
        )
    spaces = [space for space in ("voxel", "mm") if f"--{space}" in options]
    if len(spaces) > 1:
        raise SaveError("Give --voxel or --mm, not both.")
    points = [
        PointColumns(tuple(options[f"--{space}"]), space) for space in spaces
    ]
    value = options.get("--value", [None])[0]
    return SaveRequest(path, value, points[0] if points else None)


def _column(df: pd.DataFrame, name: str):
    """Return the column of *df* named *name*, or raise a SaveError."""
    found = {str(c): c for c in df.columns}.get(name)
    if found is None:
        columns = ", ".join(str(c) for c in df.columns)
        raise SaveError(
            f"The result has no column {name!r}; its columns are {columns}."
        )
    return found


def _sort_points(sort, columns: tuple) -> PointColumns:
    """Return the *columns* holding the points of *sort*, on its grid."""
    if sort.atlas:
        return PointColumns(tuple(columns), LABEL_SPACE, sort.atlas)
    return PointColumns(
        tuple(columns), sort.space,
        sort.grid or engine_registry.TEMPLATE_GRID,
    )


def _spatial_sorts(sorts: dict) -> list:
    """
    Return the *sorts* declaring a space, those of voxels first, then
    those labelling the regions of an atlas.
    """
    return [
        sort for space in engine_registry.SORT_SPACES
        for sort in sorts.values() if sort.space == space
    ] + [sort for sort in sorts.values() if sort.atlas]


def node_columns(
    df: pd.DataFrame, sorts: dict, provenance: Optional[tuple]
) -> list:
    """
    Return every group of columns of *df* locating points or regions.

    A group is the columns of a sort of the *sorts* declaring a space,
    or labelling the regions of an atlas, in the *provenance* of *df*:
    ``ans(l1, l2, p)`` of two columns of region labels has two groups,
    and two mm points ``(x, y, z, x2, y2, z2)`` two. Columns of unknown
    sort named as those of a sort declaring a space, with a suffix of
    their own, are a group too: ``x1, y1, z1`` and ``x2, y2, z2``. The
    groups are in the order of their first column.
    """
    if provenance is None or len(provenance) != len(df.columns):
        provenance = (None,) * len(df.columns)
    groups = _named_groups(df, sorts, provenance)
    for sort in _spatial_sorts(sorts):
        positions = [
            [
                (n, column) for n, (column, column_sort) in enumerate(
                    zip(df.columns, provenance)
                )
                if column_sort is not None
                and tuple(column_sort) == (sort.name, position)
            ]
            for position in range(len(sort.columns))
        ]
        for group in zip(*positions):
            groups.append((
                group[0][0],
                _sort_points(sort, tuple(column for _, column in group)),
            ))
    unique = {}
    for n, points in groups:
        unique.setdefault(points.columns, (n, points))
    return [
        points for _, points in sorted(unique.values(), key=lambda g: g[0])
    ]


def _named_groups(df: pd.DataFrame, sorts: dict, provenance) -> list:
    """
    Return the groups of the columns of *df* of unknown sort named as
    those of a sort of the *sorts* declaring a space, with a suffix,
    as (index of the first column, :class:`PointColumns`).
    """
    names = {
        str(c): (n, c) for n, (c, p) in enumerate(zip(df.columns, provenance))
        if p is None
    }
    groups = []
    for sort in _spatial_sorts(sorts):
        if not sort.space:
            continue
        first = sort.columns[0]
        for name, (n, _) in names.items():
            suffix = name[len(first):]
            group = [names.get(c + suffix) for c in sort.columns]
            if name.startswith(first) and None not in group:
                groups.append((n, _sort_points(
                    sort, tuple(column for _, column in group)
                )))
    return groups


def network_nodes(
    df: pd.DataFrame, sorts: dict, provenance: Optional[tuple]
) -> Optional[list]:
    """
    Return the two groups of columns of the nodes of a network, or None.

    *df* is a network when it has two groups of columns of region labels
    (:func:`node_columns`), pairs of regions, or two of points of the
    same sort, pairs of points: an edge per row between its two nodes.
    """
    groups = node_columns(df, sorts, provenance)
    regions = [g for g in groups if g.space == LABEL_SPACE]
    if len(regions) == 2:
        return regions
    points = [g for g in groups if g.space != LABEL_SPACE]
    if len(points) == 2 and points[0][1:] == points[1][1:]:
        return points
    return None


def _provenance_points(columns, provenance, sorts: dict):
    """
    Return the *columns* holding the points of a sort, or None.

    *provenance* holds the sort of each column (see
    :class:`~.provenance.ColumnSort`); the columns are those of the
    first sort declaring a space that the columns span whole.
    """
    found = {}
    for column, column_sort in zip(columns, provenance):
        if column_sort is not None:
            found.setdefault(tuple(column_sort), column)
    for sort in _spatial_sorts(sorts):
        spanned = [
            found.get((sort.name, n)) for n in range(len(sort.columns))
        ]
        if None not in spanned:
            return _sort_points(sort, spanned)
    return None


def point_columns(
    df: pd.DataFrame, sorts: dict, provenance: Optional[tuple] = None
) -> PointColumns:
    """
    Return the columns of *df* locating its voxels, from the *sorts*.

    They are the columns of a sort declaring a space (see
    :class:`~neurolang_apps.utils.engine_registry.Sort`), those of a sort of
    voxel indices first, then of millimetres: the columns of that sort
    in the *provenance* of *df*, the sort of each column as the query
    tells it, else the columns of unknown sort named as those of the
    sort.
    """
    if provenance is None or len(provenance) != len(df.columns):
        provenance = (None,) * len(df.columns)
    points = _provenance_points(df.columns, provenance, sorts)
    if points is not None:
        return points
    names = {c for c, p in zip(df.columns, provenance) if p is None}
    for sort in _spatial_sorts(sorts):
        if sort.space and set(sort.columns) <= names:
            return _sort_points(sort, sort.columns)
    raise SaveError(_no_points_message(df, sorts))


def _no_points_message(df: pd.DataFrame, sorts: dict) -> str:
    """Return the error on *df*, whose voxel columns cannot be found."""
    known = [
        f"{sort.name} ({', '.join(sort.columns)}, {_space_text(sort)})"
        for sort in sorts.values() if sort.space
    ]
    atlases = [sort.name for sort in sorts.values() if sort.atlas]
    columns = ", ".join(str(c) for c in df.columns)
    hint = (
        f"name them as those of {' or '.join(known)}, or "
        if known else "the engine declares no sort of voxels; "
    )
    regions = (
        f" A column of region labels is one of the sort "
        f"{' or '.join(atlases)} in the query." if atlases else ""
    )
    return (
        f"The result has no voxel columns (its columns are {columns}): "
        f"{hint}give them with --voxel I J K or --mm X Y Z.{regions}"
    )


def _space_text(sort) -> str:
    """Return what the columns of the *sort* of points are, in words."""
    if sort.space == "mm":
        return "millimetres"
    return f"indices of the {sort.grid} grid" if sort.grid else "indices"


def voxel_indices(
    df: pd.DataFrame, points: PointColumns, affine: np.ndarray
) -> np.ndarray:
    """
    Return the voxel indices of the rows of *df*, an array of shape (n, 3).

    Millimetre coordinates are mapped to the nearest voxel with the
    inverse of *affine*; voxel indices must be integers.
    """
    coordinates = df[list(points.columns)].to_numpy(dtype=float)
    if points.space == "mm":
        coordinates = np.rint(
            nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
        )
    elif not np.array_equal(coordinates, np.rint(coordinates)):
        raise SaveError(
            f"The voxel columns {', '.join(points.columns)} hold values "
            "that are not integers; give millimetres with --mm X Y Z."
        )
    return coordinates.astype(int).reshape(-1, 3)


_PROBABILITY_COLUMNS = ("probability", "prob")


def value_column(
    df: pd.DataFrame, points: PointColumns, sorts: dict,
    explicit: Optional[str] = None, provenance: Optional[tuple] = None,
):
    """
    Return the column of *df* holding the value of its voxels, or None.

    It is the *explicit* column if given, else the probability column,
    else the one numeric column that is neither of the *points* nor of an
    entity, a sort of the *sorts* that is not of type float (such as a
    study): the sort of the column in the *provenance* of *df*, as the
    query tells it, else the sort it is named as. None stands for a mask.
    """
    if explicit is not None:
        return _numeric_column(df, explicit)
    others = [c for c in df.columns if c not in points.columns]
    probability = [
        c for c in others if str(c).lower() in _PROBABILITY_COLUMNS
    ]
    if probability:
        return probability[0]
    measures = _measure_columns(df, others, sorts, provenance)
    return measures[0] if len(measures) == 1 else None


def _column_sort(df: pd.DataFrame, column, sorts: dict, provenance):
    """
    Return the sort of *column* of *df* in its *provenance*, or None.

    The sort is one of *sorts*; None when the provenance of *df* is
    unknown, or that of the column.
    """
    if provenance is None or len(provenance) != len(df.columns):
        return None
    column_sort = provenance[list(df.columns).index(column)]
    return sorts.get(column_sort.sort) if column_sort else None


def _is_computed(df: pd.DataFrame, column, provenance) -> bool:
    """Whether the query computes *column* of *df* (see :data:`COMPUTED`)."""
    if provenance is None or len(provenance) != len(df.columns):
        return False
    column_sort = provenance[list(df.columns).index(column)]
    return column_sort is not None and column_sort.sort == COMPUTED


def _measure_columns(
    df: pd.DataFrame, columns: list, sorts: dict, provenance=None
) -> list:
    """
    Return the numeric *columns* of *df* that are not of an entity.

    The columns of an entity are those of the *sorts* not of type float,
    such as ``study_id``: of such a sort in the *provenance* of *df*, or
    named as its columns when their sort is unknown.
    """
    entities = {
        column for sort in sorts.values() if sort.type != "float"
        for column in sort.columns
    }
    measures = []
    for c in columns:
        sort = _column_sort(df, c, sorts, provenance)
        if _is_computed(df, c, provenance):
            entity = False
        else:
            entity = c in entities if sort is None else sort.type != "float"
        if not entity and pd.api.types.is_numeric_dtype(df[c]):
            measures.append(c)
    return measures


def _numeric_column(df: pd.DataFrame, name: str):
    """Return the numeric column *name* of *df*, or raise a SaveError."""
    column = _column(df, name)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise SaveError(f"The column {name!r} is not numeric.")
    return column


def _max_per_voxel(indices: np.ndarray, values: np.ndarray) -> pd.Series:
    """Return the maximum of the *values* of each voxel of *indices*."""
    frame = pd.DataFrame(indices, columns=["i", "j", "k"])
    frame["value"] = values
    return frame.groupby(["i", "j", "k"])["value"].max()


def _image_dtype(values: np.ndarray):
    """
    Return the type of the voxels of an image of *values*.

    A mask is uint8, integers are int32 when they fit, and other values
    float32.
    """
    if values.dtype.kind == "b" or values.dtype == np.uint8:
        return np.uint8
    info = np.iinfo(np.int32)
    if values.dtype.kind in "iu" and (
        values.size == 0
        or (values.min() >= info.min and values.max() <= info.max)
    ):
        return np.int32
    return np.float32


def _volume(
    per_voxel: pd.Series, shape: tuple, fill: float = 0.0
) -> np.ndarray:
    """
    Return the volume of *shape* holding the values of *per_voxel*.

    The other voxels, and those whose value is NaN, are *fill*.
    """
    values = per_voxel.to_numpy()
    dtype = _image_dtype(values)
    if dtype != np.float32 and float(fill) != int(fill):
        dtype = np.float32
    data = np.full(shape, fill, dtype=dtype)
    indices = per_voxel.index.to_frame().to_numpy(dtype=int).reshape(-1, 3)
    data[tuple(indices.T)] = values
    if data.dtype.kind == "f":
        data[np.isnan(data)] = fill
    return data


def result_to_nifti(
    df: pd.DataFrame, template, sorts: dict,
    request: SaveRequest = SaveRequest(Path()),
    provenance: Optional[tuple] = None,
) -> SavedImage:
    """
    Convert the result *df* into a NIfTI image on the *template*'s grid.

    The voxels and their value follow from the *request*, else from the
    *sorts* and the *provenance* of *df*, the sort of each of its
    columns (:func:`point_columns`, :func:`value_column`); rows of the
    same voxel keep their maximum and rows outside the grid are skipped.
    The *template* is the image of the grid of the voxels.
    """
    points = request.points or point_columns(df, sorts, provenance)
    for name in points.columns:
        _column(df, name)
    value = value_column(df, points, sorts, request.value, provenance)
    indices = voxel_indices(df, points, template.affine)
    shape = tuple(template.shape[:3])
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    values = (
        df[value].to_numpy()[inside] if value is not None
        else np.ones(inside.sum(), dtype=np.uint8)
    )
    per_voxel = _max_per_voxel(indices[inside], values)
    fill = request.fill if value is not None else 0
    image = nib.Nifti1Image(
        _volume(per_voxel, shape, fill), template.affine
    )
    return SavedImage(
        image, value, len(per_voxel), int(len(df) - inside.sum()),
        points.grid,
    )


def _region_atlases(nl, sorts: dict, atlas: str) -> list:
    """
    Return the atlases of *nl* to look the labels of a column up in.

    They are the *atlas* of the column's sort, then the others the
    *sorts* label the regions of: a rule may join the regions of several
    atlases, ``region(l) :- schaefer(l, r)`` and ``region(l) :- tian(l,
    r)``, whose column is of the sort of the first.
    """
    atlases = [atlas] + [
        sort.atlas for sort in sorts.values()
        if sort.atlas and sort.atlas != atlas
    ]
    return [
        name for n, name in enumerate(atlases)
        if name not in atlases[:n]
        and name in (getattr(nl, "grids", None) or {})
    ]


def region_lookup(nl, atlases: list) -> dict:
    """
    Return the atlas and the values labelling each region, by name.

    The regions are those of the *atlases* of *nl*, in the image of the
    grid of each (see
    :func:`~neurolang_apps.utils.engine_registry.atlas_label_values`); a name
    of several atlases is that of the first.
    """
    lookup = {}
    for atlas in atlases:
        for label, values in engine_registry.atlas_label_values(
            nl, atlas
        ).items():
            lookup.setdefault(label, (atlas, values))
    return lookup


def _label_volume(nl, atlas: str, target=None) -> np.ndarray:
    """
    Return the labels of the image of the grid of the *atlas* of *nl*.

    On the grid of the image *target* if given, each voxel taking the
    label of the atlas's voxel nearest to it.
    """
    image = engine_grid(nl, atlas)
    data = np.asanyarray(image.dataobj).astype(np.int64)
    data = data.reshape(data.shape[:3] + (-1,))[..., 0]
    if target is None:
        return data
    from scipy import ndimage

    mapping = np.linalg.inv(image.affine) @ target.affine
    return ndimage.affine_transform(
        data, mapping[:3, :3], mapping[:3, 3],
        output_shape=tuple(target.shape[:3]), order=0, mode="constant",
        cval=0,
    )


def _region_values(
    df: pd.DataFrame, column, value
) -> pd.Series:
    """Return the maximum *value* of each label of *column* of *df*."""
    values = (
        df[value] if value is not None
        else pd.Series(np.ones(len(df), dtype=np.uint8), index=df.index)
    )
    frame = pd.DataFrame({"label": df[column].astype(str), "value": values})
    return frame.groupby("label")["value"].max()


def regions_to_nifti(
    nl, df: pd.DataFrame, sorts: dict, request: SaveRequest,
    provenance: Optional[tuple] = None,
) -> SavedImage:
    """
    Convert the result *df* on regions into a NIfTI image of *nl*.

    The column of the region labels is that of the *request*'s points,
    of :data:`LABEL_SPACE`, and the value of each region that of its
    rows, the maximum of them (:func:`value_column`), 1 for a mask. A
    label is looked up in the atlas of the column's sort, then in the
    other atlases of the engine (:func:`region_lookup`); the image is on
    the grid of the atlas the labels are of, else, of several, on the
    template's. Labels of no region are skipped.
    """
    points = request.points
    column = _column(df, points.columns[0])
    value = value_column(df, points, sorts, request.value, provenance)
    per_label = _region_values(df, column, value)
    lookup = region_lookup(nl, _region_atlases(nl, sorts, points.grid))
    found = [label for label in per_label.index if label in lookup]
    grid, template, index = _region_index(nl, lookup, found, points.grid)
    values = per_label[found].to_numpy()
    dtype = _image_dtype(values)
    fill = request.fill if value is not None else 0
    if dtype != np.float32 and float(fill) != int(fill):
        dtype = np.float32
    data = np.full(template.shape[:3], fill, dtype=dtype)
    painted = index >= 0
    data[painted] = values[index[painted]]
    if data.dtype.kind == "f":
        data[np.isnan(data)] = fill
    return SavedImage(
        nib.Nifti1Image(data, template.affine), value, int(painted.sum()),
        len(per_label) - len(found), grid, len(found),
    )


def _region_index(nl, lookup: dict, found: list, grid: str) -> tuple:
    """
    Return the grid, its template and the position in *found* of the
    region of each voxel, -1 outside them, of the regions *found* of the
    atlases of *lookup* (:func:`region_lookup`): on the grid of their
    atlas, else, of several, on the template's; on *grid* without any.
    """
    atlases = sorted({lookup[label][0] for label in found}) or [grid]
    if len(atlases) == 1:
        grid, target = atlases[0], None
        template = engine_grid(nl, grid)
    else:
        grid = engine_registry.TEMPLATE_GRID
        target = template = engine_template(nl)
    index = np.full(template.shape[:3], -1, dtype=np.int64)
    for atlas in atlases:
        labels = _label_volume(nl, atlas, target)
        lut = np.full(max(int(labels.max()), 0) + 1, -1, dtype=np.int64)
        for n, label in enumerate(found):
            if lookup[label][0] == atlas:
                lut[[v for v in lookup[label][1] if v < len(lut)]] = n
        at = lut[np.clip(labels, 0, None)]
        at[labels <= 0] = -1
        index[at >= 0] = at[at >= 0]
    return grid, template, index


# The colours of the regions of an atlas without colours of its own: those
# of matplotlib's tab20, in turn.
REGION_PALETTE = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
)


class RegionLayer(NamedTuple):
    """
    The regions of a result on the regions of atlases, as the viewer
    names them and colours them: in *image*, ``n + 1`` at the voxels of
    the region *names[n]*, whose colour, ``#rrggbb``, is *colours[n]*.
    The viewer starts with them in their colours if *coloured*.
    """

    image: nib.Nifti1Image
    names: List[str]
    colours: List[str]
    #: whether the viewer starts with the regions in their colours: those
    #: of a result without values, the regions alone
    coloured: bool = False


def regions_layer(
    nl, df: pd.DataFrame, sorts: dict, request: SaveRequest,
) -> RegionLayer:
    """
    Return the regions of the result *df* as the viewer names and colours
    them (:class:`RegionLayer`): those of the labels of the *request*'s
    points (see :func:`regions_to_nifti`), each of the colour its atlas
    gives it (:func:`~neurolang_apps.utils.engine_registry.atlas_colours`),
    else of :data:`REGION_PALETTE`.
    """
    points = request.points
    column = _column(df, points.columns[0])
    lookup = region_lookup(nl, _region_atlases(nl, sorts, points.grid))
    found = [
        label for label in dict.fromkeys(df[column].astype(str))
        if label in lookup
    ]
    _, template, index = _region_index(nl, lookup, found, points.grid)
    colours = []
    for n, label in enumerate(found):
        atlas, values = lookup[label]
        colour = engine_registry.atlas_colours(nl, atlas).get(values[0])
        colours.append(colour or REGION_PALETTE[n % len(REGION_PALETTE)])
    dtype = np.int16 if len(found) < np.iinfo(np.int16).max else np.int32
    return RegionLayer(
        nib.Nifti1Image((index + 1).astype(dtype), template.affine),
        found, colours,
    )


def region_centroids(nl, atlas: str) -> dict:
    """
    Return the centre of each region of the *atlas* of *nl*, by name.

    The centre is the mean of the millimetre coordinates of the centres
    of the region's voxels, on the grid of the atlas.
    """
    image = engine_grid(nl, atlas)
    labels = _label_volume(nl, atlas).ravel()
    inside = labels > 0
    indices = np.stack(np.unravel_index(
        np.flatnonzero(inside), image.shape[:3]
    ), -1)
    labels = labels[inside]
    counts = np.bincount(labels)
    sums = np.stack([
        np.bincount(labels, weights=indices[:, axis], minlength=len(counts))
        for axis in range(3)
    ], -1)
    centroids = {}
    for label, values in engine_registry.atlas_label_values(
        nl, atlas
    ).items():
        values = [v for v in values if v < len(counts) and counts[v]]
        if values:
            voxel = sums[values].sum(axis=0) / counts[values].sum()
            centroids[label] = nib.affines.apply_affine(image.affine, voxel)
    return centroids


class Network(NamedTuple):
    """
    A result of pairs of regions or points as a network, to view.

    Its nodes have *names* and the millimetre *coordinates* of their
    centre; each edge ``(first, second, value)`` joins two nodes, by
    index, with the *value* column's value of the pair, 1 without. A
    pair given both ways (*merged* of them), as a conditional probability
    is, keeps the largest value; rows whose node is unknown, *skipped*,
    and those joining a node to itself are left out. *strength* is the
    image of the sum of the values of the edges of each node.
    """

    names: list
    coordinates: np.ndarray
    edges: list
    value: Optional[str]
    merged: int
    skipped: int
    strength: SavedImage


def _point_names(df: pd.DataFrame, group: PointColumns, nl) -> tuple:
    """
    Return the name and the millimetre coordinates of the point of each
    row of *df* in the columns *group*, on the grid of *nl* it names.
    """
    coordinates = df[list(group.columns)].to_numpy(dtype=float)
    if group.space == "voxel":
        coordinates = nib.affines.apply_affine(
            engine_grid(nl, group.grid).affine, coordinates
        )
    names = [
        "(" + ", ".join(f"{v:g}" for v in row) + ")" for row in coordinates
    ]
    return names, coordinates


def _network_nodes(nl, df, groups: list, sorts: dict) -> tuple:
    """
    Return the node names of each row of *df*, of each of the two
    *groups* of columns, and the coordinates of each node, by name.
    """
    if groups[0].space != LABEL_SPACE:
        ends, centres = [], {}
        for group in groups:
            names, coordinates = _point_names(df, group, nl)
            ends.append(names)
            centres.update(zip(names, coordinates))
        return ends, centres
    atlases = _region_atlases(nl, sorts, groups[0].grid)
    atlases += [
        a for a in _region_atlases(nl, sorts, groups[1].grid)
        if a not in atlases
    ]
    lookup = region_lookup(nl, atlases)
    ends = [df[group.columns[0]].astype(str).tolist() for group in groups]
    used = {lookup[n][0] for names in ends for n in names if n in lookup}
    centres = {}
    for atlas in atlases:
        if atlas in used:
            centroids = region_centroids(nl, atlas)
            centres.update({
                name: centre for name, centre in centroids.items()
                if lookup.get(name, (None,))[0] == atlas
            })
    return ends, centres


def network_value_columns(
    df: pd.DataFrame, groups: list, sorts: dict, request: SaveRequest,
    provenance: Optional[tuple] = None,
) -> list:
    """
    Return the value columns of the edges of a network of *df*, whose
    nodes are the two *groups* of columns: that of the *request*'s
    ``--value`` if given, else its numeric columns that are not of an
    entity, the probability column first; [] for pairs of no value.
    """
    if request.value is not None:
        return [_numeric_column(df, request.value)]
    nodes = groups[0].columns + groups[1].columns
    others = [c for c in df.columns if c not in nodes]
    measures = _measure_columns(df, others, sorts, provenance)
    first = value_column(
        df, groups[0]._replace(columns=nodes), sorts, None, provenance
    )
    return sorted(measures, key=lambda c: c != first)


def result_network(
    nl, df: pd.DataFrame, sorts: dict, groups: list,
    request: SaveRequest = SaveRequest(Path()),
    provenance: Optional[tuple] = None,
) -> Network:
    """
    Return the result *df* of pairs as a :class:`Network` on *nl*.

    The two *groups* of columns locate the two ends of each edge (see
    :func:`network_nodes`): region labels, whose node is at the centre
    of the region (:func:`region_centroids`), or points. The value of an
    edge is that of the *request*'s ``--value``, else of the one value
    column (:func:`value_column`).
    """
    columns = groups[0].columns + groups[1].columns
    value = value_column(
        df, groups[0]._replace(columns=columns), sorts, request.value,
        provenance,
    )
    ends, centres = _network_nodes(nl, df, groups, sorts)
    values = (
        df[value].to_numpy(dtype=float) if value is not None
        else np.ones(len(df))
    )
    names, index, pairs, seen = [], {}, {}, set()
    skipped = merged = 0
    for first, second, weight in zip(*ends, values):
        if first not in centres or second not in centres:
            skipped += 1
            continue
        if first == second or np.isnan(weight):
            continue
        for name in (first, second):
            if name not in index:
                index[name] = len(names)
                names.append(name)
        key = tuple(sorted((index[first], index[second])))
        direction = (index[first], index[second])
        if key in pairs and direction not in seen:
            merged += 1
        seen.add(direction)
        pairs[key] = max(pairs.get(key, -np.inf), weight)
    edges = [(a, b, float(w)) for (a, b), w in sorted(pairs.items())]
    strength = np.zeros(len(names))
    for a, b, weight in edges:
        strength[a] += weight
        strength[b] += weight
    coordinates = np.array(
        [centres[name] for name in names], dtype=float
    ).reshape(-1, 3)
    return Network(
        names, coordinates, edges, value, merged, skipped,
        _strength_image(nl, names, coordinates, strength, groups, sorts),
    )


def _strength_image(nl, names, coordinates, strength, groups, sorts):
    """Return the image of the *strength* of each node of a network."""
    if groups[0].space == LABEL_SPACE:
        frame = pd.DataFrame({"label": names, "strength": strength})
        return regions_to_nifti(nl, frame, sorts, SaveRequest(
            Path(), "strength", PointColumns(
                ("label",), LABEL_SPACE, groups[0].grid
            ),
        ))
    frame = pd.DataFrame(coordinates, columns=["x", "y", "z"])
    frame["strength"] = strength
    return result_to_nifti(frame, engine_template(nl), sorts, SaveRequest(
        Path(), "strength", PointColumns(("x", "y", "z"), "mm"),
    ))


def engine_template(nl):
    """
    Return the template of the engine *nl*, or raise a SaveError.

    It is the image whose grid the engine's voxel indices refer to.
    """
    template = getattr(nl, "template", None)
    if template is None:
        raise SaveError(
            "The engine has no template: a NIfTI image is written on the "
            "grid of the template of an engine requiring the MNI mask. "
            "Save the result as CSV instead."
        )
    return template


def engine_grid(nl, name: str):
    """
    Return the image of the grid *name* of the engine *nl*.

    The template's grid is :func:`engine_template`; the others are the
    engine's ``grids``, such as its atlases'. A :class:`SaveError` tells
    the engine has no such grid.
    """
    if name == engine_registry.TEMPLATE_GRID:
        return engine_template(nl)
    image = (getattr(nl, "grids", None) or {}).get(name)
    if image is None:
        raise SaveError(
            f"The engine has no grid {name!r}, the grid of the voxel "
            "indices of the result: it registers the grid of an atlas "
            "when it loads it. Give the voxel columns with --voxel I J K "
            "to write them on the template's grid."
        )
    return image


NETWORK_MESSAGE = (
    "The result holds pairs of regions or points, a network: "
    ".view shows it, and .save writes it as CSV."
)


def image_columns(app, block) -> list:
    """
    Return the value columns a NIfTI image of the result *block* can hold,
    without making it; None stands for a mask, whose voxels are 1.

    Raise :class:`SaveError` when it cannot be an image: a network, or no
    voxel columns (what :func:`result_image` refuses).
    """
    sorts = _engine_sorts(app.settings.engine_name)
    if network_nodes(block.frame, sorts, block.sorts):
        raise SaveError(NETWORK_MESSAGE)
    point_columns(block.frame, sorts, block.sorts)
    return value_columns(app, block, SaveRequest(Path("result.nii.gz")))


def result_image(app, block, request: SaveRequest) -> SavedImage:
    """
    Convert the result *block* of *app* into a NIfTI image, as *request* asks.

    The voxel columns follow from the request, else from the sorts of
    the block's columns or their names (:func:`point_columns`); the
    image is on the grid of their sort.
    """
    sorts = _engine_sorts(app.settings.engine_name)
    points = request.points or point_columns(
        block.frame, sorts, block.sorts
    )
    if request.points is None and network_nodes(
        block.frame, sorts, block.sorts
    ):
        raise SaveError(NETWORK_MESSAGE)
    if points.space == LABEL_SPACE:
        return regions_to_nifti(
            app._nl, block.frame, sorts, request._replace(points=points),
            block.sorts,
        )
    return result_to_nifti(
        block.frame, engine_grid(app._nl, points.grid), sorts,
        request._replace(points=points), block.sorts,
    )


def value_columns(app, block, request: SaveRequest) -> list:
    """
    Return the value columns of the result *block* that ``.view`` shows.

    They are the column of the *request*'s ``--value`` if given, else
    the numeric columns that are neither of the voxels nor of an entity
    when there are several (each is an overlay), else the one column
    :func:`value_column` finds, None for a mask.
    """
    if request.value is not None:
        return [request.value]
    frame = block.frame
    sorts = _engine_sorts(app.settings.engine_name)
    points = request.points or point_columns(frame, sorts, block.sorts)
    others = [c for c in frame.columns if c not in points.columns]
    measures = _measure_columns(frame, others, sorts, block.sorts)
    if len(measures) > 1:
        return measures
    return [value_column(frame, points, sorts, None, block.sorts)]


def _engine_sorts(engine_name: str) -> dict:
    """Return the sorts of the engine *engine_name*, if any."""
    return engine_registry.get_sorts(engine_name) if engine_name else {}


def saved_image_message(path: Path, saved: SavedImage) -> str:
    """Return the message telling what :func:`result_to_nifti` wrote."""
    shape = "×".join(str(n) for n in saved.image.shape[:3])
    unit = "voxel" if saved.regions is None else "region"
    value = (
        f"1 at each {unit} (mask)" if saved.value is None
        else f"the max of {str(saved.value)!r} per {unit}"
    )
    regions = (
        "" if saved.regions is None else
        f"{saved.regions} region{'s' * (saved.regions != 1)}, "
    )
    message = (
        f"Saved NIfTI image to: {path} ({regions}{saved.voxels} voxels of "
        f"the {shape} {saved.grid} grid, value: {value})"
    )
    if saved.outside and saved.regions is not None:
        labels = (
            "1 label" if saved.outside == 1 else f"{saved.outside} labels"
        )
        message += f"; {labels} of no region of the atlas skipped"
    elif saved.outside:
        rows = "1 row" if saved.outside == 1 else f"{saved.outside} rows"
        message += f"; {rows} outside the {saved.grid} grid skipped"
    return message


def write_nifti(image: nib.Nifti1Image, path) -> None:
    """
    Write *image* to exactly *path*, gzipped if it ends in ``.gz``.

    ``nib.save`` may change the case of the suffix (``Out.Nii`` is
    written as ``Out.nii``), which leaves no file under *path* on a
    case-sensitive file system.
    """
    data = image.to_bytes()
    if str(path).lower().endswith(".gz"):
        data = gzip.compress(data)
    Path(path).write_bytes(data)


def read_nifti(path) -> nib.Nifti1Image:
    """
    Read the NIfTI image of exactly *path*, gzipped or not.

    ``nib.load`` looks for a lowercase suffix on case-sensitive file
    systems, so it cannot open ``Out.Nii``.
    """
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        return nib.Nifti1Image.from_bytes(data)
    except Exception:
        return nib.Nifti2Image.from_bytes(data)


def _save_nifti(app, block, request: SaveRequest) -> None:
    """
    Save the result *block* as the NIfTI image of *request*.

    What was written is told.
    """
    saved = result_image(app, block, request)
    write_nifti(saved.image, request.path)
    app.console.print(
        saved_image_message(request.path, saved), style="green",
        markup=False,
    )


# The default threshold of the viewer: the 95th percentile of the nonzero
# values, so the top 5% of the voxels are shown.
VIEW_THRESHOLD = brain_viewer.DEFAULTS["view_threshold"]

VIEW_USAGE = (
    ".view [PATH.nii[.gz]] [--threshold T|P%] [--cmap NAME] "
    "[--symmetric | --one-sided] [--fill V]"
)

VIEW_HELP = """\
[bold cyan].view [PATH][/bold cyan] shows a NIfTI image, or the last result,
in a browser.

Without a path, the last result is converted into an image as .save
converts it, with its options --value, --voxel and --mm. A result with
several value columns, numeric ones that are neither of the voxels nor
of an entity, gives an overlay per column, unless --value picks one.
A program with several queries, ans[name](...) :- ..., gives the layers
of all its results, each named after its query, which the viewer shows
together: images, a network and streamlines, one network and one set of
streamlines at most. The voxels without a value, not in the result or
NaN, take the value of --fill (default: 0, no value: hidden). A nonzero fill is a value of
every other voxel of the grid, which the percentiles count. The viewer
is an HTML page, saved to a temporary file and opened in the browser when
the REPL runs on a terminal; it needs no network, but a browser with
WebGL2. It shows the overlays over the MNI152 T1 in three slices, drawn
by NiiVue (left hemisphere on the left), the later ones over the earlier
ones: click or drag to move the cursor, whose MNI coordinates and values
it tells, and scroll to move through the slices. Its controls change,
live, for the overlay chosen in their list (shown, or hidden; ▲ and ▼
draw it over the next one or under the previous one), what these options
start each overlay with:
  --threshold T|P%      hide the voxels below T, or below the P-th
                        percentile of the nonzero values (default:
                        VIEW_THRESHOLD)
  --cmap NAME           the colormap, of matplotlib or nilearn (default:
                        DEFAULT_CMAP), of the first overlay; the others
                        take OVERLAY_CMAPS... in turn
  --symmetric           a colormap symmetric around 0, hiding |v| < T
  --one-sided           a colormap from the threshold to the maximum
Without either, the colormap of an overlay is symmetric when it has
negative values. The page also sets the opacity of each overlay; an
infinite value is drawn as the largest finite one.

A result on the regions of an atlas shows every region of the result,
painted with its value, unless --threshold is given. A result of pairs,
two columns of region labels, or two points (x1, y1, z1, x2, y2, z2),
and a value, such as the co-activation of two regions,
  ans(l1, l2, p) :- (l1 < l2), PROB[co_active(l1, l2, s) // ...] = p
is a network: a node at the centre of each region or at each point, an
edge per pair, drawn in the 3D render and on the slices near them, and
an overlay of the strength of each node, the sum of its edges. The
edges shown are those of |value| at least --threshold, by default the
top 5% (20 at least); a pair given both ways keeps its largest value,
and the strongest edges are listed under the views.

A result of streamline ids, of an engine of tractograms, shows the
streamlines, coloured by their direction, in the 3D render and on the
slices near them, over the engine's background, and an overlay of the
number of streamlines through each voxel; 20000 at most are drawn,
evenly subsampled. A column holds streamline ids when, without voxel
columns, it is of the sort streamline (engines.yaml), or, the engine
having streamlines, it is the one column of integers of unknown sort:
  ans(x) :- uncinate(x, "left")
ans(x, y, z) stays millimetres. An engine has streamlines when it has
the method streamline_points(ids), giving the points of each id, an
(n, 3) array in millimetres (RAS+); its streamline_background(), if
any, gives the image they are drawn on, in their space, else the MNI152
T1 is.""".replace("VIEW_THRESHOLD", str(VIEW_THRESHOLD)).replace(
    "DEFAULT_CMAP", brain_viewer.DEFAULT_CMAP).replace(
    "OVERLAY_CMAPS", ", ".join(brain_viewer.OVERLAY_CMAPS[:3]))


HELP_TOPICS["view"] = VIEW_HELP

# The options of .view and the number of values each takes; without a
# path, the options of .save tell how to convert the last result.
_VIEW_OPTIONS = {
    "--threshold": 1, "--cmap": 1, "--symmetric": 0, "--one-sided": 0,
    "--fill": 1,
}


class ViewRequest(NamedTuple):
    """
    What ``.view`` shows: the image *path*, or the last result if None.

    *save* tells how to convert the last result, as ``.save`` would;
    *symmetric* is None when the image decides (see :func:`symmetry`).
    *threshold_given* tells whether ``--threshold`` gave the *threshold*:
    the regions of a result on an atlas are all shown otherwise.
    """

    path: Optional[Path]
    save: SaveRequest
    threshold: object = VIEW_THRESHOLD
    cmap: Optional[str] = None
    symmetric: Optional[bool] = None
    threshold_given: bool = False


def parse_view_args(arg: str) -> ViewRequest:
    """
    Return the request of the arguments *arg* of ``.view``.

    A :class:`SaveError` tells why they do not follow :data:`VIEW_USAGE`.
    """
    paths, options = split_args(
        arg, {**_VIEW_OPTIONS, **_SAVE_OPTIONS}, VIEW_USAGE
    )
    if len(paths) > 1:
        raise SaveError(f"Usage: {VIEW_USAGE}")
    conversion = {
        k: v for k, v in options.items() if k in _SAVE_OPTIONS
        or k == "--fill"
    }
    if paths and conversion:
        raise SaveError(
            f"{', '.join(conversion)}: options to view the last result, "
            "not a file."
        )
    save = _save_request(Path("last.nii.gz"), {
        k: v for k, v in conversion.items() if k != "--fill"
    })
    if "--fill" in conversion:
        save = save._replace(fill=_fill_value(conversion["--fill"][0]))
    return ViewRequest(
        Path(paths[0]).expanduser() if paths else None, save,
        view_threshold(options.get("--threshold", [None])[0]),
        options.get("--cmap", [None])[0], _symmetry_option(options),
        "--threshold" in options,
    )


def _fill_value(text: str) -> float:
    """Return the value of ``--fill`` *text*, or raise a SaveError."""
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not np.isfinite(value):
        raise SaveError(f"--fill takes a number, not {text!r}.")
    return value


def view_threshold(text: Optional[str]):
    """
    Return the threshold *text*: a value, or a percentile such as ``95%``.

    Without *text*, it is :data:`VIEW_THRESHOLD`.
    """
    if text is None:
        return VIEW_THRESHOLD
    number = text[:-1] if text.endswith("%") else text
    try:
        value = float(number)
    except ValueError:
        value = -1.0
    upper = 100.0 if text.endswith("%") else float("inf")
    if not 0 <= value <= upper:
        raise SaveError(
            f"--threshold takes a value ≥ 0 or a percentile such as 95%, "
            f"not {text!r}."
        )
    return text if text.endswith("%") else value


def _symmetry_option(options: dict) -> Optional[bool]:
    """Return the colormap symmetry the *options* ask for, if any."""
    if "--symmetric" in options and "--one-sided" in options:
        raise SaveError("Give --symmetric or --one-sided, not both.")
    if "--symmetric" in options:
        return True
    return False if "--one-sided" in options else None


def symmetry(
    console, image: nib.Nifti1Image, symmetric: Optional[bool],
    name: str = "",
) -> bool:
    """
    Return whether the colormap of the *image* is symmetric.

    Unless *symmetric* says, it is when the image has negative values;
    *console* is then told which was chosen and how to change it, for
    the overlay *name* if given.
    """
    if symmetric is not None:
        return symmetric
    signed = bool((np.asanyarray(image.dataobj) < 0).any())
    colormap = f"Colormap of {name!r}" if name else "Colormap"
    console.print(
        f"{colormap}: symmetric, the image has negative values "
        "(--one-sided for a one-sided one)." if signed else
        f"{colormap}: one-sided, the image has no negative values "
        "(--symmetric for a symmetric one).",
        style="dim", markup=False,
    )
    return signed


def _check_cmap(name: Optional[str]) -> None:
    """Raise a SaveError unless *name* is a matplotlib or nilearn colormap."""
    if name is None:
        return
    import matplotlib
    from nilearn.plotting import cm

    if name not in matplotlib.colormaps and name not in getattr(
        cm, "_cmap_d", {}
    ):
        raise SaveError(f"Unknown colormap {name!r}.")


def _viewer_page(
    app, layers: list, request: ViewRequest, title: str, **extra
) -> str:
    """
    Return the viewer page, entitled *title*, of the images *layers*.

    *layers* are (name, path) pairs, an overlay each; a single one is
    shown without its name. The page starts as *request* asks. The images
    are drawn over the MNI152 T1 template of nilearn, whatever the
    engine's template, a mask, is; nilearn's warnings are kept off the
    prompt. The *extra* keyword arguments of
    :func:`~.brain_viewer.viewer_html`, a ``connectome`` or
    ``streamlines``, are drawn over them, in the 3D render; a
    ``background`` replaces the MNI152 T1.
    """
    _check_cmap(request.cmap)
    images = [(name, read_nifti(path)) for name, path in layers]
    several = len(images) > 1
    symmetric = [
        symmetry(
            app.console, image, request.symmetric, name if several else ""
        )
        for name, image in images
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        background = extra.pop("background", None)
        return brain_viewer.viewer_html(
            images if several else images[0][1],
            background or brain_viewer.mni_background(), title=title,
            threshold=request.threshold, cmap=request.cmap,
            symmetric=symmetric if several else symmetric[0], **extra,
        )


def _view_nifti(
    app, layers: list, request: ViewRequest, title: str, **extra
) -> None:
    """
    Show the NIfTI images *layers* in the brain viewer, as *request* asks.

    *layers* are (name, path) pairs, an overlay each, and *extra* what
    is drawn over them, if any (see :func:`_viewer_page`). The viewer
    page, entitled *title*, is saved as an HTML file whose path is
    printed, and opened in a browser when the REPL runs on a terminal.
    """
    try:
        html = _html_file(
            _viewer_page(app, layers, request, title, **extra)
        )
    except Exception as exc:
        app.console.print(
            f"Visualisation failed: {exc}", style="red", markup=False
        )
        return
    app.console.print(
        f"Saved the viewer to: {html}", style="green", markup=False,
        highlight=False,
    )
    _open_in_browser(app.console, html)


def _html_file(page: str) -> Path:
    """Save the HTML *page* in a new temporary file, its path."""
    fd, html = tempfile.mkstemp(suffix=".html", prefix="neurolang_view_")
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(page)
    return Path(html)


def _open_in_browser(console, html: Path) -> None:
    """Open *html* in a browser on a terminal, else tell to open it."""
    if console.is_terminal and webbrowser.open(html.as_uri()):
        console.print("Opened it in the browser.", style="dim")
    else:
        console.print("Open it in a browser to see the image.", style="dim")


_COMPOUND_SUFFIXES = (".nii.gz", ".csv.gz", ".tsv.gz")


def split_suffix(path: Path) -> tuple:
    """
    Return the name of *path* without its suffix, and its suffix.

    The suffix of ``out.nii.gz`` is ``.nii.gz``, in its own case, as
    are those of ``.csv.gz`` and ``.tsv.gz``.
    """
    name = path.name
    for suffix in _COMPOUND_SUFFIXES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)], name[-len(suffix):]
    return path.stem, path.suffix


def is_nifti_path(path: Path) -> bool:
    """Return whether *path* names a NIfTI file: ``.nii`` or ``.nii.gz``."""
    return split_suffix(path)[1].lower() in (".nii", ".nii.gz")


def _save_path(path: Path, title: str, several: bool) -> Path:
    """
    Return where to save the block *title* of a result in *path*.

    With *several* blocks, the block's title is added to the name, before
    its suffix (``out_first.nii.gz``).
    """
    if not several:
        return path
    stem, suffix = split_suffix(path)
    return path.with_name(f"{stem}_{title}{suffix}")


def table_format(path: Path) -> tuple:
    """
    Return the separator of the table file *path* and its format's name.

    A ``.tsv`` file, possibly gzipped, is tab-separated and any other
    comma-separated; pandas compresses the file after its extension.
    """
    suffix = split_suffix(path)[1].lower()
    sep, name = ("\t", "TSV") if suffix.startswith(".tsv") else (",", "CSV")
    if suffix.endswith(".gz"):
        name += " (gzip)"
    return sep, name


def _save_table(console, df: pd.DataFrame, path: Path) -> None:
    """Save *df* as the CSV or TSV file *path* and tell what was written."""
    sep, name = table_format(path)
    df.to_csv(path, index=False, sep=sep)
    rows = "1 row" if len(df) == 1 else f"{len(df)} rows"
    console.print(
        f"Saved {rows} as {name} to: {path}", style="green", markup=False
    )


def _save_failed(app, message: str) -> None:
    """Report that ``.save`` failed with *message*."""
    app.failed = True
    app.err_console.print(
        f"Save failed: {message}", style="red", markup=False, soft_wrap=True
    )


def _save_block(app, block, request: SaveRequest) -> None:
    """
    Save the result *block* as *request* asks.

    It is saved as NIfTI for ``.nii(.gz)``, else as a table.
    """
    try:
        if is_nifti_path(request.path):
            _save_nifti(app, block, request)
        else:
            _save_table(app.console, block.frame, request.path)
    except Exception as exc:
        _save_failed(app, str(exc))


@command(
    ".save", "Save the last result as CSV/TSV, or NIfTI for .nii(.gz) "
    "(.help save)", SAVE_USAGE,
)
def cmd_save(app, arg: str) -> None:
    """
    Save the last result as the arguments *arg* ask (see SAVE_HELP).

    With several ``obtain`` blocks, each is saved in its own file, named
    after the block (``PATH_NAME.csv``).
    """
    if not app.results:
        app.console.print(
            "[yellow]Nothing to save — run a query first.[/yellow]"
        )
        return
    try:
        request = parse_save_args(arg)
    except SaveError as exc:
        _save_failed(app, str(exc))
        return
    request.path.parent.mkdir(parents=True, exist_ok=True)
    several = len(app.results) > 1
    for block in app.results:
        path = _save_path(request.path, block.title, several)
        _save_block(app, block, request._replace(path=path))


def last_result_images(app, block, request: SaveRequest) -> Optional[list]:
    """
    Save the result *block* as NIfTI images in temporary files.

    The block is converted as ``.save`` converts it, as *request* asks,
    into an image per value column (:func:`value_columns`). Return the
    (column name, path) of each, or None, telling why, if it has no
    voxels.
    """
    try:
        saved = [
            result_image(app, block, request._replace(value=column))
            for column in value_columns(app, block, request)
        ]
    except Exception as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return None
    return [
        (str(image.value or "mask"), _view_image_file(app, image))
        for image in saved
    ]


def _view_image_file(app, image: SavedImage) -> Path:
    """Write *image* to a temporary file for the viewer, its path."""
    fd, name = tempfile.mkstemp(suffix=".nii.gz", prefix="neurolang_view_")
    os.close(fd)
    write_nifti(image.image, name)
    app.console.print(
        saved_image_message(Path(name), image), style="green", markup=False,
    )
    return Path(name)


def _result_kind(app, block, request: SaveRequest) -> str:
    """
    Return what the result *block* of *app* holds: ``network`` for pairs
    of regions or points, ``regions`` for regions of an atlas, else
    ``voxels``; without voxel columns, ``streamlines`` for streamline ids
    (:func:`streamline_column`), of the sort :data:`STREAMLINE_SORT` or,
    the engine having streamlines (:func:`has_streamlines`), integers.
    """
    sorts = _engine_sorts(app.settings.engine_name)
    if request.points is not None:
        return "voxels"
    if network_nodes(block.frame, sorts, block.sorts):
        return "network"
    try:
        points = point_columns(block.frame, sorts, block.sorts)
    except SaveError:
        column = streamline_column(
            block.frame, block.sorts, integers=has_streamlines(app._nl)
        )
        return "voxels" if column is None else "streamlines"
    return "regions" if points.space == LABEL_SPACE else "voxels"


def last_result_network(app, block, request: ViewRequest):
    """
    Return the result *block*, of pairs, as a network to view.

    Return the layer of the strength of its nodes, (name, path) of a
    temporary NIfTI file, and the network as the viewer draws it
    (:func:`~.brain_viewer.connectome_state`), or None, telling why, if
    it cannot be built.
    """
    sorts = _engine_sorts(app.settings.engine_name)
    try:
        groups = network_nodes(block.frame, sorts, block.sorts)
        choices = network_value_columns(
            block.frame, groups, sorts, request.save, block.sorts
        )
        threshold = (
            request.threshold if request.threshold_given else VIEW_THRESHOLD
        )
        states, networks = [], []
        for choice in choices or [request.save.value]:
            network = result_network(
                app._nl, block.frame, sorts, groups,
                request.save._replace(value=choice), block.sorts,
            )
            if not network.edges:
                raise SaveError(
                    "The result has no pair of two known regions or points."
                )
            networks.append(network)
            states.append(brain_viewer.connectome_state(
                network.names, network.coordinates, network.edges, threshold
            ))
        network, connectome = networks[0], states[0]
        if len(states) > 1:  # each value column, for the page to choose
            connectome["value"] = str(choices[0])
            connectome["values"] = [
                {"value": str(choice), **state}
                for choice, state in zip(choices, states)
            ]
    except Exception as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return None
    shown = len(brain_viewer.shown_edges(connectome))
    value = repr(str(network.value)) if network.value else "1 (pairs)"
    if len(states) > 1:
        value += " (also " + ", ".join(
            repr(str(choice)) for choice in choices[1:]
        ) + ": the edge value of the viewer)"
    message = (
        f"Network of {len(network.names)} nodes and {len(network.edges)} "
        f"edges, value: {value}; {shown} shown, those of values from "
        f"{connectome['threshold']:g} (--threshold, or the edge threshold "
        "of the viewer)"
    )
    if network.merged:
        message += (
            f"; {network.merged} pairs given both ways keep their largest "
            "value"
        )
    if network.skipped:
        message += f"; {network.skipped} rows of unknown nodes skipped"
    app.console.print(message, style="green", markup=False)
    path = _view_image_file(app, network.strength)
    return [("node strength", path)], connectome


# The sort of the columns of streamline ids.
STREAMLINE_SORT = "streamline"


def has_streamlines(nl) -> bool:
    """Return whether the engine *nl* has streamlines: its hook
    ``streamline_points``."""
    return callable(getattr(nl, "streamline_points", None))


def streamline_column(
    df: pd.DataFrame, provenance: Optional[tuple] = None,
    integers: bool = True,
):
    """
    Return the column of *df* holding streamline ids, or None.

    It is the column of the sort :data:`STREAMLINE_SORT` in the
    *provenance* of *df*, else, with *integers*, its one column of
    integers of unknown sort, as ``ans(x) :- uncinate(x, "left")`` gives
    when the query's predicates declare no arguments.
    """
    if provenance is None or len(provenance) != len(df.columns):
        provenance = (None,) * len(df.columns)
    for column, sort in zip(df.columns, provenance):
        if sort is not None and sort.sort == STREAMLINE_SORT:
            return column
    found = [
        column for column, sort in zip(df.columns, provenance)
        if sort is None and pd.api.types.is_integer_dtype(df[column])
    ]
    return found[0] if integers and len(found) == 1 else None


def _visits(lines, shape, affine: np.ndarray) -> np.ndarray:
    """Return the number of the streamlines *lines* through each voxel of
    the grid (*shape*, *affine*)."""
    counts = np.zeros(shape[:3], dtype=np.int32)
    inverse = np.linalg.inv(affine)
    for line in lines:
        voxels = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        inside = np.all((voxels >= 0) & (voxels < shape[:3]), axis=1)
        counts[tuple(np.unique(voxels[inside], axis=0).T)] += 1
    return counts


def streamline_background(nl):
    """Return the background of the streamlines of the engine *nl* and
    the name of its millimetres: its ``streamline_background()``, in
    ``RAS``, else, without one or if it gives None (streamlines in MNI
    space), the MNI152 T1, in ``MNI``."""
    hook = getattr(nl, "streamline_background", None)
    background = hook() if callable(hook) else None
    if background is not None:
        return background, "RAS"
    background = brain_viewer.mni_background()
    if background is None:
        raise SaveError("No background to draw the streamlines on.")
    return background, "MNI"


def last_result_streamlines(app, block):
    """
    Return the result *block*, of streamline ids, as streamlines.

    The engine's ``streamline_points`` gives the points of
    :data:`~.brain_viewer.MAX_STREAMLINES` of them at most, evenly
    subsampled (:func:`~.brain_viewer.shown_streamlines`). Return the
    layer of the number of the streamlines drawn through each voxel of
    the background, (name, path) of a temporary NIfTI file, and the
    keyword arguments of the viewer: the streamlines
    (:func:`~.brain_viewer.streamlines_state`) and the background
    (:func:`streamline_background`); or None, telling why, if they cannot
    be built.
    """
    column = streamline_column(block.frame, block.sorts)
    try:
        if not has_streamlines(app._nl):
            raise SaveError(
                f"The column {str(column)!r} holds streamline ids, but the "
                "engine has no streamlines: it gives their points with "
                "streamline_points(ids) (.help view)."
            )
        ids = sorted(int(s) for s in pd.unique(block.frame[column]))
        if not ids:
            raise SaveError("The result has no streamlines.")
        shown = brain_viewer.shown_streamlines(ids)
        lines = [
            np.asarray(points, dtype=np.float32).reshape(-1, 3)
            for points in app._nl.streamline_points(shown)
        ]
        background, space = streamline_background(app._nl)
        counts = _visits(lines, background.shape, background.affine)
        state = brain_viewer.streamlines_state(lines, len(ids))
    except Exception as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return None
    subsampled = (
        f", evenly subsampled to {brain_viewer.MAX_STREAMLINES} at most"
        if len(lines) < len(ids) else ""
    )
    app.console.print(
        f"Streamlines of the column {str(column)!r}: {len(lines)} of "
        f"{len(ids)} shown{subsampled}", style="green", markup=False,
    )
    if not callable(getattr(app._nl, "streamline_background", None)):
        app.console.print(
            "The engine has no streamline_background(): they are drawn on "
            "the MNI152 T1, where only streamlines in MNI space fall right.",
            style="yellow", markup=False,
        )
    path = _view_image_file(app, SavedImage(
        nib.Nifti1Image(counts, background.affine), "streamlines",
        int((counts > 0).sum()), 0, "background",
    ))
    return (
        [("streamlines", path)],
        {"streamlines": state, "background": background, "space": space},
    )


def _one_of_a_kind(extra: dict, found: dict, console, title: str) -> bool:
    """
    Add what a block draws, *found*, to *extra*, the viewer's keyword
    arguments, and return whether it was added: the viewer draws one
    network and one set of streamlines, so those of a later block are
    left out, telling it on *console*.
    """
    for key in ("connectome", "streamlines"):
        if key in found and key in extra:
            console.print(
                f"Only one {'network' if key == 'connectome' else key} is "
                f"drawn: that of {title or 'the result'} is left out.",
                style="yellow", markup=False,
            )
            return False
    # The regions named on the views are those of the first block of
    # regions.
    for key, value in found.items():
        extra.setdefault(key, value)
    return True


def _block_layers(app, block, request: ViewRequest):
    """
    Return the layers of the result *block* for the viewer, as
    (kind, layers, extra), or None, telling why, if it cannot be shown.

    *kind* is what the block holds (:func:`_result_kind`), *layers* are
    (name, path) pairs of temporary NIfTI files and *extra* the keyword
    arguments of the viewer that the block draws, if any.
    """
    kind = _result_kind(app, block, request.save)
    if kind == "network":
        built = last_result_network(app, block, request)
        if built is not None:
            built = (built[0], {"connectome": built[1]})
    elif kind == "streamlines":
        built = last_result_streamlines(app, block)
    else:
        layers = last_result_images(app, block, request.save)
        built = None if layers is None else (layers, {})
        if built is not None and kind == "regions":
            regions = result_regions(app, block, request.save)
            if regions is not None:
                masks = all(name == "mask" for name, _ in layers)
                built = (layers, {"regions": regions._replace(
                    coloured=masks
                )})
    return None if built is None else (kind, *built)


def result_regions(app, block, request: SaveRequest):
    """
    Return the regions of the result *block* of *app*, on the regions of
    atlases, as the viewer names and colours them (:func:`regions_layer`),
    or None if they cannot be found.
    """
    sorts = _engine_sorts(app.settings.engine_name)
    try:
        points = request.points or point_columns(
            block.frame, sorts, block.sorts
        )
        return regions_layer(
            app._nl, block.frame, sorts, request._replace(points=points)
        )
    except Exception:  # the result is shown without its regions' names
        return None


def last_results_layers(app, request: ViewRequest):
    """
    Return the layers of all the results of *app*, as (layers, extra,
    request), or None, telling why, if they cannot be shown.

    Each block of the last result, a query's or an ``obtain``'s or the
    ``ans[name]`` of a program with several, gives its images, network
    or streamlines; the viewer shows them together, the layers of a
    result of several named after its block. The *request*'s threshold
    (the top 5% by default) is 0 for results neither of voxels nor of
    regions with a value (regions alone, networks, streamlines), unless
    it is given: the one the viewer starts all the layers with.
    """
    if not app.results:
        app.console.print(
            "[yellow]Nothing to view — run a query first, or give a NIfTI "
            "file.[/yellow]"
        )
        return None
    several = len(app.results) > 1
    layers, extra, kinds = [], {}, set()
    for block in app.results:
        built = _block_layers(app, block, request)
        if built is None:
            return None
        kind, block_layers, found = built
        if not _one_of_a_kind(extra, found, app.console, block.title):
            continue
        kinds.add(kind)
        # regions with a value, not a mask: their top 5% as voxels'
        if kind == "regions" and any(
            name != "mask" for name, _ in block_layers
        ):
            kinds.add("values")
        layers += [
            (f"{block.title}: {name}" if several and block.title else name,
             path)
            for name, path in block_layers
        ]
    if not kinds & {"voxels", "values"} and not request.threshold_given:
        request = request._replace(threshold=0.0)
    return layers, extra, request


@command(
    ".view", "Show a NIfTI image, or the last result, in an interactive "
    "viewer in the browser (.help view)", VIEW_USAGE,
)
def cmd_view(app, arg: str) -> None:
    """
    Show the NIfTI image of *arg*, or the last result, in the brain viewer.

    Without a path, the last result is converted as ``.save`` would, its
    options included, into temporary ``.nii.gz`` files, an overlay per
    value column; those of all the results of a program with several
    queries are shown together.
    """
    try:
        request = parse_view_args(arg)
    except SaveError as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return
    extra = {}
    if request.path is None:
        shown = last_results_layers(app, request)
        if shown is None:
            return
        layers, extra, request = shown
    elif not request.path.is_file():
        app.console.print(
            f"File not found: {request.path}", style="red", markup=False
        )
        return
    else:
        layers = [(request.path.name, request.path)]
    _view_nifti(
        app, layers, request, _view_title(app, request, layers[0][1]),
        **extra,
    )


def _view_title(app, request: ViewRequest, path: Path) -> str:
    """Return the title of the viewer: the image's name, or the result's."""
    if request.path is not None:
        return path.name
    titles = [block.title for block in app.results if block.title]
    if len(titles) > 1:
        return f"Last results: {', '.join(titles)}"
    return f"Last result: {titles[0]}" if titles else "Last result"
