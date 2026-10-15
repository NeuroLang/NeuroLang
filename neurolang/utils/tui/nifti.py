"""
Saving and viewing of query results in the interactive REPL.

``.save PATH`` writes the last result as a table, or as a NIfTI image
when PATH ends with ``.nii`` or ``.nii.gz``; ``.view PATH`` shows a NIfTI
image, or the last result, in a browser: an HTML page of
:mod:`~neurolang.utils.tui.brain_viewer`, whose controls change the
threshold, the colormap and its symmetry live (see :data:`VIEW_HELP`).

A NIfTI image is written on the grid of the engine's template, the MNI
image its voxel indices refer to (the ``template`` of the engine
:func:`~neurolang.utils.engine_registry.build_engine` returns), with its
shape and affine. The engine's sorts tell which columns of the result
are the voxels (:func:`point_columns`) and :func:`value_column` which
column holds their value; see :data:`SAVE_HELP`.
"""

import os
import shlex
import tempfile
import warnings
import webbrowser
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.utils import engine_registry

from . import brain_viewer
from .commands import HELP_TOPICS, command

SAVE_USAGE = ".save PATH [--value COLUMN] [--voxel I J K | --mm X Y Z]"

SAVE_HELP = """\
[bold cyan].save PATH[/bold cyan] writes the last result (a file per obtain block).

  out.csv, out.tsv      a table, comma- or tab-separated, gzipped for a
  out.csv.gz            .gz path; any other path is CSV
  out.nii, out.nii.gz   a NIfTI image on the grid of the engine's
                        template (its shape and affine)

The voxels of a NIfTI image are the columns of the result named as those
of a sort of the engine declaring a space (engines.yaml):
  space: voxel          grid indices (i, j, k), written as they are
  space: mm             millimetres (x, y, z), converted to the nearest
                        voxel with the inverse of the template's affine
or those given with --voxel I J K (indices) or --mm X Y Z (millimetres).
In Datalog, name the variables of the head after the sort, e.g.
ans(i, j, k) :- voxel(i, j, k).

The value of a voxel is that of the column given with --value COLUMN,
else of the probability column, else of the one numeric column that is
neither a voxel column nor an entity's (a column of a sort that is not
of type float, such as study_id); otherwise the image is a mask, 1 at
each voxel of the result. Rows of the same voxel keep their maximum.
A mask is written as uint8, integers as int32 and other values as
float32; rows outside the template grid are skipped."""


HELP_TOPICS["save"] = SAVE_HELP


class SaveError(ValueError):
    """Raised when a result cannot be saved as asked."""


class PointColumns(NamedTuple):
    """
    The columns of a result locating its voxels, and their space.

    The *space* is ``voxel`` for the indices of the template grid, ``mm``
    for millimetres in the template's space.
    """

    columns: Tuple[str, ...]
    space: str


class SaveRequest(NamedTuple):
    """
    What ``.save`` writes: its *path* and, for NIfTI, the columns to use.

    *value* names the value column and *points* the voxel columns; when
    None, they follow from the result and the engine's sorts.
    """

    path: Path
    value: Optional[str] = None
    points: Optional[PointColumns] = None


class SavedImage(NamedTuple):
    """
    A result converted to a NIfTI *image*, and how.

    *value* is the column of the voxel values, None for a mask; *voxels*
    the number of voxels written and *outside* the number of rows outside
    the template grid, skipped.
    """

    image: nib.Nifti1Image
    value: Optional[str]
    voxels: int
    outside: int


# The options of .save and the number of columns each takes.
_SAVE_OPTIONS = {"--value": 1, "--voxel": 3, "--mm": 3}


# How each option is written, for the messages.
_OPTION_FORMS = {
    "--value": "--value COLUMN", "--voxel": "--voxel I J K",
    "--mm": "--mm X Y Z", "--threshold": "--threshold T",
    "--cmap": "--cmap NAME", "--symmetric": "--symmetric",
    "--one-sided": "--one-sided",
}


def split_args(arg: str, widths: dict, usage: str) -> tuple:
    """
    Return the paths and the options of the command arguments *arg*.

    *widths* gives the number of values each option takes; the options
    are returned as a dict of their values. A :class:`SaveError` tells
    why *arg* does not follow *usage*.
    """
    try:
        tokens = shlex.split(arg)
    except ValueError as exc:
        raise SaveError(f"{exc}. Usage: {usage}") from exc
    paths, options = [], {}
    while tokens:
        token = tokens.pop(0)
        if token.startswith("--"):
            options[token] = _option_values(token, tokens, widths)
        else:
            paths.append(token)
    return paths, options


def parse_save_args(arg: str) -> SaveRequest:
    """
    Return the request of the arguments *arg* of ``.save``.

    They are a path, quoted if it holds spaces, and the options of
    :data:`SAVE_USAGE`; a :class:`SaveError` tells why they are not.
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


def point_columns(df: pd.DataFrame, sorts: dict) -> PointColumns:
    """
    Return the columns of *df* locating its voxels, from the *sorts*.

    They are the columns of a sort declaring a space (see
    :class:`~neurolang.utils.engine_registry.Sort`) that *df* has, those
    of a sort of voxel indices first, then of millimetres.
    """
    names = set(df.columns)
    for space in engine_registry.SORT_SPACES:
        for sort in sorts.values():
            if sort.space == space and set(sort.columns) <= names:
                return PointColumns(sort.columns, space)
    raise SaveError(_no_points_message(df, sorts))


def _no_points_message(df: pd.DataFrame, sorts: dict) -> str:
    """Return the error on *df*, whose voxel columns cannot be found."""
    spaces = {"voxel": "indices", "mm": "millimetres"}
    known = [
        f"{sort.name} ({', '.join(sort.columns)}, {spaces[sort.space]})"
        for sort in sorts.values() if sort.space
    ]
    columns = ", ".join(str(c) for c in df.columns)
    hint = (
        f"name them as those of {' or '.join(known)}, or "
        if known else "the engine declares no sort of voxels; "
    )
    return (
        f"The result has no voxel columns (its columns are {columns}): "
        f"{hint}give them with --voxel I J K or --mm X Y Z."
    )


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
    explicit: Optional[str] = None,
):
    """
    Return the column of *df* holding the value of its voxels, or None.

    It is the *explicit* column if given, else the probability column,
    else the one numeric column that is neither of the *points* nor of an
    entity, a sort of the *sorts* that is not of type float (such as a
    study); None stands for a mask.
    """
    if explicit is not None:
        return _numeric_column(df, explicit)
    others = [c for c in df.columns if c not in points.columns]
    probability = [
        c for c in others if str(c).lower() in _PROBABILITY_COLUMNS
    ]
    if probability:
        return probability[0]
    measures = _measure_columns(df, others, sorts)
    return measures[0] if len(measures) == 1 else None


def _measure_columns(df: pd.DataFrame, columns: list, sorts: dict) -> list:
    """
    Return the numeric *columns* of *df* that are not of an entity.

    The columns of an entity are those of the *sorts* not of type float,
    such as ``study_id``.
    """
    entities = {
        column for sort in sorts.values() if sort.type != "float"
        for column in sort.columns
    }
    return [
        c for c in columns
        if c not in entities and pd.api.types.is_numeric_dtype(df[c])
    ]


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


def _volume(per_voxel: pd.Series, shape: tuple) -> np.ndarray:
    """Return the volume of *shape* holding the values of *per_voxel*."""
    values = per_voxel.to_numpy()
    data = np.zeros(shape, dtype=_image_dtype(values))
    indices = per_voxel.index.to_frame().to_numpy(dtype=int).reshape(-1, 3)
    data[tuple(indices.T)] = values
    return data


def result_to_nifti(
    df: pd.DataFrame, template, sorts: dict,
    request: SaveRequest = SaveRequest(Path()),
) -> SavedImage:
    """
    Convert the result *df* into a NIfTI image on the *template*'s grid.

    The voxels and their value follow from the *request*, else from the
    *sorts* (:func:`point_columns`, :func:`value_column`); rows of the
    same voxel keep their maximum and rows outside the grid are skipped.
    """
    points = request.points or point_columns(df, sorts)
    for name in points.columns:
        _column(df, name)
    value = value_column(df, points, sorts, request.value)
    indices = voxel_indices(df, points, template.affine)
    shape = tuple(template.shape[:3])
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    values = (
        df[value].to_numpy()[inside] if value is not None
        else np.ones(inside.sum(), dtype=np.uint8)
    )
    per_voxel = _max_per_voxel(indices[inside], values)
    image = nib.Nifti1Image(_volume(per_voxel, shape), template.affine)
    return SavedImage(
        image, value, len(per_voxel), int(len(df) - inside.sum())
    )


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


def _engine_sorts(engine_name: str) -> dict:
    """Return the sorts of the engine *engine_name*, if any."""
    return engine_registry.get_sorts(engine_name) if engine_name else {}


def saved_image_message(path: Path, saved: SavedImage, shape) -> str:
    """Return the message telling what :func:`result_to_nifti` wrote."""
    grid = "×".join(str(n) for n in shape[:3])
    value = (
        "1 at each voxel (mask)" if saved.value is None
        else f"the max of {str(saved.value)!r} per voxel"
    )
    message = (
        f"Saved NIfTI image to: {path} ({saved.voxels} voxels of the "
        f"{grid} template grid, value: {value})"
    )
    if saved.outside:
        rows = "1 row" if saved.outside == 1 else f"{saved.outside} rows"
        message += f"; {rows} outside the template grid skipped"
    return message


def _save_nifti(app, df: pd.DataFrame, request: SaveRequest) -> None:
    """Save *df* as the NIfTI image of *request* and tell what was written."""
    template = engine_template(app._nl)
    saved = result_to_nifti(
        df, template, _engine_sorts(app.settings.engine_name), request
    )
    nib.save(saved.image, request.path)
    app.console.print(
        saved_image_message(request.path, saved, template.shape),
        style="green", markup=False,
    )


# The default threshold of the viewer: every nonzero voxel is shown.
VIEW_THRESHOLD = 1e-6

VIEW_USAGE = (
    ".view [PATH.nii[.gz]] [--threshold T|P%] [--cmap NAME] "
    "[--symmetric | --one-sided]"
)

VIEW_HELP = """\
[bold cyan].view [PATH][/bold cyan] shows a NIfTI image, or the last result,
in a browser.

Without a path, the last result is converted into an image as .save
converts it, with its options --value, --voxel and --mm. The viewer is an
HTML page, saved to a temporary file and opened in the browser when the
REPL runs on a terminal; it needs no network. It shows the image over the
MNI152 T1 in three slices (left hemisphere on the left): click or drag to
move the cursor, whose MNI coordinates and value it tells. Its controls
change, live, what these options start with:
  --threshold T|P%      hide the voxels below T, or below the P-th
                        percentile of the nonzero values (default: show
                        every nonzero voxel)
  --cmap NAME           the colormap, of matplotlib or nilearn (default:
                        RdBu_r)
  --symmetric           a colormap symmetric around 0, hiding |v| < T
  --one-sided           a colormap from the threshold to the maximum
Without either, the colormap is symmetric when the image has negative
values. The page also sets the opacity of the image."""


HELP_TOPICS["view"] = VIEW_HELP

# The options of .view and the number of values each takes; without a
# path, the options of .save tell how to convert the last result.
_VIEW_OPTIONS = {
    "--threshold": 1, "--cmap": 1, "--symmetric": 0, "--one-sided": 0,
}


class ViewRequest(NamedTuple):
    """
    What ``.view`` shows: the image *path*, or the last result if None.

    *save* tells how to convert the last result, as ``.save`` would;
    *symmetric* is None when the image decides (see :func:`symmetry`).
    """

    path: Optional[Path]
    save: SaveRequest
    threshold: object = VIEW_THRESHOLD
    cmap: Optional[str] = None
    symmetric: Optional[bool] = None


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
    conversion = {k: v for k, v in options.items() if k in _SAVE_OPTIONS}
    if paths and conversion:
        raise SaveError(
            f"{', '.join(conversion)}: options to view the last result, "
            "not a file."
        )
    return ViewRequest(
        Path(paths[0]).expanduser() if paths else None,
        _save_request(Path("last.nii.gz"), conversion),
        view_threshold(options.get("--threshold", [None])[0]),
        options.get("--cmap", [None])[0], _symmetry_option(options),
    )


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


def symmetry(console, path: Path, symmetric: Optional[bool]) -> bool:
    """
    Return whether the colormap of the image *path* is symmetric.

    Unless *symmetric* says, it is when the image has negative values;
    *console* is then told which was chosen and how to change it.
    """
    if symmetric is not None:
        return symmetric
    signed = bool((np.asanyarray(nib.load(path).dataobj) < 0).any())
    console.print(
        "Colormap: symmetric, the image has negative values (--one-sided "
        "for a one-sided one)." if signed else
        "Colormap: one-sided, the image has no negative values "
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


def _viewer_page(app, path: Path, request: ViewRequest, title: str) -> str:
    """
    Return the viewer page, entitled *title*, of the image *path*.

    It starts as *request* asks. The image is drawn over the MNI152 T1
    template of nilearn, whatever the engine's template, a mask, is;
    nilearn's warnings are kept off the prompt.
    """
    _check_cmap(request.cmap)
    symmetric = symmetry(app.console, path, request.symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return brain_viewer.viewer_html(
            nib.load(path), brain_viewer.mni_background(),
            title=title, threshold=request.threshold, cmap=request.cmap,
            symmetric=symmetric,
        )


def _view_nifti(app, path: Path, request: ViewRequest, title: str) -> None:
    """
    Show the NIfTI image *path* in the brain viewer, as *request* asks.

    The viewer page, entitled *title*, is saved as an HTML file whose
    path is printed, and opened in a browser when the REPL runs on a
    terminal.
    """
    try:
        html = _html_file(_viewer_page(app, path, request, title))
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


def _save_frame(app, df: pd.DataFrame, request: SaveRequest) -> None:
    """Save *df* as *request* asks: NIfTI for ``.nii(.gz)``, else a table."""
    try:
        if is_nifti_path(request.path):
            _save_nifti(app, df, request)
        else:
            _save_table(app.console, df, request.path)
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
        _save_frame(app, block.frame, request._replace(path=path))


def last_result_image(app, request: SaveRequest) -> Optional[Path]:
    """
    Save the last result as a NIfTI image in a temporary file, its path.

    The result, the last ``obtain`` block of a SQUALL program with
    several, is converted as ``.save`` converts it, as *request* asks.
    Return None, and tell why, if there is no result or it has no voxels.
    """
    if not app.results:
        app.console.print(
            "[yellow]Nothing to view — run a query first, or give a NIfTI "
            "file.[/yellow]"
        )
        return None
    try:
        template = engine_template(app._nl)
        saved = result_to_nifti(
            app.results[-1].frame, template,
            _engine_sorts(app.settings.engine_name), request,
        )
    except Exception as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return None
    fd, name = tempfile.mkstemp(suffix=".nii.gz", prefix="neurolang_view_")
    os.close(fd)
    nib.save(saved.image, name)
    app.console.print(
        saved_image_message(Path(name), saved, template.shape),
        style="green", markup=False,
    )
    return Path(name)


@command(
    ".view", "Show a NIfTI image, or the last result, in an interactive "
    "viewer in the browser (.help view)", VIEW_USAGE,
)
def cmd_view(app, arg: str) -> None:
    """
    Show the NIfTI image of *arg*, or the last result, in the brain viewer.

    Without a path, the last result is converted as ``.save`` would, its
    options included, into a temporary ``.nii.gz`` file.
    """
    try:
        request = parse_view_args(arg)
    except SaveError as exc:
        app.console.print(f"View failed: {exc}", style="red", markup=False)
        return
    path = request.path or last_result_image(app, request.save)
    if path is None:
        return
    if not path.is_file():
        app.console.print(
            f"File not found: {path}", style="red", markup=False
        )
        return
    _view_nifti(app, path, request, _view_title(app, request, path))


def _view_title(app, request: ViewRequest, path: Path) -> str:
    """Return the title of the viewer: the image's name, or the result's."""
    if request.path is not None:
        return path.name
    block = app.results[-1].title
    return f"Last result: {block}" if block else "Last result"
