"""
Saving and viewing of query results in the interactive REPL.

``.save PATH`` writes the last result as CSV, or as a NIfTI image when
PATH ends with ``.nii``; ``.view PATH.nii`` plots a NIfTI image with
nilearn and opens the picture.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from .commands import command


def _result_to_nifti(
    df: pd.DataFrame, template_path: Optional[str] = None
) -> nib.Nifti1Image:
    """
    Convert a DataFrame with voxel coordinates into a NIfTI image.

    Expects columns named (i, j, k), or else uses the first three numeric
    columns, and an optional value column (the first other column). If
    *template_path* is given the image has that template's affine and
    shape; otherwise an identity-affine 256×256×256 volume is created.
    """
    # TODO(B5): redesign the conversion: use the engine's MNI template
    # affine and shape rather than an identity 256³ volume, and require
    # an explicit value column instead of taking the first non-coordinate
    # one (e.g. study_id) or guessing coordinates from numeric columns.
    i_col, j_col, k_col = _coordinate_columns(df)
    coord_cols = {i_col, j_col, k_col}
    val_cols = [c for c in df.columns if c not in coord_cols]
    if template_path:
        ref = nib.load(template_path)
        shape = ref.shape[:3]
        affine = ref.affine
    else:
        shape = (256, 256, 256)
        affine = np.eye(4)
    data = np.zeros(shape, dtype=np.float32)
    coords = df[[i_col, j_col, k_col]].values.astype(int)
    if val_cols:
        values = df[val_cols[0]].values.astype(float)
    else:
        values = np.ones(len(df), dtype=np.float32)
    inside = np.all((coords >= 0) & (coords < shape), axis=1)
    data[tuple(coords[inside].T)] = values[inside]
    return nib.Nifti1Image(data, affine)


def _coordinate_columns(df: pd.DataFrame) -> tuple:
    """Return the (i, j, k) columns of *df*, or its first numeric ones."""
    col_lower = {str(c).lower(): c for c in df.columns}
    coords = tuple(col_lower.get(name) for name in "ijk")
    if None not in coords:
        return coords
    num_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(num_cols) >= 3:
        return tuple(num_cols[:3])
    raise ValueError(
        "Cannot determine voxel coordinates: no (i, j, k) or numeric "
        "columns found."
    )


def _view_nifti(console, path: str) -> None:
    """Plot the NIfTI image *path* with nilearn and open the picture."""
    try:
        from nilearn import plotting
    except ImportError:
        console.print("[red]nilearn is required for visualisation.[/red]")
        return
    fd, png = tempfile.mkstemp(suffix=".png", prefix="neurolang_view_")
    os.close(fd)
    try:
        display = plotting.plot_img(
            path, display_mode="ortho", cut_coords=None, black_bg=True,
            cmap="cold_hot",
        )
        display.savefig(png, dpi=150)
        display.close()
        console.print(f"[green]Saved visualisation to: {png}[/green]")
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, png])
    except Exception as exc:
        console.print(f"[red]Visualisation failed: {exc}[/red]")


def _save_path(path: Path, title: str, several: bool) -> Path:
    """Return where to save the block *title* of a result in *path*."""
    if not several:
        return path
    return path.with_name(f"{path.stem}_{title}{path.suffix}")


def _save_frame(console, df: pd.DataFrame, path: Path) -> None:
    """Save *df* in *path*, as NIfTI if it ends with ``.nii``, else CSV."""
    try:
        if path.suffix == ".nii":
            nib.save(_result_to_nifti(df), path)
            console.print(f"[green]NIfTI saved to: {path}[/green]")
        else:
            df.to_csv(path, index=False)
            console.print(f"[green]CSV saved to: {path}[/green]")
    except Exception as exc:
        console.print(f"[red]Save failed: {exc}[/red]")


@command(
    ".save", "Save the last result as CSV, or as NIfTI for a .nii path",
    ".save PATH[.nii]",
)
def cmd_save(app, arg: str) -> None:
    """
    Save the last result in the file *arg*.

    With several ``obtain`` blocks, each is saved in its own file, named
    after the block (``PATH_NAME.csv``).
    """
    if not app.results:
        app.console.print(
            "[yellow]Nothing to save — run a query first.[/yellow]"
        )
        return
    if not arg.strip():
        app.console.print(
            "[yellow]Usage: .save <path>  or  .save <path>.nii[/yellow]"
        )
        return
    path = Path(arg.strip()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    several = len(app.results) > 1
    for block in app.results:
        target = _save_path(path, block.title, several)
        _save_frame(app.console, block.frame, target)


@command(".view", "Plot a NIfTI image with nilearn", ".view PATH.nii")
def cmd_view(app, arg: str) -> None:
    """Plot the NIfTI image *arg* and open the picture."""
    path = os.path.expanduser(arg.strip())
    if not path:
        app.console.print("[yellow]Usage: .view <path>.nii[/yellow]")
    elif not os.path.isfile(path):
        app.console.print(f"[red]File not found: {path}[/red]")
    else:
        _view_nifti(app.console, path)
