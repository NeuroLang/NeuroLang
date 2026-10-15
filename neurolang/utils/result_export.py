"""
Saving query results as tables or brain images, and viewing the images.

A result is saved as a table (:func:`save_table`: CSV, TSV or JSON) or
as a NIfTI brain image (:func:`result_image`), chosen by the extension
of the file. A result is an image when it has

* a region column (:class:`~neurolang.regions.ExplicitVBR`, or the
  list of its voxels as an engine stores an atlas region, whose grid
  *grid_of* tells): one region is its image (an overlay, e.g. a
  probability map, gives its values); several regions give a label
  image, the region of row *n* having the value *n* + 1, described by a
  table of labels;
* or the voxel columns ``i``, ``j``, ``k`` of the engine's MNI grid and
  at most one other column, numeric, giving the value of each voxel
  (1 without it). Voxels outside the grid, such as peaks reported
  outside the template's box, are left out.

:func:`view_image` shows an image in the browser, with nilearn.
"""

from pathlib import Path
from typing import Callable, NamedTuple, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.regions import ExplicitVBR, ExplicitVBROverlay

VOXEL_COLUMNS = ("i", "j", "k")
TABLE_SUFFIXES = (".csv", ".tsv", ".json")
IMAGE_SUFFIXES = (".nii",)


class ExportError(ValueError):
    """Raised when a result cannot be saved or viewed as asked."""


class ResultImage(NamedTuple):
    """
    The brain image of a result, with the *labels* of its values and the
    number of voxels left *outside* its grid.
    """

    image: nib.Nifti1Image
    labels: Optional[pd.DataFrame] = None
    outside: int = 0


def _suffix(path) -> str:
    """Return the extension of *path*, without ``.gz``."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def is_image_path(path) -> bool:
    """Whether *path* names a NIfTI image (``.nii`` or ``.nii.gz``)."""
    return _suffix(path) in IMAGE_SUFFIXES


def check_path(path) -> None:
    """Raise :class:`ExportError` if *path* is neither a table nor image."""
    if _suffix(path) not in TABLE_SUFFIXES + IMAGE_SUFFIXES:
        raise ExportError(
            f"cannot tell the format of {path}: use .csv, .tsv or .json "
            "(optionally .gz) for a table, or .nii or .nii.gz for an image."
        )


def _is_voxel_list(value) -> bool:
    """Whether *value* is a region stored as the tuple of its voxels."""
    return (
        isinstance(value, tuple) and len(value) > 0
        and np.shape(value[0]) == (3,)
    )


def _is_region(value) -> bool:
    return isinstance(value, ExplicitVBR) or _is_voxel_list(value)


def _describe(value):
    """Return *value* as a table shows it: regions by their size."""
    if isinstance(value, ExplicitVBR):
        return f"<region of {len(value.voxels)} voxels>"
    if _is_voxel_list(value):
        return f"<region of {len(value)} voxels>"
    return value


def save_table(df: pd.DataFrame, path) -> None:
    """Write *df* at *path*, as CSV, TSV or JSON after its extension."""
    df = df.apply(lambda column: column.map(_describe)) if len(df) else df
    suffix = _suffix(path)
    if suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        df.to_csv(path, index=False, sep="\t" if suffix == ".tsv" else ",")


def region_columns(df: pd.DataFrame) -> list:
    """Return the columns of *df* holding regions."""
    if df.empty:
        return []
    return [c for c in df.columns if df[c].map(_is_region).all()]


def result_image(
    df: pd.DataFrame, mask=None, grid_of: Optional[Callable] = None
) -> ResultImage:
    """
    Return the brain image of the result *df*.

    *mask* is the image whose grid the voxel columns index (the engine's
    MNI template); it is needed only for them. *grid_of* returns the
    affine and the shape of the grid of a region stored as its voxels,
    or None when it is unknown.
    """
    if df.empty:
        raise ExportError("the result is empty: there is no image to make.")
    regions = region_columns(df)
    if len(regions) > 1:
        raise ExportError(
            f"the result has several region columns ({', '.join(regions)}):"
            " keep one in the query to make its image."
        )
    if regions:
        return _regions_image(df, regions[0], grid_of)
    if all(column in df.columns for column in VOXEL_COLUMNS):
        return _voxels_image(df, mask)
    raise ExportError(
        "the result has no region column and no voxel columns i, j, k: "
        "save it as a table (.csv) instead."
    )


def _nifti(image) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine)


def _explicit_region(value, grid_of) -> ExplicitVBR:
    """Return the region *value*, given as a region or its voxels."""
    if isinstance(value, ExplicitVBR):
        return value
    grid = grid_of(value) if grid_of else None
    if grid is None:
        raise ExportError(
            "the regions of the result are lists of voxels whose grid is "
            "unknown: build them in the query with agg_create_region(i, j, "
            "k), or save the result as a table (.csv)."
        )
    affine, shape = grid
    return ExplicitVBR(np.asarray(value, dtype=int), affine, image_dim=shape)


def _regions_image(df: pd.DataFrame, column: str, grid_of) -> ResultImage:
    regions = [_explicit_region(value, grid_of) for value in df[column]]
    if len(regions) == 1:
        return ResultImage(_nifti(regions[0].spatial_image()))
    if any(isinstance(r, ExplicitVBROverlay) for r in regions):
        raise ExportError(
            f"the result has {len(regions)} overlays: select one in the "
            "query to make its image."
        )
    first = regions[0]
    for region in regions[1:]:
        if tuple(region.image_dim) != tuple(first.image_dim) or (
            not np.allclose(region.affine, first.affine)
        ):
            raise ExportError(
                "the regions of the result are not on the same grid."
            )
    data = np.zeros(first.image_dim, dtype=np.int32)
    for value, region in enumerate(regions, start=1):
        data[tuple(region.voxels.T)] = value
    labels = df.drop(columns=[column]).reset_index(drop=True)
    labels.insert(0, "value", range(1, len(regions) + 1))
    return ResultImage(nib.Nifti1Image(data, first.affine), labels)


def _voxels_image(df: pd.DataFrame, mask) -> ResultImage:
    if mask is None:
        raise ExportError(
            "the engine has no voxel grid (MNI mask) to place the voxels i, "
            "j, k in."
        )
    others = [c for c in df.columns if c not in VOXEL_COLUMNS]
    if len(others) > 1 or (
        others and not pd.api.types.is_numeric_dtype(df[others[0]])
    ):
        raise ExportError(
            f"the result has the columns {', '.join(others)} besides the "
            "voxel i, j, k: keep at most one, numeric, giving the value of "
            "each voxel."
        )
    voxels = df[list(VOXEL_COLUMNS)].to_numpy()
    if not np.issubdtype(voxels.dtype, np.integer):
        voxels = voxels.astype(float)
        if not np.allclose(voxels, np.round(voxels)):
            raise ExportError("the voxel columns i, j, k are not integers.")
        voxels = np.round(voxels).astype(int)
    shape = mask.shape[:3]
    inside = ((voxels >= 0) & (voxels < np.array(shape))).all(axis=1)
    if not inside.any():
        raise ExportError(
            f"all the voxels of the result are outside the engine's grid "
            f"{shape}."
        )
    data = np.zeros(shape, dtype=np.float32 if others else np.int16)
    data[tuple(voxels[inside].T)] = (
        df[others[0]].to_numpy(dtype=float)[inside] if others else 1
    )
    return ResultImage(
        nib.Nifti1Image(data, mask.affine), outside=int((~inside).sum())
    )


def save_image(result: ResultImage, path) -> list:
    """
    Write the image of *result* at *path*, and its labels beside it.

    Return the paths written: the labels of a label image go to
    ``<name>_labels.csv``.
    """
    nib.save(result.image, str(path))
    written = [Path(path)]
    if result.labels is not None:
        name = Path(path).name
        stem = name[: -len(".nii.gz")] if name.lower().endswith(".nii.gz") \
            else Path(path).stem
        labels_path = Path(path).with_name(f"{stem}_labels.csv")
        save_table(result.labels, labels_path)
        written.append(labels_path)
    return written


def view_image(result: ResultImage, title: str = "") -> None:
    """Open the image of *result* in the browser, on the MNI template."""
    from nilearn import plotting

    data = np.asanyarray(result.image.dataobj)
    labelled = result.labels is not None
    view = plotting.view_img(
        result.image,
        threshold=0.5 if labelled or np.issubdtype(data.dtype, np.integer)
        else 1e-6 * float(np.abs(data).max() or 1),
        symmetric_cmap=False,
        cmap="tab20" if labelled else "hot",
        title=title or None,
    )
    view.open_in_browser()
