"""
An interactive viewer of a brain image, as a self-contained HTML page.

:func:`viewer_html` writes a page showing a NIfTI image over a background,
the MNI152 T1 of nilearn (:func:`mni_background`), in three orthogonal
slices with a crosshair. Its controls change live, in the browser, the
threshold (a value or a percentile), the colormap, whether the colormap
is symmetric and the opacity of the image; the page shows the MNI
coordinates and the value of the voxel under the cursor.

The page needs no network: its script (``static/brain_viewer.js``) and
its data are inlined, the data as a JSON document whose arrays are bytes
deflated with zlib then base64-encoded, which the page inflates with the
browser's ``DecompressionStream``. Volumes are flattened in Fortran order
(the first index varies fastest). The image is first reoriented to the
closest RAS+ orientation, so the slices are drawn as nilearn draws them:
neurological, the left hemisphere on the left.
"""

import base64
import html
import json
import re
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

# The colormap of nilearn's viewer, used when none is asked for.
DEFAULT_CMAP = "RdBu_r"

# The colormaps the page offers, from matplotlib and nilearn.
COLORMAPS = (
    "hot", "inferno", "viridis", "plasma", "magma", "cividis", "cold_hot",
    "RdBu_r", "bwr", "coolwarm", "gray",
)

# The largest background embedded, in voxels: the 1 mm MNI grid fits.
BACKGROUND_MAX_VOXELS = 2 ** 24

_STATIC = Path(__file__).parent / "static"

# The little-endian types of the arrays, as the page reads them.
_DTYPES = {"float16": "<f2", "float32": "<f4"}


def encode_array(array: np.ndarray) -> str:
    """Return the bytes of *array*, deflated then base64-encoded."""
    raw = np.ascontiguousarray(array).tobytes()
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _value_type(values: np.ndarray) -> str:
    """Return ``float16`` if it holds the *values* exactly, else float32."""
    half = values.astype(np.float16).astype(np.float32)
    return "float16" if np.array_equal(half, values) else "float32"


def encode_overlay(data: np.ndarray) -> dict:
    """
    Return the voxel values of the volume *data* as the page reads them.

    They are sparse, the flat indices (uint32) of the nonzero voxels and
    their values, or dense, the values of every voxel, whichever is
    smaller; the values are half floats when that keeps them exactly.
    NaN stands for no value, as 0.
    """
    flat = np.nan_to_num(
        np.asarray(data, dtype=np.float32), nan=0.0
    ).ravel(order="F")
    index = np.flatnonzero(flat)
    dtype = _value_type(flat[index])
    itemsize = np.dtype(_DTYPES[dtype]).itemsize
    overlay = {"dtype": dtype, "count": int(index.size)}
    if index.size * (4 + itemsize) < flat.size * itemsize:
        overlay.update(
            encoding="sparse", index=encode_array(index.astype("<u4")),
            values=encode_array(flat[index].astype(_DTYPES[dtype])),
        )
    else:
        overlay.update(
            encoding="dense", values=encode_array(flat.astype(_DTYPES[dtype]))
        )
    return overlay


def _colormap(name: str):
    """Return the matplotlib or nilearn colormap *name*, or None."""
    import matplotlib

    if name in matplotlib.colormaps:
        return matplotlib.colormaps[name]
    try:
        from nilearn.plotting import cm
    except ImportError:
        return None
    return getattr(cm, "_cmap_d", {}).get(name)


def colormap_luts(extra=()) -> dict:
    """
    Return the colours of the colormaps, 256 RGB uint8 each, by name.

    They are those of :data:`COLORMAPS` and the *extra* ones that
    matplotlib or nilearn know, encoded as :func:`encode_array` does.
    """
    luts = {}
    for name in dict.fromkeys((*COLORMAPS, *extra)):
        cmap = _colormap(name)
        if cmap is not None:
            rgb = cmap(np.linspace(0, 1, 256))[:, :3]
            luts[name] = encode_array(np.rint(rgb * 255).astype(np.uint8))
    return luts


def mni_background():
    """
    Return the MNI152 T1 template packaged with nilearn, or None.

    It ships with nilearn: nothing is downloaded.
    """
    try:
        from nilearn.datasets import load_mni152_template
    except ImportError:
        return None
    return load_mni152_template(resolution=1)


def _background_factor(shape, max_voxels: int) -> int:
    """Return the smallest subsampling of *shape* within *max_voxels*."""
    factor = 1
    while np.prod([-(-n // factor) for n in shape]) > max_voxels:
        factor += 1
    return factor


def _to_uint8(volume: np.ndarray) -> np.ndarray:
    """Return *volume* scaled from 0 to its maximum into 0 to 255."""
    volume = np.clip(np.nan_to_num(volume), 0, None)
    top = volume.max() if volume.size else 0
    if top <= 0:
        return np.zeros(volume.shape, dtype=np.uint8)
    return np.rint(volume / top * 255).astype(np.uint8)


def resample_background(
    background, shape, affine: np.ndarray,
    max_voxels: int = BACKGROUND_MAX_VOXELS,
) -> tuple:
    """
    Return the image *background* on the grid (*shape*, *affine*).

    The grid is subsampled by the smallest integer factor that keeps it
    within *max_voxels*, a background voxel covering factor³ voxels of
    the grid. Return the volume, as uint8 from 0 to its maximum, and the
    factor.
    """
    from scipy import ndimage

    factor = _background_factor(shape[:3], max_voxels)
    target_shape = tuple(-(-n // factor) for n in shape[:3])
    subsample = np.diag([factor] * 3 + [1.0])
    subsample[:3, 3] = (factor - 1) / 2
    mapping = np.linalg.inv(background.affine) @ affine @ subsample
    source = np.asarray(background.dataobj, dtype=np.float32)
    volume = ndimage.affine_transform(
        source.reshape(source.shape[:3] + (-1,))[..., 0],
        mapping[:3, :3], mapping[:3, 3], output_shape=target_shape,
        order=1, mode="constant", cval=0.0,
    )
    return _to_uint8(volume), factor


def initial_cursor(data: np.ndarray) -> list:
    """
    Return the voxel the cursor starts at: a peak of ``|data|``.

    Of the voxels of largest absolute value, it is the closest to their
    centre; the centre of the grid when *data* is empty.
    """
    magnitude = np.abs(np.nan_to_num(data))
    if not magnitude.any():
        return [int(n // 2) for n in data.shape]
    peaks = np.argwhere(magnitude == magnitude.max())
    distances = ((peaks - peaks.mean(axis=0)) ** 2).sum(axis=1)
    return [int(v) for v in peaks[np.argmin(distances)]]


def _threshold_state(threshold) -> dict:
    """Return the threshold of the page: a value, or ``P%`` a percentile."""
    if isinstance(threshold, str) and threshold.endswith("%"):
        return {"threshold": float(threshold[:-1]),
                "thresholdMode": "percentile"}
    return {"threshold": float(threshold), "thresholdMode": "value"}


def _volume(img) -> tuple:
    """
    Return *img* reoriented to RAS+: its first volume and its affine.

    The affine maps the voxels of the volume to millimetres.
    """
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj, dtype=np.float32)
    return data.reshape(data.shape[:3] + (-1,))[..., 0], canonical.affine


def _background_state(background, shape, affine):
    """Return the *background* resampled for the page, None without one."""
    if background is None:
        return None
    volume, factor = resample_background(background, shape, affine)
    return {
        "shape": list(volume.shape), "factor": factor,
        "data": encode_array(volume.ravel(order="F")),
    }


def viewer_state(
    img, background, *, title, threshold, cmap, symmetric
) -> dict:
    """
    Return the data of the viewer page of *img*, as a JSON-able dict.

    See :func:`viewer_html`; ``toImageVoxel`` maps the voxels shown, in
    RAS+ orientation, to those of *img*.
    """
    data, affine = _volume(img)
    cmap = cmap or DEFAULT_CMAP
    return {
        "title": title, "shape": list(data.shape), "affine": affine.tolist(),
        "toImageVoxel": (np.linalg.inv(img.affine) @ affine).tolist(),
        "overlay": encode_overlay(data),
        "background": _background_state(background, data.shape, affine),
        "colormaps": colormap_luts([cmap]),
        "initial": {
            **_threshold_state(threshold), "cmap": cmap,
            "symmetric": bool(symmetric), "opacity": 1.0,
            "cursor": initial_cursor(data),
        },
    }


def _script_json(state: dict) -> str:
    """Return *state* as JSON that cannot end the script element it is in."""
    return json.dumps(state, separators=(",", ":")).replace("<", "\\u003c")


def viewer_html(
    img, background, *, title: str, threshold, cmap, symmetric: bool
) -> str:
    """
    Return the HTML page of an interactive viewer of the image *img*.

    *background* is drawn under it (e.g. :func:`mni_background`), or
    nothing if None. The page starts with the *threshold*, a value or a
    percentile such as ``"95%"`` of the nonzero (absolute, when
    symmetric) values; the colormap *cmap*, :data:`DEFAULT_CMAP` if None;
    and a colormap *symmetric* around 0 or not. Its controls change them.
    """
    state = viewer_state(
        img, background, title=title, threshold=threshold, cmap=cmap,
        symmetric=symmetric,
    )
    parts = {
        "TITLE": html.escape(title),
        "DATA": _script_json(state),
        "SCRIPT": (_STATIC / "brain_viewer.js").read_text(encoding="utf-8"),
    }
    page = (_STATIC / "brain_viewer.html").read_text(encoding="utf-8")
    return re.sub(
        r"__(TITLE|DATA|SCRIPT)__", lambda m: parts[m.group(1)], page
    )
