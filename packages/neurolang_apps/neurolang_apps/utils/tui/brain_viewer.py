"""
An interactive viewer of brain images, as a self-contained HTML page.

:func:`viewer_html` writes a page showing NIfTI images on one grid, its
overlays, over a background, the MNI152 T1 of nilearn
(:func:`mni_background`), with NiiVue [niivue]_: three orthogonal slices
with a crosshair, drawn with WebGL2, which a click or a drag moves and
the mouse wheel scrolls. Its controls change live, in the browser, the
threshold (a value or a percentile), the colormap, whether the colormap
is symmetric, the opacity and the visibility of each overlay, the one
they apply to being chosen in a list, and the order the overlays are
drawn in, the later ones over the earlier ones. Other controls set
NiiVue's display: the view (an axial, coronal or sagittal slice, the
three slices, in a row, a column or a grid, with or without the 3D
render, or the 3D render alone), what a drag with the right button does
(contrast, pan and zoom, measure, angle, 3D slicer), the colour, width
and 3D drawing of the crosshair, the gamma of the background, NiiVue's
colour bars, labels, orientation cube and ruler, the radiological
convention, the interpolation of the voxels and a clip plane of the 3D
render; NiiVue's hot keys, V and C on the views, cycle the views and the
clip planes. The page shows the MNI coordinates of the voxel under the
cursor, the value of each overlay there, and a colour bar per overlay.
It also draws a network (:func:`connectome_state`) or streamlines
(:func:`streamlines_state`), in the 3D render and on the slices near
them.

The page needs no network: NiiVue (``static/niivue/niivue.umd.js``, a
copy of its release :data:`NIIVUE_VERSION`), the script of the page
(``static/brain_viewer.js``), its data and NeuroLang's favicon
(``static/favicon.svg``) are inlined. The data is a JSON
document holding the overlays and the background as gzipped NIfTI
images, base64-encoded, and the colormaps as 256 RGB colours, deflated
with zlib then base64-encoded, which the page inflates with the browser's
``DecompressionStream``; streamlines are a TrackVis ``.trk`` file,
encoded as the colormaps. NiiVue reorients the images itself, and draws
them as nilearn does: neurological, the left hemisphere on the left.

.. [niivue] https://github.com/niivue/niivue, BSD 2-Clause License.
"""

import base64
import gzip
import html
import io
import json
import re
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

# The release of NiiVue the page inlines.
NIIVUE_VERSION = "0.69.0"

# The colormap used when none is asked for.
DEFAULT_CMAP = "plasma"

# The colormaps of the overlays after the first, in turn: the first has
# the colormap asked for, :data:`DEFAULT_CMAP` by default.
OVERLAY_CMAPS = ("viridis", "cool", "hot", "cividis", "magma")

# The colormaps the page offers, from matplotlib and nilearn.
COLORMAPS = (
    "hot", "inferno", "viridis", "plasma", "magma", "cividis", "cold_hot",
    "RdBu_r", "bwr", "coolwarm", "gray",
)

# The largest background embedded, in voxels: the 1 mm MNI grid fits.
BACKGROUND_MAX_VOXELS = 2 ** 24

_STATIC = Path(__file__).parent / "static"


def frame_hooks() -> str:
    """Return the script that lets a page in a frame of the REPL take its
    theme and say how tall it is, to inline in the page."""
    return (_STATIC / "frame_hooks.js").read_text(encoding="utf-8")


def encode_array(array: np.ndarray) -> str:
    """Return the bytes of *array*, deflated then base64-encoded."""
    raw = np.ascontiguousarray(array).tobytes()
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def encode_nifti(data: np.ndarray, affine: np.ndarray) -> str:
    """Return the NIfTI image of *data* and *affine*, gzipped then
    base64-encoded, as NiiVue reads it."""
    image = nib.Nifti1Image(np.asarray(data), affine)
    return base64.b64encode(
        gzip.compress(image.to_bytes(), compresslevel=9)
    ).decode("ascii")


def overlay_values(data: np.ndarray) -> np.ndarray:
    """
    Return the values of the volume *data* the page draws, as float32.

    NaN stands for no value, as 0; an infinite value is drawn at the end
    of the colour scale, as the largest finite one of its sign.
    """
    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(data.shape[:3] + (-1,))[..., 0]
    finite = data[np.isfinite(data)]
    return np.nan_to_num(
        data, nan=0.0,
        posinf=max(finite.max(), 0.0) if finite.size else 0.0,
        neginf=min(finite.min(), 0.0) if finite.size else 0.0,
    )


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


def _background_state(background, shape, affine) -> dict:
    """
    Return the *background* on the grid (*shape*, *affine*) of the
    overlays, subsampled to stay within :data:`BACKGROUND_MAX_VOXELS`, as
    the page reads it, zeros without one: NiiVue draws the overlays
    over a background.
    """
    if background is None:
        volume, factor = np.zeros(shape[:3], dtype=np.uint8), 1
    else:
        volume, factor = resample_background(background, shape, affine)
    subsample = np.diag([factor] * 3 + [1.0])
    subsample[:3, 3] = (factor - 1) / 2
    return {
        "nifti": encode_nifti(volume, affine @ subsample),
        "factor": factor,
    }


def _layers(img) -> list:
    """
    Return the overlays *img* stands for, as (name, image) pairs.

    *img* is an image, an overlay without a name, or a sequence of
    (name, image) pairs on one grid.
    """
    if isinstance(img, nib.spatialimages.SpatialImage):
        return [("", img)]
    layers = list(img)
    first = layers[0][1]
    for name, other in layers[1:]:
        if other.shape[:3] != first.shape[:3] or not np.allclose(
            other.affine, first.affine
        ):
            raise ValueError(
                f"The overlay {name!r} is not on the grid of the first one."
            )
    return layers


def _overlay_cmaps(cmap, count: int) -> list:
    """
    Return the colormaps of *count* overlays: *cmap* for the first, then
    those of :data:`OVERLAY_CMAPS` other than it, in turn.
    """
    first = cmap or DEFAULT_CMAP
    others = [name for name in OVERLAY_CMAPS if name != first]
    return [first] + [others[n % len(others)] for n in range(count - 1)]


def viewer_state(
    img, background, *, title, threshold, cmap, symmetric
) -> dict:
    """
    Return the data of the viewer page of *img*, as a JSON-able dict.

    See :func:`viewer_html`. The ``affine`` maps the voxels of the
    images, in which the ``cursor`` is, to millimetres.
    """
    layers = _layers(img)
    first = layers[0][1]
    volumes = [overlay_values(image.dataobj) for _, image in layers]
    affine = first.affine
    cmaps = _overlay_cmaps(cmap, len(layers))
    if isinstance(symmetric, (bool, np.bool_)):
        symmetric = [symmetric] * len(layers)
    return {
        "title": title, "shape": list(volumes[0].shape),
        "affine": affine.tolist(),
        "overlays": [
            {"name": name, "nifti": encode_nifti(volume, affine)}
            for (name, _), volume in zip(layers, volumes)
        ],
        "background": _background_state(
            background, volumes[0].shape, affine
        ),
        "colormaps": colormap_luts(cmaps),
        "initial": {
            "cursor": initial_cursor(volumes[0]), "active": 0,
            "layers": [
                {
                    **_threshold_state(threshold), "cmap": layer_cmap,
                    "symmetric": bool(layer_symmetric), "opacity": 1.0,
                    "visible": True,
                }
                for layer_cmap, layer_symmetric in zip(cmaps, symmetric)
            ],
        },
    }


# The radius, in millimetres, of the largest node and of the thickest edge
# of a network.
NODE_RADIUS = 4.0
EDGE_RADIUS = 1.5

# The opacity the overlays of a network or of streamlines start with, for
# its edges or them to show on the slices.
NETWORK_OVERLAY_OPACITY = 0.5

# The number of edges a percentile threshold shows at least, all of them
# when there are fewer.
MIN_SHOWN_EDGES = 20

# The number of edges, the largest, given to the page to threshold there.
MAX_EDGES = 20000


def edge_threshold(values, threshold, at_least: int = 0) -> float:
    """
    Return the smallest |value| of the edges shown: *threshold*, a value,
    or the percentile ``"P%"`` of the |*values*|, lowered to show the
    *at_least* largest.
    """
    magnitudes = np.sort(np.abs(np.asarray(values, dtype=float)))[::-1]
    if not (isinstance(threshold, str) and threshold.endswith("%")):
        return float(threshold)
    if not magnitudes.size:
        return 0.0
    minimum = float(np.percentile(magnitudes, float(threshold[:-1])))
    if at_least:
        minimum = min(minimum, magnitudes[:at_least][-1])
    return float(minimum)


def connectome_state(names, coordinates, edges, threshold) -> dict:
    """
    Return the network of *edges* for the page to draw, a JSON-able dict.

    The nodes have *names* and the millimetre *coordinates* of their
    centre; an edge ``(first, second, value)`` joins two of them, by
    index. The page draws the edges of |value| at least its edge
    threshold, which starts at *threshold* (:func:`edge_threshold`; a
    percentile shows the :data:`MIN_SHOWN_EDGES` largest at least) and
    can be changed there, and the nodes they join: the colour and the
    size of a node is its strength, the sum of the values of its edges
    drawn, and the colour and the thickness of an edge its value, from
    the threshold to the largest. The :data:`MAX_EDGES` largest edges
    are given to the page.
    """
    edges = sorted(
        (e for e in edges if e[2] != 0 and not np.isnan(e[2])),
        key=lambda e: -abs(e[2]),
    )[:MAX_EDGES]
    return {
        "name": "network",
        "threshold": edge_threshold(
            [w for _, _, w in edges], threshold, MIN_SHOWN_EDGES
        ),
        "nodeRadius": NODE_RADIUS, "edgeRadius": EDGE_RADIUS,
        "nodes": [
            {
                "name": str(name), "x": float(x), "y": float(y),
                "z": float(z),
            }
            for name, (x, y, z) in zip(names, coordinates)
        ],
        "edges": [
            {"first": int(a), "second": int(b), "value": float(w)}
            for a, b, w in edges
        ],
    }


def shown_edges(connectome: dict) -> list:
    """Return the edges of *connectome* the page draws at first."""
    return [
        e for e in connectome["edges"]
        if abs(e["value"]) >= connectome["threshold"]
    ]


# The number of streamlines drawn at most, evenly subsampled.
MAX_STREAMLINES = 20000


def shown_streamlines(ids, cap=None) -> list:
    """Return the *ids* drawn: all of them, or *cap* evenly spaced,
    :data:`MAX_STREAMLINES` by default."""
    ids = list(ids)
    cap = MAX_STREAMLINES if cap is None else cap
    if len(ids) <= cap:
        return ids
    return [ids[n] for n in np.linspace(0, len(ids) - 1, cap).round()
            .astype(int)]


def trk_bytes(lines) -> bytes:
    """
    Return the TrackVis ``.trk`` file of the streamlines *lines*, (n, 3)
    arrays of points in millimetres (RAS+).

    NiiVue 0.69.0 reads a ``.tck`` without its first streamline (its
    ``readTCK`` misses the offset 0), a ``.trk`` whole.
    """
    tractogram = nib.streamlines.Tractogram(
        [np.asarray(line, dtype=np.float32) for line in lines],
        affine_to_rasmm=np.eye(4),
    )
    file = io.BytesIO()
    nib.streamlines.TrkFile(tractogram).save(file)
    return file.getvalue()


def streamlines_state(lines, total: int) -> dict:
    """
    Return the streamlines *lines* for the page to draw, a JSON-able dict.

    *lines* are (n, 3) arrays of points in millimetres, those drawn of
    the *total* of the result; they go in the page as a ``.trk`` file
    (:func:`trk_bytes`), encoded as :func:`encode_array` does.
    """
    return {
        "trk": encode_array(np.frombuffer(trk_bytes(lines), np.uint8)),
        "shown": len(lines), "total": int(total),
    }


def regions_state(regions) -> dict:
    """
    Return the regions of a result (a :class:`~.nifti.RegionLayer`) as the
    page reads them: the image of their numbers, which names the one under
    the cursor, and that of their colours, RGBA, transparent elsewhere,
    which the page draws.
    """
    numbers = np.asanyarray(regions.image.dataobj)
    table = np.zeros(len(regions.names) + 1, dtype=RGBA)
    for n, colour in enumerate(regions.colours, 1):
        table[n] = (*bytes.fromhex(colour[1:7]), 255)
    return {
        "nifti": encode_nifti(numbers, regions.image.affine),
        "colours": encode_nifti(table[numbers], regions.image.affine),
        "names": [str(name) for name in regions.names],
        "coloured": bool(regions.coloured),
    }


# The voxels of an RGBA image, as NIfTI stores them (its datatype 2304).
RGBA = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1"), ("A", "u1")])


def _script_json(state: dict) -> str:
    """Return *state* as JSON that cannot end the script element it is in."""
    return json.dumps(state, separators=(",", ":")).replace("<", "\\u003c")


def viewer_html(
    img, background, *, title: str, threshold, cmap, symmetric: bool,
    connectome=None, streamlines=None, space: str = "MNI", regions=None,
) -> str:
    """
    Return the HTML page of an interactive viewer of the image *img*.

    *img* is an image, or a sequence of (name, image) pairs on one grid,
    each an overlay. *background* is drawn under them (e.g.
    :func:`mni_background`), or nothing if None. Each overlay starts with
    the *threshold*, a value or a percentile such as ``"95%"`` of its
    nonzero (absolute, when symmetric) values; the first with the
    colormap *cmap*, :data:`DEFAULT_CMAP` if None, the others with those
    of :data:`OVERLAY_CMAPS`; and with a colormap symmetric around 0 or
    not, as *symmetric*, a bool or one per overlay, tells. Its controls
    change them. A *connectome*, a network of :func:`connectome_state`,
    is drawn in the 3D render, which the page then starts with, beside
    the slices, and its strongest edges are listed. *streamlines*, of
    :func:`streamlines_state`, are drawn so too, coloured by their
    direction, with a control of their visibility and opacity, on
    slices cut in millimetres, as they are drawn. *space* names the
    millimetres of the cursor: ``MNI``, or ``RAS`` over the background of
    a subject. *regions*, a :class:`~.nifti.RegionLayer`, are the regions
    of a result: the readout names the one under the cursor, and a
    control shows them in their colours, those of their atlas.
    """
    state = viewer_state(
        img, background, title=title, threshold=threshold, cmap=cmap,
        symmetric=symmetric,
    )
    state["space"] = space
    if streamlines is not None:
        state["streamlines"] = streamlines
    if connectome is not None:
        state["connectome"] = connectome
    if regions is not None:
        state["regions"] = regions_state(regions)
    if connectome is not None or streamlines is not None:
        state["initial"]["display"] = {"renderTile": True, "layout": "grid"}
        for layer in state["initial"]["layers"]:
            layer["opacity"] = NETWORK_OVERLAY_OPACITY
    parts = {
        "TITLE": html.escape(title),
        "DATA": _script_json(state),
        "NIIVUE": (_STATIC / "niivue" / "niivue.umd.js").read_text(
            encoding="utf-8"
        ),
        "SCRIPT": (_STATIC / "brain_viewer.js").read_text(encoding="utf-8"),
        "FRAME_HOOKS": frame_hooks(),
        "FAVICON": base64.b64encode(
            (_STATIC / "favicon.svg").read_bytes()
        ).decode("ascii"),
    }
    page = (_STATIC / "brain_viewer.html").read_text(encoding="utf-8")
    return re.sub(
        r"__(TITLE|DATA|NIIVUE|SCRIPT|FRAME_HOOKS|FAVICON)__",
        lambda m: parts[m.group(1)], page,
    )
