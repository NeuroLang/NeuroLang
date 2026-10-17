"""Tests for the interactive brain viewer page that ``.view`` writes."""

import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.utils.tui import brain_viewer  # noqa: E402

AFFINE = np.array([
    [2.0, 0.0, 0.0, -3.0],
    [0.0, 2.0, 0.0, -5.0],
    [0.0, 0.0, 2.0, -7.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _decode(text, dtype):
    """Decode an array of the page: base64 of deflated bytes."""
    return np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype)


def _nifti(item):
    """Return the NIfTI image of an item of the page: base64 of gzip."""
    data = gzip.decompress(base64.b64decode(item["nifti"]))
    return nib.Nifti1Image.from_bytes(data)


def _overlay_volume(overlay, shape=None):
    """Return the volume of an overlay of the page."""
    volume = np.asarray(_nifti(overlay).dataobj)
    assert shape is None or volume.shape == shape
    return volume


def _state(html):
    """Return the state the page embeds as JSON."""
    match = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>',
        html, re.S,
    )
    return json.loads(match.group(1))


def _image(data, affine=AFFINE):
    return nib.Nifti1Image(np.asarray(data), affine)


class TestOverlayEncoding:
    """Check the overlays are embedded as gzipped NIfTI images."""

    def test_round_trip(self):
        data = np.zeros((20, 30, 10), dtype=np.float32)
        data[1, 2, 3], data[19, 29, 9], data[5, 0, 7] = 0.25, -1.5, 3.1
        image = _nifti({"nifti": brain_viewer.encode_nifti(data, AFFINE)})
        assert np.array_equal(np.asarray(image.dataobj), data)
        assert np.allclose(image.affine, AFFINE)

    def test_nan_is_no_value(self):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[0, 0, 0], data[1, 1, 1] = np.nan, 2.0
        volume = brain_viewer.overlay_values(data)
        assert volume[0, 0, 0] == 0 and volume[1, 1, 1] == 2

    def test_infinite_values_take_the_largest_finite_ones(self):
        """An infinite value is drawn at the end of the colour scale."""
        data = np.zeros((2, 2, 2), dtype=np.float32)
        data[0, 0, 0], data[1, 0, 0] = np.inf, 3.0
        data[0, 1, 0], data[1, 1, 0] = -np.inf, -2.0
        volume = brain_viewer.overlay_values(data)
        assert volume[0, 0, 0] == 3.0 and volume[0, 1, 0] == -2.0


class TestColormaps:
    """Check the colormaps are embedded as 256 RGB colours."""

    def test_lut_shapes(self):
        luts = brain_viewer.colormap_luts(["viridis"])
        assert set(brain_viewer.COLORMAPS) <= set(luts)
        for lut in luts.values():
            assert _decode(lut, np.uint8).shape == (256 * 3,)

    def test_luts_follow_matplotlib_and_nilearn(self):
        import matplotlib

        luts = brain_viewer.colormap_luts([])
        hot = _decode(luts["hot"], np.uint8).reshape(256, 3)
        expected = matplotlib.colormaps["hot"](np.linspace(0, 1, 256))
        assert np.abs(hot - expected[:, :3] * 255).max() <= 1
        assert "cold_hot" in luts

    def test_the_requested_colormap_is_added(self):
        luts = brain_viewer.colormap_luts(["Spectral"])
        assert "Spectral" in luts and "hot" in luts

    def test_unknown_colormap_is_left_out(self):
        assert "no_such_map" not in brain_viewer.colormap_luts(["no_such_map"])


class TestBackground:
    """Check the background is resampled to the overlay grid."""

    def test_same_grid(self):
        data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        volume, factor = brain_viewer.resample_background(
            _image(data), (4, 4, 4), AFFINE
        )
        assert factor == 1 and volume.dtype == np.uint8
        assert volume.shape == (4, 4, 4)
        assert volume[3, 3, 3] == 255 and volume[0, 0, 0] == 0
        assert np.all(np.diff(volume.ravel()) >= 0)

    def test_capped_resolution(self):
        volume, factor = brain_viewer.resample_background(
            _image(np.ones((10, 10, 10), dtype=np.float32)), (40, 30, 20),
            AFFINE, max_voxels=2000,
        )
        assert factor == 3 and volume.shape == (14, 10, 7)

    def test_mni_template_on_the_mni_grid(self):
        template = brain_viewer.mni_background()
        assert template is not None and template.shape == (197, 233, 189)
        volume, factor = brain_viewer.resample_background(
            template, template.shape, template.affine
        )
        assert factor == 1 and volume.shape == template.shape
        data = template.get_fdata()
        assert np.abs(volume - data / data.max() * 255).max() <= 1


class TestPage:
    """Check the page holds NiiVue, its controls, its data and no URL."""

    @pytest.fixture
    def html(self):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[1, 2, 3], data[0, 0, 0] = -2.0, 3.0
        return brain_viewer.viewer_html(
            _image(data), _image(np.ones((4, 4, 4), np.float32)),
            title="signed <map>", threshold="95%", cmap="viridis",
            symmetric=True,
        )

    def test_controls(self, html):
        for control in (
            'id="threshold"', 'id="threshold-slider"',
            'id="threshold-mode"', 'id="cmap"', 'id="symmetric"',
            'id="opacity"', 'id="reset"', 'id="colorbar"', 'id="gl"',
            'id="layer"', 'id="layer-up"', 'id="layer-down"',
            'id="visible"', 'id="readout"',
        ):
            assert control in html

    def test_initial_state(self, html):
        state = _state(html)
        assert state["initial"] == {
            "cursor": [0, 0, 0], "active": 0,
            "layers": [{
                "threshold": 95.0, "thresholdMode": "percentile",
                "cmap": "viridis", "symmetric": True, "opacity": 1.0,
                "visible": True,
            }],
        }
        assert [o["name"] for o in state["overlays"]] == [""]
        assert np.array_equal(state["affine"], AFFINE)
        assert "viridis" in state["colormaps"]

    def test_the_overlay_keeps_its_grid(self, html):
        image = _nifti(_state(html)["overlays"][0])
        assert image.shape == (4, 4, 4)
        assert np.allclose(image.affine, AFFINE)
        assert np.asarray(image.dataobj)[1, 2, 3] == -2

    def test_background_on_the_grid_of_the_overlays(self, html):
        background = _nifti(_state(html)["background"])
        assert background.shape == (4, 4, 4)
        assert np.allclose(background.affine, AFFINE)
        assert np.asarray(background.dataobj).dtype == np.uint8

    def test_without_a_background_one_of_zeros(self):
        state = _state(brain_viewer.viewer_html(
            _image(np.ones((2, 2, 2), np.float32)), None, title="t",
            threshold=0.5, cmap=None, symmetric=False,
        ))
        background = np.asarray(_nifti(state["background"]).dataobj)
        assert background.shape == (2, 2, 2) and not background.any()

    def test_value_threshold_and_default_colormap(self):
        html = brain_viewer.viewer_html(
            _image(np.ones((2, 2, 2), np.float32)), None, title="t",
            threshold=0.5, cmap=None, symmetric=False,
        )
        layer = _state(html)["initial"]["layers"][0]
        assert layer["threshold"] == 0.5
        assert layer["thresholdMode"] == "value"
        assert layer["cmap"] == brain_viewer.DEFAULT_CMAP

    def test_default_colormap_is_plasma_and_percentile_threshold(self):
        html = brain_viewer.viewer_html(
            _image(np.ones((2, 2, 2), np.float32)), None, title="t",
            threshold="95%", cmap=None, symmetric=False,
        )
        layer = _state(html)["initial"]["layers"][0]
        assert layer["cmap"] == "plasma"
        assert layer["threshold"] == 95.0
        assert layer["thresholdMode"] == "percentile"

    def test_title_is_escaped(self, html):
        assert "<title>signed &lt;map&gt;</title>" in html

    def test_niivue_is_inlined(self, html):
        bundle = (
            Path(brain_viewer.__file__).parent / "static" / "niivue"
            / "niivue.umd.js"
        ).read_text(encoding="utf-8")
        assert bundle in html
        assert "<script src" not in html
        assert html.count("</script>") == 3

    def test_no_external_urls(self, html):
        """No URL outside the NiiVue bundle, whose comments have some."""
        bundle = (
            Path(brain_viewer.__file__).parent / "static" / "niivue"
            / "niivue.umd.js"
        ).read_text(encoding="utf-8")
        page = html.replace(bundle, "")
        assert not re.search(r"https?://|//cdn|\bsrc=|\bhref=", page)

    def test_cursor_on_the_peak(self):
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[1, 1, 1], data[3, 2, 4] = 1.0, -4.0
        state = _state(brain_viewer.viewer_html(
            _image(data), None, title="t", threshold=0, cmap="hot",
            symmetric=True,
        ))
        assert state["initial"]["cursor"] == [3, 2, 4]

    def test_cursor_in_the_voxels_of_the_image(self):
        """The cursor is in the voxels of the image as it is, NiiVue
        reorienting it to draw it."""
        flip = np.diag([-2.0, 2.0, 2.0, 1.0])
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[0, 1, 2] = 5.0
        state = _state(brain_viewer.viewer_html(
            _image(data, flip), None, title="t", threshold=0,
            cmap="hot", symmetric=False,
        ))
        assert state["initial"]["cursor"] == [0, 1, 2]
        assert np.array_equal(state["affine"], flip)


_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def _chrome():
    """Return the path of a Chrome or Chromium to run headless, or None."""
    names = ("google-chrome", "chromium", "chromium-browser", "chrome")
    found = [os.environ.get("NEUROLANG_TEST_CHROME")]
    found += [shutil.which(name) for name in names]
    found += [p for p in _CHROME_PATHS if Path(p).exists()]
    return next((p for p in found if p), None)


# The flags running WebGL2 in headless Chrome, which NiiVue draws with.
_CHROME_FLAGS = (
    "--headless=new", "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader", "--no-first-run",
    "--host-rules=MAP * ~NOTFOUND",
)


def _run_in_chrome(tmp_path, html, check):
    """Return what the *check* script writes once the page has loaded,
    the page being offline."""
    page = tmp_path / "page.html"
    # Before the last </body>: the NiiVue bundle holds one.
    head, _, tail = html.rpartition("</body>")
    page.write_text(head + check + "</body>" + tail)
    dom = subprocess.run(
        [
            _chrome(), *_CHROME_FLAGS,
            f"--user-data-dir={tmp_path / 'profile'}",
            "--virtual-time-budget=20000", "--dump-dom", page.as_uri(),
        ],
        capture_output=True, text=True, timeout=180,
    ).stdout
    return json.loads(
        re.search(r'<pre id="test-result">(.*?)</pre>', dom, re.S).group(1)
    )


# The helpers of the checks: the pixels NiiVue draws, read right after
# it draws them, and the setting of a control as the user would.
_CHECK_HELPERS = """
<script>
window.addEventListener('error', function (e) {
  window.__errors = (window.__errors || []).concat([String(e.message)]);
});
function pixels() {
  var nv = brainViewer.nv, gl = nv.gl;
  nv.drawScene();
  var w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
  var p = new Uint8Array(4 * w * h);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p);
  return p;
}
function changed(a, b) {
  for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return true;
  return false;
}
function set(id, value, kind) {
  var el = document.getElementById(id);
  if (el.type === 'checkbox') el.checked = value; else el.value = value;
  el.dispatchEvent(new Event(kind || 'input', {bubbles: true}));
}
function click(id) {
  document.getElementById(id).dispatchEvent(new Event('click'));
}
function report(out) {
  out.errors = window.__errors || [];
  document.getElementById('test-result').textContent = JSON.stringify(out);
}
</script>
<pre id="test-result">pending</pre>
"""

_BROWSER_CHECK = _CHECK_HELPERS + """
<script>
brainViewer.ready.then(function () {
  var out = {};
  var volume = brainViewer.nv.volumes[1];
  out.volumes = brainViewer.nv.volumes.length;
  var before = pixels();
  set('symmetric', false, 'change');
  out.oneSidedNegative = volume.colormapNegative;
  var oneSided = pixels();
  out.symmetric = changed(before, oneSided);
  set('threshold', '2.5', 'change');
  out.calMin = volume.cal_min;
  var thresholded = pixels();
  out.threshold = changed(oneSided, thresholded);
  set('cmap', 'viridis', 'change');
  out.colormap = volume.colormap;
  out.cmap = changed(thresholded, pixels());
  out.colorbarTicks = document.getElementById('colorbar').dataset.ticks;
  brainViewer.moveCursor([1, 2, 3]);
  out.value = document.getElementById('value').textContent;
  out.mm = document.getElementById('mm').textContent;
  out.voxel = document.getElementById('voxel').textContent;
  click('reset');
  out.resetCmap = document.getElementById('cmap').value;
  out.resetNegative = volume.colormapNegative;
  report(out);
}).catch(function (e) { report({failed: String(e)}); });
</script>
"""


@pytest.mark.skipif(_chrome() is None, reason="no headless Chrome")
def test_the_page_in_a_headless_browser(tmp_path):
    """Check the page runs offline, and its controls set the NiiVue
    overlay and redraw it."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 2, 3], data[0, 0, 3], data[2, 2, 3] = -2.0, 3.0, 1.0
    html = brain_viewer.viewer_html(
        _image(data), _image(np.linspace(0, 1, 64).reshape(4, 4, 4)),
        title="signed", threshold=0.5, cmap="hot", symmetric=True,
    )
    result = _run_in_chrome(tmp_path, html, _BROWSER_CHECK)
    assert result["errors"] == [] and "failed" not in result
    assert result["volumes"] == 2
    assert result["oneSidedNegative"] == ""
    assert result["resetNegative"] != ""
    assert result["calMin"] == 2.5
    assert result["colormap"].endswith("viridis")
    assert result["threshold"] and result["cmap"] and result["symmetric"]
    assert result["value"] == "-2"
    assert result["mm"] == "(-1, -1, -1)"
    assert result["voxel"] == "(1, 2, 3)"
    assert result["resetCmap"] == "hot"
    assert "3" in result["colorbarTicks"]


class TestLayers:
    """Check several images on one grid become the overlays of a page."""

    @pytest.fixture
    def state(self):
        first = np.zeros((4, 4, 4), dtype=np.float32)
        second = np.zeros((4, 4, 4), dtype=np.float32)
        first[1, 2, 3], second[0, 0, 0], second[1, 1, 1] = 2.0, -1.0, 4.0
        return _state(brain_viewer.viewer_html(
            [("lift", _image(first)), ("uniformity", _image(second))],
            None, title="t", threshold="95%", cmap="hot",
            symmetric=[False, True],
        ))

    def test_one_overlay_per_image(self, state):
        assert [o["name"] for o in state["overlays"]] == [
            "lift", "uniformity"
        ]
        assert _overlay_volume(state["overlays"][0], (4, 4, 4))[1, 2, 3] == 2
        second = _overlay_volume(state["overlays"][1], (4, 4, 4))
        assert second[0, 0, 0] == -1 and second[1, 1, 1] == 4

    def test_each_overlay_has_its_own_settings(self, state):
        layers = state["initial"]["layers"]
        assert [layer["symmetric"] for layer in layers] == [False, True]
        assert layers[0]["cmap"] == "hot"
        assert layers[1]["cmap"] != "hot"
        assert all(layer["visible"] for layer in layers)
        assert all(
            layer["threshold"] == 95.0
            and layer["thresholdMode"] == "percentile" for layer in layers
        )
        assert {layer["cmap"] for layer in layers} <= set(state["colormaps"])
        assert state["initial"]["active"] == 0

    def test_the_cursor_starts_on_the_peak_of_the_first(self, state):
        assert state["initial"]["cursor"] == [1, 2, 3]

    def test_the_images_must_share_a_grid(self):
        with pytest.raises(ValueError, match="grid"):
            brain_viewer.viewer_html(
                [("a", _image(np.ones((2, 2, 2), np.float32))),
                 ("b", _image(np.ones((3, 2, 2), np.float32)))],
                None, title="t", threshold=0, cmap=None, symmetric=False,
            )


_LAYERS_CHECK = _CHECK_HELPERS + """
<script>
brainViewer.ready.then(function () {
  var out = {};
  var names = function () {
    return brainViewer.nv.volumes.slice(1).map(function (v) {
      return v.neurolangName;
    });
  };
  out.colorbars = document.querySelectorAll('#colorbars canvas').length;
  out.layerShown = getComputedStyle(
    document.getElementById('layer-control')).display !== 'none';
  brainViewer.moveCursor([1, 2, 3]);
  out.value = document.getElementById('value').textContent;
  out.firstCmap = document.getElementById('cmap').value;
  set('layer', '1', 'change');
  out.secondCmap = document.getElementById('cmap').value;
  var before = pixels();
  set('visible', false, 'change');
  out.hidden = changed(before, pixels());
  out.hiddenOpacity = brainViewer.nv.volumes[2].opacity;
  set('visible', true, 'change');
  out.order = names();
  set('layer', '0', 'change');
  var drawn = pixels();
  click('layer-up');
  out.raised = names();
  out.raisedChanged = changed(drawn, pixels());
  click('layer-down');
  out.lowered = names();
  click('layer-up');
  click('reset');
  out.reset = names();
  report(out);
}).catch(function (e) { report({failed: String(e)}); });
</script>
"""


@pytest.mark.skipif(_chrome() is None, reason="no headless Chrome")
def test_the_overlays_in_a_headless_browser(tmp_path):
    """Check each overlay has a colour bar, a value at the cursor and
    its own controls, that hiding one redraws the slices, and that the
    order they are drawn in changes."""
    first = np.zeros((4, 4, 4), dtype=np.float32)
    second = np.zeros((4, 4, 4), dtype=np.float32)
    first[1:3, 1:3, :], second[1:3, 1:3, :] = 2.0, 5.0
    html = brain_viewer.viewer_html(
        [("lift", _image(first)), ("uniformity", _image(second))],
        None, title="layers", threshold=0.5, cmap="hot",
        symmetric=[False, False],
    )
    result = _run_in_chrome(tmp_path, html, _LAYERS_CHECK)
    assert result["errors"] == [] and "failed" not in result
    assert result["colorbars"] == 2 and result["layerShown"]
    assert result["value"] == "lift 2 · uniformity 5"
    assert result["firstCmap"] == "hot" and result["secondCmap"] != "hot"
    assert result["hidden"] and result["hiddenOpacity"] == 0
    assert result["order"] == ["lift", "uniformity"]
    assert result["raised"] == ["uniformity", "lift"]
    assert result["raisedChanged"]
    assert result["lowered"] == ["lift", "uniformity"]
    assert result["reset"] == ["lift", "uniformity"]
