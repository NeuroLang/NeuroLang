"""Tests for the interactive brain viewer page that ``.view`` writes."""

import base64
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


def _overlay_volume(overlay, shape):
    """Return the volume an encoded overlay stands for."""
    values = _decode(overlay["values"], overlay["dtype"]).astype(np.float32)
    if overlay["encoding"] == "dense":
        return values.reshape(shape, order="F")
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    flat[_decode(overlay["index"], np.uint32)] = values
    return flat.reshape(shape, order="F")


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
    """Check the voxel values are embedded compactly, and losslessly."""

    def test_sparse_round_trip(self):
        data = np.zeros((20, 30, 10), dtype=np.float32)
        data[1, 2, 3], data[19, 29, 9], data[5, 0, 7] = 0.25, -1.5, 3.1
        overlay = brain_viewer.encode_overlay(data)
        assert overlay["encoding"] == "sparse" and overlay["count"] == 3
        assert np.array_equal(_overlay_volume(overlay, data.shape), data)

    def test_dense_round_trip(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(6, 7, 8)).astype(np.float32)
        overlay = brain_viewer.encode_overlay(data)
        assert overlay["encoding"] == "dense"
        assert overlay["dtype"] == "float32"
        assert np.array_equal(_overlay_volume(overlay, data.shape), data)

    def test_exact_values_take_half_floats(self):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[2:4, 3, 5] = 1
        overlay = brain_viewer.encode_overlay(data)
        assert overlay["dtype"] == "float16"
        assert np.array_equal(_overlay_volume(overlay, data.shape), data)

    def test_dense_when_smaller_than_sparse(self):
        data = np.ones((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 0
        overlay = brain_viewer.encode_overlay(data)
        assert overlay["encoding"] == "dense"
        assert np.array_equal(_overlay_volume(overlay, data.shape), data)

    def test_nan_is_no_value(self):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[0, 0, 0], data[1, 1, 1] = np.nan, 2.0
        volume = _overlay_volume(brain_viewer.encode_overlay(data), (4, 4, 4))
        assert volume[0, 0, 0] == 0 and volume[1, 1, 1] == 2


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
    """Check the page holds its controls, its data and no external URL."""

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
            'id="opacity"', 'id="reset"', 'id="colorbar"',
            'id="sagittal"', 'id="coronal"', 'id="axial"', 'id="readout"',
        ):
            assert control in html

    def test_initial_state(self, html):
        state = _state(html)
        assert state["initial"] == {
            "threshold": 95.0, "thresholdMode": "percentile",
            "cmap": "viridis", "symmetric": True, "opacity": 1.0,
            "cursor": [0, 0, 0],
        }
        assert state["shape"] == [4, 4, 4]
        assert np.array_equal(state["affine"], AFFINE)
        assert "viridis" in state["colormaps"]
        assert state["background"]["factor"] == 1

    def test_value_threshold_and_default_colormap(self):
        html = brain_viewer.viewer_html(
            _image(np.ones((2, 2, 2), np.float32)), None, title="t",
            threshold=0.5, cmap=None, symmetric=False,
        )
        state = _state(html)
        assert state["initial"]["threshold"] == 0.5
        assert state["initial"]["thresholdMode"] == "value"
        assert state["initial"]["cmap"] == brain_viewer.DEFAULT_CMAP
        assert state["background"] is None

    def test_title_is_escaped(self, html):
        assert "<title>signed &lt;map&gt;</title>" in html

    def test_no_external_urls(self, html):
        assert not re.search(r"https?://|//cdn|\bsrc=|\bhref=", html)

    def test_script_is_inlined(self, html):
        assert "function" in html and "<script src" not in html
        assert html.count("</script>") == 2

    def test_canonical_orientation(self):
        flip = np.diag([-2.0, 2.0, 2.0, 1.0])
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[0, 1, 2] = 5.0
        state = _state(brain_viewer.viewer_html(
            _image(data, flip), None, title="t", threshold=0,
            cmap="hot", symmetric=False,
        ))
        assert state["affine"][0][0] > 0
        volume = _overlay_volume(state["overlay"], (4, 4, 4))
        assert volume[3, 1, 2] == 5.0
        assert state["initial"]["cursor"] == [3, 1, 2]

    def test_cursor_on_the_peak(self):
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[1, 1, 1], data[3, 2, 4] = 1.0, -4.0
        state = _state(brain_viewer.viewer_html(
            _image(data), None, title="t", threshold=0, cmap="hot",
            symmetric=True,
        ))
        assert state["initial"]["cursor"] == [3, 2, 4]


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


# Driven through the page's controls, then the results are written to the
# DOM, which Chrome dumps once the page has settled.
_BROWSER_CHECK = """
<script>
window.addEventListener('error', function (e) {
  window.__errors = (window.__errors || []).concat([String(e.message)]);
});
brainViewer.ready.then(function () {
  var out = {errors: window.__errors || []};
  var axial = document.getElementById('axial');
  function pixels() {
    var c = axial.getContext('2d');
    return Array.from(c.getImageData(0, 0, axial.width, axial.height).data);
  }
  function changed(a, b) {
    return a.some(function (v, i) { return v !== b[i]; });
  }
  function set(id, value, kind) {
    var el = document.getElementById(id);
    if (el.type === 'checkbox') el.checked = value; else el.value = value;
    el.dispatchEvent(new Event(kind || 'input', {bubbles: true}));
  }
  out.readout = document.getElementById('readout').textContent;
  var before = pixels();
  set('symmetric', false, 'change');
  var oneSided = pixels();
  out.symmetric = changed(before, oneSided);
  set('threshold', '2.5', 'change');
  var thresholded = pixels();
  out.threshold = changed(oneSided, thresholded);
  set('cmap', 'viridis', 'change');
  out.cmap = changed(thresholded, pixels());
  out.colorbarTicks = document.getElementById('colorbar').dataset.ticks;
  brainViewer.moveCursor([1, 2, 3]);
  out.value = document.getElementById('value').textContent;
  out.mm = document.getElementById('mm').textContent;
  set('reset', null, 'click');
  out.resetCmap = document.getElementById('cmap').value;
  out.errors = window.__errors || [];
  document.getElementById('test-result').textContent = JSON.stringify(out);
}).catch(function (e) {
  document.getElementById('test-result').textContent =
    JSON.stringify({errors: [String(e)]});
});
</script>
<pre id="test-result">pending</pre>
"""


@pytest.mark.skipif(_chrome() is None, reason="no headless Chrome")
def test_the_page_in_a_headless_browser(tmp_path):
    """Check the page runs, and its controls redraw the slices."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 2, 3], data[0, 0, 3], data[2, 2, 3] = -2.0, 3.0, 1.0
    html = brain_viewer.viewer_html(
        _image(data), _image(np.linspace(0, 1, 64).reshape(4, 4, 4)),
        title="signed", threshold=0.5, cmap="hot", symmetric=True,
    )
    page = tmp_path / "page.html"
    page.write_text(html.replace("</body>", _BROWSER_CHECK + "</body>"))
    dom = subprocess.run(
        [
            _chrome(), "--headless", "--disable-gpu", "--no-first-run",
            f"--user-data-dir={tmp_path / 'profile'}",
            "--virtual-time-budget=10000", "--dump-dom", page.as_uri(),
        ],
        capture_output=True, text=True, timeout=120,
    ).stdout
    result = json.loads(
        re.search(r'<pre id="test-result">(.*?)</pre>', dom, re.S).group(1)
    )
    assert result["errors"] == []
    assert result["threshold"] and result["cmap"] and result["symmetric"]
    assert result["value"] == "-2"
    assert result["mm"] == "(-1, -1, -1)"
    assert result["resetCmap"] == "hot"
    assert "3" in result["colorbarTicks"]
