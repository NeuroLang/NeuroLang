/*
 * Interactive viewer of a brain image, inlined in the page that
 * neurolang/utils/tui/brain_viewer.py writes (see its docstring for the
 * data). Three orthogonal slices of the image over its background, with
 * a crosshair at the cursor, redrawn when a control or the cursor changes.
 *
 * The volumes are in RAS+ orientation, flattened with the first index
 * fastest. The slices follow nilearn: neurological (left on the left),
 * anterior on the right of the sagittal slice, superior / anterior up.
 */
(function () {
  'use strict';

  var data = JSON.parse(document.getElementById('viewer-data').textContent);
  var shape = data.shape;
  var strides = [1, shape[0], shape[0] * shape[1]];
  var SIZE = 320;  // pixels of the longest side of the slices

  // Each view shows the axes h (left to right) and v (bottom to top) at
  // the cursor's position on the fixed axis.
  var VIEWS = [
    {id: 'sagittal', h: 1, v: 2, fixed: 0, labels: ['P', 'A', 'S', 'I']},
    {id: 'coronal', h: 0, v: 2, fixed: 1, labels: ['L', 'R', 'S', 'I']},
    {id: 'axial', h: 0, v: 1, fixed: 2, labels: ['L', 'R', 'A', 'P']}
  ];
  var AXIS_NAMES = ['x', 'y', 'z'];

  var viewer = {
    volume: null, background: null, luts: {}, state: null,
    sortedAbs: null, sortedValues: null, min: 0, max: 0, maxAbs: 0
  };

  function $(id) { return document.getElementById(id); }

  /* ---------------------------------------------------------------
   * Decoding of the embedded arrays: base64 of zlib-deflated bytes.
   */

  function base64Bytes(text) {
    var binary = atob(text);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function inflate(text) {
    var stream = new Blob([base64Bytes(text)]).stream()
      .pipeThrough(new DecompressionStream('deflate'));
    return new Response(stream).arrayBuffer();
  }

  // An IEEE 754 half float, from its 16 bits.
  function halfToFloat(h) {
    var sign = h & 0x8000 ? -1 : 1;
    var exponent = (h >> 10) & 0x1f;
    var fraction = h & 0x3ff;
    if (exponent === 0) return sign * Math.pow(2, -14) * fraction / 1024;
    if (exponent === 31) return fraction ? NaN : sign * Infinity;
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  }

  function floats(buffer, dtype) {
    if (dtype === 'float32') return new Float32Array(buffer);
    var halves = new Uint16Array(buffer);
    var out = new Float32Array(halves.length);
    for (var i = 0; i < halves.length; i++) out[i] = halfToFloat(halves[i]);
    return out;
  }

  async function decodeOverlay(overlay) {
    var values = floats(await inflate(overlay.values), overlay.dtype);
    if (overlay.encoding === 'dense') return values;
    var index = new Uint32Array(await inflate(overlay.index));
    var volume = new Float32Array(shape[0] * shape[1] * shape[2]);
    for (var i = 0; i < index.length; i++) volume[index[i]] = values[i];
    return volume;
  }

  async function decode() {
    viewer.volume = await decodeOverlay(data.overlay);
    if (data.background) {
      viewer.background = new Uint8Array(await inflate(data.background.data));
    }
    for (var name in data.colormaps) {
      viewer.luts[name] = new Uint8Array(await inflate(data.colormaps[name]));
    }
    summarise(viewer.volume);
  }

  // The range of the nonzero values, and their sorted lists for the
  // percentiles: of |v| (symmetric colormap) and of v (one-sided).
  function summarise(volume) {
    var values = [];
    for (var i = 0; i < volume.length; i++) {
      if (volume[i] !== 0 && !isNaN(volume[i])) values.push(volume[i]);
    }
    viewer.sortedValues = Float32Array.from(values).sort();
    viewer.sortedAbs = Float32Array.from(values, Math.abs).sort();
    var n = values.length;
    viewer.min = n ? viewer.sortedValues[0] : 0;
    viewer.max = n ? viewer.sortedValues[n - 1] : 0;
    viewer.maxAbs = n ? viewer.sortedAbs[n - 1] : 0;
  }

  /* ---------------------------------------------------------------
   * Threshold and colour scale.
   */

  // The percentile p (0 to 100) of sorted, interpolated as numpy does.
  function percentile(sorted, p) {
    if (!sorted.length) return 0;
    var position = (sorted.length - 1) * p / 100;
    var below = Math.floor(position);
    var above = Math.min(below + 1, sorted.length - 1);
    return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
  }

  // The fraction (0 to 100) of sorted below value: percentile's inverse.
  function rank(sorted, value) {
    if (sorted.length < 2) return 0;
    var lo = 0, hi = sorted.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (sorted[mid] < value) lo = mid + 1; else hi = mid;
    }
    return Math.min(100, 100 * lo / (sorted.length - 1));
  }

  function percentileSource() {
    return viewer.state.symmetric ? viewer.sortedAbs : viewer.sortedValues;
  }

  function thresholdValue() {
    var s = viewer.state;
    if (s.thresholdMode === 'percentile') {
      return percentile(percentileSource(), s.threshold);
    }
    return s.threshold;
  }

  // The colour scale: values from lo to hi span the colormap, and only
  // the voxels shown(v) are drawn.
  function scale() {
    var t = thresholdValue();
    if (viewer.state.symmetric) {
      return {lo: -viewer.maxAbs, hi: viewer.maxAbs, threshold: t,
              shown: function (v) { return Math.abs(v) >= t; }};
    }
    var lo = t > 0 ? t : Math.min(0, viewer.min);
    return {lo: lo, hi: Math.max(viewer.max, lo), threshold: t,
            shown: function (v) { return v >= t; }};
  }

  // The index (0 to 255) of the colour of v in the scale.
  function colourIndex(v, s) {
    var span = s.hi - s.lo;
    var t = span > 0 ? (v - s.lo) / span : 1;
    return Math.round(255 * Math.min(1, Math.max(0, t)));
  }

  /* ---------------------------------------------------------------
   * Drawing of the slices.
   */

  function voxelSizes() {
    return [0, 1, 2].map(function (c) {
      var a = data.affine;
      return Math.hypot(a[0][c], a[1][c], a[2][c]);
    });
  }

  // Size the canvases to the millimetres they show, the longest side of
  // the largest SIZE pixels; each keeps an image of one pixel per voxel.
  function setUpViews() {
    var sizes = voxelSizes();
    var extents = [0, 1, 2].map(function (a) { return shape[a] * sizes[a]; });
    var pixelsPerMm = SIZE / Math.max.apply(null, extents);
    VIEWS.forEach(function (view) {
      view.canvas = $(view.id);
      view.canvas.width = Math.max(1, Math.round(extents[view.h] * pixelsPerMm));
      view.canvas.height = Math.max(1, Math.round(extents[view.v] * pixelsPerMm));
      view.buffer = document.createElement('canvas');
      view.buffer.width = shape[view.h];
      view.buffer.height = shape[view.v];
      view.image = view.buffer.getContext('2d')
        .createImageData(shape[view.h], shape[view.v]);
    });
  }

  // The flat index in the background of the voxel (i, j, k).
  function backgroundIndex(ijk) {
    var b = data.background, f = b.factor;
    return Math.floor(ijk[0] / f) + b.shape[0] * (
      Math.floor(ijk[1] / f) + b.shape[1] * Math.floor(ijk[2] / f));
  }

  function fillSlice(view, s) {
    var nh = shape[view.h], nv = shape[view.v];
    var lut = viewer.luts[viewer.state.cmap];
    var alpha = viewer.state.opacity;
    var pixels = view.image.data;
    var ijk = viewer.state.cursor.slice();
    for (var y = 0; y < nv; y++) {
      ijk[view.v] = nv - 1 - y;
      for (var x = 0; x < nh; x++) {
        ijk[view.h] = x;
        var o = 4 * (y * nh + x);
        var grey = viewer.background ? viewer.background[backgroundIndex(ijk)] : 0;
        var v = viewer.volume[ijk[0] + strides[1] * ijk[1] + strides[2] * ijk[2]];
        var r = grey, g = grey, b = grey;
        if (v !== 0 && s.shown(v)) {
          var c = 3 * colourIndex(v, s);
          r = grey + (lut[c] - grey) * alpha;
          g = grey + (lut[c + 1] - grey) * alpha;
          b = grey + (lut[c + 2] - grey) * alpha;
        }
        pixels[o] = r; pixels[o + 1] = g; pixels[o + 2] = b; pixels[o + 3] = 255;
      }
    }
  }

  function drawLabels(context, view, width, height) {
    var cursor = viewer.state.cursor;
    context.fillStyle = '#fff';
    context.font = '13px system-ui, sans-serif';
    context.textBaseline = 'middle';
    context.textAlign = 'left';
    context.fillText(view.labels[0], 4, height / 2);
    context.textAlign = 'right';
    context.fillText(view.labels[1], width - 4, height / 2);
    context.textAlign = 'center';
    context.fillText(view.labels[2], width / 2, 10);
    context.fillText(view.labels[3], width / 2, height - 10);
    context.textAlign = 'left';
    var mm = toMm(cursor)[view.fixed];
    context.fillText(AXIS_NAMES[view.fixed] + ' = ' + round(mm), 4, height - 10);
  }

  function drawView(view, s) {
    fillSlice(view, s);
    view.buffer.getContext('2d').putImageData(view.image, 0, 0);
    var canvas = view.canvas, context = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    context.imageSmoothingEnabled = false;
    context.drawImage(view.buffer, 0, 0, w, h);
    // The crosshair, through the centre of the cursor's voxel.
    var cursor = viewer.state.cursor;
    var cx = (cursor[view.h] + 0.5) * w / shape[view.h];
    var cy = (shape[view.v] - cursor[view.v] - 0.5) * h / shape[view.v];
    context.strokeStyle = 'rgba(80, 160, 255, 0.85)';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(Math.round(cx) + 0.5, 0);
    context.lineTo(Math.round(cx) + 0.5, h);
    context.moveTo(0, Math.round(cy) + 0.5);
    context.lineTo(w, Math.round(cy) + 0.5);
    context.stroke();
    drawLabels(context, view, w, h);
  }

  /* ---------------------------------------------------------------
   * Colour bar, read-out and controls.
   */

  function format(v) {
    if (v === 0 || !isFinite(v)) return String(v);
    if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) return v.toExponential(2);
    return String(Number(v.toPrecision(3)));
  }

  function round(v) { return String(Math.round(v * 100) / 100); }

  function ticksOf(s) {
    var ticks = viewer.state.symmetric
      ? [s.lo, -s.threshold, 0, s.threshold, s.hi]
      : [s.lo, s.threshold, s.hi];
    return ticks.filter(function (t, i) {
      return t >= s.lo && t <= s.hi && ticks.indexOf(t) === i;
    });
  }

  function drawColourBar(s) {
    var canvas = $('colorbar'), context = canvas.getContext('2d');
    var lut = viewer.luts[viewer.state.cmap];
    var left = 12, width = canvas.width - 2 * left, span = s.hi - s.lo;
    var xOf = function (v) { return left + (span > 0 ? (v - s.lo) / span : 1) * width; };
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (var x = 0; x < width; x++) {
      var v = s.lo + span * x / Math.max(1, width - 1);
      var c = 3 * colourIndex(v, s);
      context.fillStyle = 'rgb(' + lut[c] + ',' + lut[c + 1] + ',' + lut[c + 2] + ')';
      context.fillRect(left + x, 2, 1, 16);
      if (!s.shown(v)) {  // hidden values are greyed out
        context.fillStyle = 'rgba(17, 17, 17, 0.75)';
        context.fillRect(left + x, 2, 1, 16);
      }
    }
    var ticks = ticksOf(s).map(format);
    context.fillStyle = '#ddd';
    context.font = '12px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'top';
    ticksOf(s).forEach(function (t, i) {
      var x = xOf(t), half = context.measureText(ticks[i]).width / 2 + 1;
      context.fillRect(Math.round(x), 18, 1, 5);
      // Keep the labels at the ends of the bar inside the canvas.
      var at = Math.min(Math.max(x, half), canvas.width - half);
      context.fillText(ticks[i], at, 26);
    });
    canvas.dataset.ticks = ticks.join(' ');
  }

  function toMm(ijk) {
    var a = data.affine;
    return [0, 1, 2].map(function (r) {
      return a[r][0] * ijk[0] + a[r][1] * ijk[1] + a[r][2] * ijk[2] + a[r][3];
    });
  }

  function updateReadout() {
    var cursor = viewer.state.cursor, m = data.toImageVoxel;
    var voxel = [0, 1, 2].map(function (r) {
      return Math.round(m[r][0] * cursor[0] + m[r][1] * cursor[1] +
                        m[r][2] * cursor[2] + m[r][3]);
    });
    var value = viewer.volume[cursor[0] + strides[1] * cursor[1] + strides[2] * cursor[2]];
    $('mm').textContent = '(' + toMm(cursor).map(round).join(', ') + ')';
    $('voxel').textContent = '(' + voxel.join(', ') + ')';
    $('value').textContent = String(Number(value.toPrecision(6)));
  }

  function redraw() {
    var s = scale();
    VIEWS.forEach(function (view) { drawView(view, s); });
    drawColourBar(s);
    updateReadout();
    var mode = viewer.state.thresholdMode;
    $('threshold-value').textContent =
      mode === 'percentile' ? '= ' + format(s.threshold) : '';
  }

  // The largest threshold of the slider, in the unit of the mode.
  function sliderTop() {
    return viewer.state.thresholdMode === 'percentile'
      ? 100 : Math.max(viewer.maxAbs, viewer.max, 1e-12);
  }

  function syncControls() {
    var s = viewer.state;
    $('threshold').value = Number(s.threshold.toPrecision(6));
    $('threshold').max = s.thresholdMode === 'percentile' ? 100 : '';
    $('threshold-slider').value = Math.round(1000 * s.threshold / sliderTop());
    $('threshold-mode').value = s.thresholdMode;
    $('cmap').value = s.cmap;
    $('symmetric').checked = s.symmetric;
    $('opacity').value = s.opacity;
  }

  function setThreshold(value) {
    if (isNaN(value)) return;
    var top = viewer.state.thresholdMode === 'percentile' ? 100 : Infinity;
    viewer.state.threshold = Math.min(Math.max(value, 0), top);
    syncControls();
    redraw();
  }

  // Switch between a value and a percentile, keeping the threshold.
  function setThresholdMode(mode) {
    var s = viewer.state;
    if (mode === s.thresholdMode) return;
    var value = thresholdValue();
    s.thresholdMode = mode;
    s.threshold = mode === 'percentile'
      ? rank(percentileSource(), value) : Math.max(0, value);
    syncControls();
    redraw();
  }

  function setOption(name, value) {
    viewer.state[name] = value;
    syncControls();
    redraw();
  }

  function reset() {
    viewer.state = JSON.parse(JSON.stringify(data.initial));
    syncControls();
    redraw();
  }

  function setUpControls() {
    var cmaps = $('cmap');
    Object.keys(viewer.luts).forEach(function (name) {
      cmaps.add(new Option(name, name));
    });
    var onThreshold = function (e) { setThreshold(parseFloat(e.target.value)); };
    $('threshold').addEventListener('change', onThreshold);
    $('threshold').addEventListener('input', onThreshold);
    $('threshold-slider').addEventListener('input', function (e) {
      setThreshold(sliderTop() * e.target.value / 1000);
    });
    $('threshold-mode').addEventListener('change', function (e) {
      setThresholdMode(e.target.value);
    });
    $('cmap').addEventListener('change', function (e) { setOption('cmap', e.target.value); });
    $('symmetric').addEventListener('change', function (e) {
      setOption('symmetric', e.target.checked);
    });
    $('opacity').addEventListener('input', function (e) {
      setOption('opacity', parseFloat(e.target.value));
    });
    $('reset').addEventListener('click', reset);
  }

  /* ---------------------------------------------------------------
   * Cursor: click or drag on a slice, arrow keys (Page Up / Down
   * through the slices) on the focused one.
   */

  function moveCursor(ijk) {
    viewer.state.cursor = ijk.map(function (v, a) {
      return Math.min(Math.max(Math.round(v), 0), shape[a] - 1);
    });
    redraw();
  }

  function pointAt(view, event) {
    var rect = view.canvas.getBoundingClientRect();
    var cursor = viewer.state.cursor.slice();
    cursor[view.h] = Math.floor((event.clientX - rect.left) / rect.width * shape[view.h]);
    cursor[view.v] = shape[view.v] - 1 -
      Math.floor((event.clientY - rect.top) / rect.height * shape[view.v]);
    moveCursor(cursor);
  }

  var KEY_STEPS = {
    ArrowLeft: ['h', -1], ArrowRight: ['h', 1], ArrowUp: ['v', 1],
    ArrowDown: ['v', -1], PageUp: ['fixed', 1], PageDown: ['fixed', -1]
  };

  function setUpCursor() {
    VIEWS.forEach(function (view) {
      var canvas = view.canvas;
      canvas.addEventListener('pointerdown', function (e) {
        canvas.setPointerCapture(e.pointerId);
        canvas.focus();
        pointAt(view, e);
      });
      canvas.addEventListener('pointermove', function (e) {
        if (e.buttons) pointAt(view, e);
      });
      canvas.addEventListener('keydown', function (e) {
        var step = KEY_STEPS[e.key];
        if (!step) return;
        e.preventDefault();
        var cursor = viewer.state.cursor.slice();
        cursor[view[step[0]]] += step[1];
        moveCursor(cursor);
      });
    });
  }

  async function start() {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot read the page\'s data ' +
                      '(no DecompressionStream): use a recent browser.');
    }
    await decode();
    setUpViews();
    setUpControls();
    setUpCursor();
    reset();
  }

  window.brainViewer = {
    viewer: viewer,
    moveCursor: moveCursor,
    ready: start().catch(function (error) {
      $('message').textContent = 'The viewer failed: ' + error.message;
      throw error;
    })
  };
})();
