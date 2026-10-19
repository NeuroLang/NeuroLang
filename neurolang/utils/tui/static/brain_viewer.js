/*
 * Interactive viewer of brain images, inlined in the page that
 * neurolang/utils/tui/brain_viewer.py writes (see its docstring for the
 * data), after NiiVue, which draws the slices.
 *
 * The background is NiiVue's volume 0 and the overlays its next volumes,
 * drawn in turn, each over the previous ones. Each overlay has its own
 * threshold, colormap, symmetry, opacity and visibility, which the
 * controls set for the active one, and the order the overlays are drawn
 * in can be changed. NiiVue's display is set by controls of its own:
 * the view, the layout of the slices and the 3D render, the drag mode,
 * the crosshair, the gamma, NiiVue's colour bar, labels, orientation
 * cube and ruler, the radiological convention, the interpolation and
 * the clip plane; its hot keys V and C, on the views, change the view
 * and the clip plane. The colormaps of the page are given to NiiVue so
 * that it colours a value as the colour bars do:
 *
 * - one-sided, the colormap spans the threshold to the maximum, and the
 *   values below the threshold are transparent;
 * - symmetric, it spans -max|v| to max|v|, its upper half for the
 *   positive values and its lower half for the negative ones, and the
 *   values of |v| below the threshold are transparent.
 */
(function () {
  'use strict';

  var data = JSON.parse(document.getElementById('viewer-data').textContent);
  // NiiVue's colormap types: from the minimum to the maximum, or from 0,
  // transparent below the minimum.
  var MIN_TO_MAX = 0, ZERO_TO_MAX_TRANSPARENT = 2;

  // The display of NiiVue the page starts with, and Reset restores.
  var DISPLAY = {
    sliceType: 'multiplanar', layout: 'row', renderTile: false,
    dragMode: 'contrast', crosshairColor: '#4fa1ff', crosshairWidth: 1,
    crosshair3D: false, gamma: 1, colorbar: false, orientationText: true,
    orientCube: false, ruler: false, radiological: false, smooth: true,
    clipPlane: false
  };
  var SLICE_TYPES = ['axial', 'coronal', 'sagittal', 'multiplanar', 'render'];

  // Each layer holds the NiiVue volume of an overlay and the summary of
  // its values; the state holds the cursor, the active layer, the order
  // the layers are drawn in and the settings of each layer.
  var viewer = {layers: [], luts: {}, state: null, nv: null};

  function $(id) { return document.getElementById(id); }

  /* ---------------------------------------------------------------
   * Decoding of the embedded colormaps: base64 of zlib-deflated bytes.
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

  /* ---------------------------------------------------------------
   * The colormaps of NiiVue: for a colormap of the page, the whole of
   * it, its upper half and its lower half reversed, each transparent at
   * its first colour, which NiiVue gives the values below the minimum.
   */

  function niivueColormap(lut, index) {
    var map = {R: [], G: [], B: [], A: [], I: []};
    for (var i = 0; i < 256; i++) {
      var c = 3 * index(i);
      map.R.push(lut[c]); map.G.push(lut[c + 1]); map.B.push(lut[c + 2]);
      map.A.push(i === 0 ? 0 : 255); map.I.push(i);
    }
    return map;
  }

  function colormapKey(name, part) {
    return 'neurolang_' + name.toLowerCase() + (part ? '_' + part : '');
  }

  function addColormaps(nv) {
    Object.keys(viewer.luts).forEach(function (name) {
      var lut = viewer.luts[name];
      nv.addColormap(colormapKey(name), niivueColormap(lut, function (i) {
        return i;
      }));
      nv.addColormap(colormapKey(name, 'positive'), niivueColormap(
        lut, function (i) { return 128 + Math.round(i * 127 / 255); }
      ));
      nv.addColormap(colormapKey(name, 'negative'), niivueColormap(
        lut, function (i) { return 127 - Math.round(i * 127 / 255); }
      ));
    });
  }

  /* ---------------------------------------------------------------
   * Loading of the volumes.
   */

  function loadVolume(item, name) {
    return niivue.NVImage.loadFromBase64({base64: item.nifti, name: name});
  }

  // The layer of the NiiVue volume: the range of its nonzero values, and
  // their sorted lists for the percentiles, of |v| (symmetric colormap)
  // and of v (one-sided).
  function summarise(name, volume) {
    var img = volume.img, values = [];
    for (var i = 0; i < img.length; i++) {
      if (img[i] !== 0 && !isNaN(img[i])) values.push(img[i]);
    }
    var layer = {name: name, volume: volume};
    layer.sortedValues = Float32Array.from(values).sort();
    layer.sortedAbs = Float32Array.from(values, Math.abs).sort();
    var n = values.length;
    layer.min = n ? layer.sortedValues[0] : 0;
    layer.max = n ? layer.sortedValues[n - 1] : 0;
    layer.maxAbs = n ? layer.sortedAbs[n - 1] : 0;
    return layer;
  }

  async function load() {
    for (var name in data.colormaps) {
      viewer.luts[name] = new Uint8Array(await inflate(data.colormaps[name]));
    }
    var nv = new niivue.Niivue({
      backColor: [0, 0, 0, 1], crosshairColor: [0.31, 0.63, 1, 0.85],
      crosshairWidth: 1, isColorbar: false, isRadiologicalConvention: false,
      multiplanarShowRender: niivue.SHOW_RENDER.NEVER,
      multiplanarLayout: niivue.MULTIPLANAR_TYPE.ROW,
      show3Dcrosshair: false, dragAndDropEnabled: false
    });
    await nv.attachTo('gl');
    addColormaps(nv);
    var background = await loadVolume(data.background, 'background.nii.gz');
    background.colormap = 'gray';
    nv.addVolume(background);
    for (var l = 0; l < data.overlays.length; l++) {
      var volume = await loadVolume(data.overlays[l], 'overlay' + l + '.nii.gz');
      volume.neurolangName = data.overlays[l].name;
      nv.addVolume(volume);
      viewer.layers.push(summarise(data.overlays[l].name, volume));
    }
    background.colorbarVisible = false;
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.onLocationChange = function () { updateReadout(); };
    viewer.nv = nv;
  }

  /* ---------------------------------------------------------------
   * NiiVue's display: the display of the state, set by its controls.
   */

  // The colour of the crosshair, as NiiVue takes it, from #rrggbb.
  function hexColour(hex) {
    return [1, 3, 5].map(function (i) {
      return parseInt(hex.substr(i, 2), 16) / 255;
    }).concat([0.85]);
  }

  function isRow(d) {
    return d.sliceType === 'multiplanar' && d.layout === 'row';
  }

  // Set NiiVue's views: the slice type, the layout and the 3D tile, the
  // views being taller unless they are the 3 slices in a row.
  function applyViews(d) {
    var nv = viewer.nv;
    $('views').classList.toggle('tall', !isRow(d));
    nv.opts.multiplanarShowRender =
      niivue.SHOW_RENDER[d.renderTile ? 'ALWAYS' : 'NEVER'];
    nv.setMultiplanarLayout(niivue.MULTIPLANAR_TYPE[d.layout.toUpperCase()]);
    nv.setSliceType(SLICE_TYPES.indexOf(d.sliceType));
    if (nv.resizeListener) nv.resizeListener();
  }

  function applyDisplay() {
    var nv = viewer.nv, d = viewer.state.display;
    nv.opts.dragMode = nv.dragModes[d.dragMode];
    nv.opts.show3Dcrosshair = d.crosshair3D;
    nv.opts.isColorbar = d.colorbar;
    nv.opts.isOrientCube = d.orientCube;
    nv.opts.isRuler = d.ruler;
    nv.setCrosshairColor(hexColour(d.crosshairColor));
    nv.setCrosshairWidth(d.crosshairWidth);
    nv.setGamma(d.gamma);
    nv.setIsOrientationTextVisible(d.orientationText);
    nv.setRadiologicalConvention(d.radiological);
    nv.setInterpolation(!d.smooth);
    nv.setClipPlane(d.clipPlane ? [0, 270, 0] : [2, 0, 0]);
    applyViews(d);
  }

  // The controls of the display, by id, and the setting each holds.
  var DISPLAY_CONTROLS = {
    'slice-type': 'sliceType', 'layout': 'layout',
    'render-tile': 'renderTile', 'drag-mode': 'dragMode',
    'crosshair-color': 'crosshairColor',
    'crosshair-width': 'crosshairWidth', 'crosshair-3d': 'crosshair3D',
    'gamma': 'gamma', 'nv-colorbar': 'colorbar',
    'orientation-text': 'orientationText', 'orient-cube': 'orientCube',
    'ruler': 'ruler', 'radiological': 'radiological', 'smooth': 'smooth',
    'clip-plane': 'clipPlane'
  };

  function controlValue(el) {
    if (el.type === 'checkbox') return el.checked;
    if (el.type === 'number' || el.type === 'range') {
      return parseFloat(el.value);
    }
    return el.value;
  }

  function syncDisplayControls() {
    var d = viewer.state.display;
    Object.keys(DISPLAY_CONTROLS).forEach(function (id) {
      var el = $(id), value = d[DISPLAY_CONTROLS[id]];
      if (el.type === 'checkbox') el.checked = value; else el.value = value;
    });
  }

  // Keep the state in step with what NiiVue's hot keys changed.
  function followHotKeys() {
    var nv = viewer.nv, d = viewer.state.display;
    d.sliceType = SLICE_TYPES[nv.opts.sliceType];
    d.clipPlane = nv.scene.clipPlaneDepthAziElevs[0][0] < 2;
    $('views').classList.toggle('tall', !isRow(d));
    if (nv.resizeListener) nv.resizeListener();
    syncDisplayControls();
  }

  function setUpDisplay() {
    Object.keys(DISPLAY_CONTROLS).forEach(function (id) {
      var el = $(id);
      var kind = el.type === 'range' || el.type === 'color' ? 'input' : 'change';
      el.addEventListener(kind, function () {
        var value = controlValue(el);
        if (typeof value === 'number' && isNaN(value)) return;
        viewer.state.display[DISPLAY_CONTROLS[id]] = value;
        applyDisplay();
      });
    });
    $('gl').addEventListener('keyup', followHotKeys);
  }

  // The settings of the layer l, of the active one by default.
  function settings(l) {
    var s = viewer.state;
    return s.layers[l === undefined ? s.active : l];
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

  function percentileSource(l) {
    var layer = viewer.layers[l];
    return settings(l).symmetric ? layer.sortedAbs : layer.sortedValues;
  }

  function thresholdValue(l) {
    var s = settings(l);
    if (s.thresholdMode === 'percentile') {
      return percentile(percentileSource(l), s.threshold);
    }
    return s.threshold;
  }

  // The colour scale of the layer l: values from lo to hi span its
  // colormap, and only the voxels shown(v) are drawn.
  function scale(l) {
    var t = thresholdValue(l), layer = viewer.layers[l];
    if (settings(l).symmetric) {
      return {lo: -layer.maxAbs, hi: layer.maxAbs, threshold: t,
              shown: function (v) { return Math.abs(v) >= t; }};
    }
    var lo = t > 0 ? t : Math.min(0, layer.min);
    return {lo: lo, hi: Math.max(layer.max, lo), threshold: t,
            shown: function (v) { return v >= t; }};
  }

  // The index (0 to 255) of the colour of v in the scale.
  function colourIndex(v, s) {
    var span = s.hi - s.lo;
    var t = span > 0 ? (v - s.lo) / span : 1;
    return Math.round(255 * Math.min(1, Math.max(0, t)));
  }

  // Set the NiiVue volume of the layer l as its settings and scale s
  // tell.
  function applyLayer(l, s) {
    var settingsOf = settings(l), volume = viewer.layers[l].volume;
    if (settingsOf.symmetric) {
      volume.colormap = colormapKey(settingsOf.cmap, 'positive');
      volume.colormapNegative = colormapKey(settingsOf.cmap, 'negative');
      volume.colormapType = ZERO_TO_MAX_TRANSPARENT;
      volume.cal_min = s.threshold;
      volume.cal_max = Math.max(s.hi, s.threshold);
    } else {
      volume.colormap = colormapKey(settingsOf.cmap);
      volume.colormapNegative = '';
      volume.colormapType = MIN_TO_MAX;
      volume.cal_min = s.lo;
      volume.cal_max = s.hi;
    }
    volume.cal_minNeg = NaN;
    volume.cal_maxNeg = NaN;
    volume.opacity = settingsOf.visible ? settingsOf.opacity : 0;
  }

  /* ---------------------------------------------------------------
   * Colour bars, read-out and controls.
   */

  function format(v) {
    if (v === 0 || !isFinite(v)) return String(v);
    if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) return v.toExponential(2);
    return String(Number(v.toPrecision(3)));
  }

  function round(v) { return String(Math.round(v * 100) / 100); }

  function ticksOf(s, l) {
    var ticks = settings(l).symmetric
      ? [s.lo, -s.threshold, 0, s.threshold, s.hi]
      : [s.lo, s.threshold, s.hi];
    return ticks.filter(function (t, i) {
      return t >= s.lo && t <= s.hi && ticks.indexOf(t) === i;
    });
  }

  // The colour bar of the layer l, on its canvas.
  function drawColourBar(s, l) {
    var canvas = viewer.layers[l].colorbar, context = canvas.getContext('2d');
    var lut = viewer.luts[settings(l).cmap];
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
    var ticks = ticksOf(s, l).map(format);
    context.fillStyle = '#ddd';
    context.font = '12px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'top';
    ticksOf(s, l).forEach(function (t, i) {
      var x = xOf(t), half = context.measureText(ticks[i]).width / 2 + 1;
      context.fillRect(Math.round(x), 18, 1, 5);
      // Keep the labels at the ends of the bar inside the canvas.
      var at = Math.min(Math.max(x, half), canvas.width - half);
      context.fillText(ticks[i], at, 26);
    });
    canvas.dataset.ticks = ticks.join(' ');
    canvas.style.opacity = settings(l).visible ? 1 : 0.4;
  }

  // The millimetres of the voxel ijk of the images, and its inverse.
  function toMm(ijk) {
    var a = data.affine;
    return [0, 1, 2].map(function (r) {
      return a[r][0] * ijk[0] + a[r][1] * ijk[1] + a[r][2] * ijk[2] + a[r][3];
    });
  }

  function toVoxel(mm) {
    var a = data.affine;
    var m = [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]],
             [a[2][0], a[2][1], a[2][2]]];
    var d = [mm[0] - a[0][3], mm[1] - a[1][3], mm[2] - a[2][3]];
    var det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // Cramer's rule, column by column.
    return [0, 1, 2].map(function (c) {
      var mc = m.map(function (row, r) {
        return row.map(function (v, k) { return k === c ? d[r] : v; });
      });
      var dc = mc[0][0] * (mc[1][1] * mc[2][2] - mc[1][2] * mc[2][1]) -
        mc[0][1] * (mc[1][0] * mc[2][2] - mc[1][2] * mc[2][0]) +
        mc[0][2] * (mc[1][0] * mc[2][1] - mc[1][1] * mc[2][0]);
      return Math.round(dc / det);
    });
  }

  function cursorMm() {
    var mm = viewer.nv.frac2mm(viewer.nv.scene.crosshairPos, 0, true);
    return [mm[0], mm[1], mm[2]];
  }

  function updateReadout() {
    if (!viewer.state) return;
    var mm = cursorMm();
    var several = viewer.layers.length > 1;
    $('mm').textContent = '(' + mm.map(round).join(', ') + ')';
    $('voxel').textContent = '(' + toVoxel(mm).join(', ') + ')';
    $('value').textContent = viewer.layers.map(function (layer) {
      var vox = layer.volume.mm2vox(mm);
      var value = layer.volume.getValue(vox[0], vox[1], vox[2]);
      value = String(Number(Number(value).toPrecision(6)));
      return several ? layer.name + ' ' + value : value;
    }).join(' · ');
  }

  function redraw() {
    var scales = viewer.layers.map(function (layer, l) { return scale(l); });
    scales.forEach(function (s, l) { applyLayer(l, s); });
    viewer.nv.updateGLVolume();
    scales.forEach(drawColourBar);
    updateReadout();
    var mode = settings().thresholdMode;
    $('threshold-value').textContent =
      mode === 'percentile' ? '= ' + format(scales[viewer.state.active].threshold) : '';
  }

  // The largest threshold of the slider, in the unit of the mode.
  function sliderTop() {
    var layer = viewer.layers[viewer.state.active];
    return settings().thresholdMode === 'percentile'
      ? 100 : Math.max(layer.maxAbs, layer.max, 1e-12);
  }

  function syncControls() {
    var s = settings();
    $('layer').value = String(viewer.state.active);
    $('visible').checked = s.visible;
    $('threshold').value = Number(s.threshold.toPrecision(6));
    $('threshold').max = s.thresholdMode === 'percentile' ? 100 : '';
    $('threshold-slider').value = Math.round(1000 * s.threshold / sliderTop());
    $('threshold-mode').value = s.thresholdMode;
    $('cmap').value = s.cmap;
    $('symmetric').checked = s.symmetric;
    $('opacity').value = s.opacity;
    var position = viewer.state.order.indexOf(viewer.state.active);
    $('layer-up').disabled = position === viewer.state.order.length - 1;
    $('layer-down').disabled = position === 0;
  }

  function setThreshold(value) {
    if (isNaN(value)) return;
    var top = settings().thresholdMode === 'percentile' ? 100 : Infinity;
    settings().threshold = Math.min(Math.max(value, 0), top);
    syncControls();
    redraw();
  }

  // Switch between a value and a percentile, keeping the threshold.
  function setThresholdMode(mode) {
    var s = settings(), active = viewer.state.active;
    if (mode === s.thresholdMode) return;
    var value = thresholdValue(active);
    s.thresholdMode = mode;
    s.threshold = mode === 'percentile'
      ? rank(percentileSource(active), value) : Math.max(0, value);
    syncControls();
    redraw();
  }

  function setOption(name, value) {
    settings()[name] = value;
    syncControls();
    redraw();
  }

  /* ---------------------------------------------------------------
   * The order the overlays are drawn in, shown from the top in the list
   * of the layers and in the colour bars.
   */

  function drawInOrder() {
    viewer.state.order.forEach(function (l, position) {
      viewer.nv.setVolume(viewer.layers[l].volume, position + 1);
    });
  }

  function listLayers() {
    var list = $('layer'), bars = $('colorbars');
    var several = viewer.layers.length > 1;
    list.innerHTML = '';
    viewer.state.order.slice().reverse().forEach(function (l) {
      var layer = viewer.layers[l];
      list.add(new Option(layer.name || 'overlay', String(l)));
      if (several) bars.appendChild(layer.label);
      bars.appendChild(layer.colorbar);
    });
  }

  // Draw the active layer over the next one (step 1) or under the
  // previous one (step -1).
  function moveLayer(step) {
    var order = viewer.state.order;
    var position = order.indexOf(viewer.state.active);
    var to = position + step;
    if (to < 0 || to >= order.length) return;
    order[position] = order[to];
    order[to] = viewer.state.active;
    drawInOrder();
    listLayers();
    syncControls();
    redraw();
  }

  function reset() {
    viewer.state = JSON.parse(JSON.stringify(data.initial));
    viewer.state.order = viewer.layers.map(function (layer, l) { return l; });
    viewer.state.display = Object.assign({}, DISPLAY);
    drawInOrder();
    listLayers();
    syncControls();
    syncDisplayControls();
    applyDisplay();
    redraw();
  }

  // A colour bar per layer, labelled with its name when there are
  // several, and the list of the layers, shown with several.
  function setUpLayers() {
    var several = viewer.layers.length > 1;
    $('layer-control').style.display = several ? '' : 'none';
    viewer.layers.forEach(function (layer, l) {
      var canvas = l === 0 ? $('colorbar') : document.createElement('canvas');
      if (l > 0) {
        canvas.id = 'colorbar-' + l;
        canvas.width = $('colorbar').width;
        canvas.height = $('colorbar').height;
      }
      layer.colorbar = canvas;
      layer.label = document.createElement('div');
      layer.label.className = 'colorbar-name';
      layer.label.textContent = layer.name;
    });
  }

  function setUpControls() {
    var cmaps = $('cmap');
    Object.keys(viewer.luts).forEach(function (name) {
      cmaps.add(new Option(name, name));
    });
    $('layer').addEventListener('change', function (e) {
      viewer.state.active = parseInt(e.target.value, 10);
      syncControls();
      redraw();
    });
    $('layer-up').addEventListener('click', function () { moveLayer(1); });
    $('layer-down').addEventListener('click', function () { moveLayer(-1); });
    $('visible').addEventListener('change', function (e) {
      setOption('visible', e.target.checked);
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
    $('reset').addEventListener('click', function () {
      reset();
      moveCursor(data.initial.cursor);
    });
  }

  // Move the cursor to the voxel ijk of the images.
  function moveCursor(ijk) {
    var nv = viewer.nv;
    nv.scene.crosshairPos = nv.mm2frac(toMm(ijk), 0, true);
    nv.drawScene();
    updateReadout();
  }

  async function start() {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot read the page\'s data ' +
                      '(no DecompressionStream): use a recent browser.');
    }
    await load();
    setUpLayers();
    setUpControls();
    setUpDisplay();
    reset();
    moveCursor(data.initial.cursor);
  }

  window.brainViewer = {
    viewer: viewer,
    get nv() { return viewer.nv; },
    moveCursor: moveCursor,
    ready: start().catch(function (error) {
      $('message').textContent = 'The viewer failed: ' + error.message;
      throw error;
    })
  };
})();
