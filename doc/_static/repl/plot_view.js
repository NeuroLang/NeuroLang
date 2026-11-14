// The plot of a result's table: bars, a histogram, points or a line of its
// columns (plot_model.js), drawn in SVG with the page's colours, with the
// values of a mark under the pointer. The table stays the view of every value.
import { SETTINGS } from "./settings.js";
import { barData, histogram, kindsOf, niceTicks, plotChoice, pointData } from "./plot_model.js";
import { cellText } from "./table_model.js";

const SVG = "http://www.w3.org/2000/svg";
const KIND_LABELS = { bar: "Bars", histogram: "Histogram", scatter: "Scatter", line: "Line" };
const NUMERIC = new Set(["int", "float"]);
const COUNT = -1; // the value of bars without a value column: the rows of each label

function element(tag, properties = {}, ...children) {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  node.append(...children);
  return node;
}

const text = (x, y, content, attributes = {}) => svg("text", { x, y, ...attributes }, content);
const format = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });
const number = (value) => format.format(value);
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumSignificantDigits: 3 });
// A tick of an axis: in full under 10,000, compact (20K, 3M) from there.
const tickText = (value) => (Math.abs(value) >= 1e4 ? compact.format(value) : number(value));
const linear = ([d0, d1], [r0, r1]) => (value) => r0 + ((value - d0) / (d1 - d0 || 1)) * (r1 - r0);
// A label cut to about *width* pixels of 12px text.
const fit = (label, width) => (label.length * 6.6 > width ? `${label.slice(0, Math.max(1, Math.floor(width / 6.6) - 1))}…` : label);

// A bar from *x0* to *x1*, *height* high from *y*, its end away from the baseline rounded.
function barPath(x0, x1, y, height, radius = 4) {
  const [left, right] = [Math.min(x0, x1), Math.max(x0, x1)];
  const r = Math.min(radius, (right - left) / 2, height / 2);
  if (x1 >= x0) return `M${left},${y}H${right - r}A${r},${r} 0 0 1 ${right},${y + r}V${y + height - r}A${r},${r} 0 0 1 ${right - r},${y + height}H${left}Z`;
  return `M${right},${y}H${left + r}A${r},${r} 0 0 0 ${left},${y + r}V${y + height - r}A${r},${r} 0 0 0 ${left + r},${y + height}H${right}Z`;
}

// A column from the baseline *y0* up to *y1*, *width* wide from *x*, its top rounded.
function columnPath(x, width, y0, y1, radius = 4) {
  const r = Math.min(radius, width / 2, Math.abs(y0 - y1));
  return `M${x},${y0}V${y1 + r}A${r},${r} 0 0 1 ${x + r},${y1}H${x + width - r}A${r},${r} 0 0 1 ${x + width},${y1 + r}V${y0}Z`;
}

// The plot of the table of *model* in *box*: its controls and its figure.
// update() draws it again, after the filter or the sort changed.
export function plotView(box, model) {
  const choice = plotChoice(model);
  const kinds = kindsOf(model);
  const numeric = model.columns.map((_, column) => column).filter((column) => NUMERIC.has(model.kinds[column]));
  const option = (value, label) => element("option", { value, textContent: label });
  const kind = element("select", {}, ...kinds.map((name) => option(name, KIND_LABELS[name])));
  const x = element("select");
  const y = element("select");
  kind.setAttribute("aria-label", "Kind of plot");
  const xLabel = element("label", {}, element("span", { textContent: "x" }), x);
  const yLabel = element("label", {}, element("span", { textContent: "y" }), y);
  const controls = element("div", { className: "plot-controls" }, kind, xLabel, yLabel);
  const figure = element("div", { className: "plot-figure" });
  const tip = element("div", { className: "plot-tip", hidden: true });
  tip.setAttribute("role", "status");
  const caption = element("p", { className: "note plot-caption" });
  box.replaceChildren(controls, figure, caption);

  // The columns each axis can take for the kind of plot chosen, keeping the choice made when it still can.
  function offerColumns(keep = true) {
    const [oldX, oldY] = [Number(x.value), Number(y.value)];
    const bar = kind.value === "bar";
    const xs = bar ? model.columns.map((_, column) => column) : numeric;
    const ys = bar ? [COUNT, ...numeric] : numeric;
    x.replaceChildren(...xs.map((column) => option(column, model.columns[column])));
    y.replaceChildren(...ys.map((column) => option(column, column === COUNT ? "number of rows" : model.columns[column])));
    xLabel.firstChild.textContent = bar ? "label" : "x";
    yLabel.firstChild.textContent = bar ? "value" : "y";
    yLabel.hidden = kind.value === "histogram";
    x.value = String(keep && xs.includes(oldX) ? oldX : xs.find((column) => column !== Number(y.value)) ?? xs[0]);
    const fallback = ys.find((column) => column !== Number(x.value) && column !== COUNT) ?? ys[0];
    y.value = String(keep && ys.includes(oldY) && oldY !== Number(x.value) ? oldY : fallback);
  }

  kind.value = choice.kind;
  x.replaceChildren(option(choice.x, ""));
  y.replaceChildren(option(choice.y, ""));
  x.value = String(choice.x);
  y.value = String(choice.y);
  offerColumns();
  kind.addEventListener("change", () => { offerColumns(); update(); });
  x.addEventListener("change", update);
  y.addEventListener("change", update);

  let width = 0;
  new ResizeObserver(() => {
    if (figure.clientWidth && Math.abs(figure.clientWidth - width) > 4) update();
  }).observe(figure);

  // The values of a mark, by the pointer, kept inside the figure.
  function showTip(event, lines) {
    tip.replaceChildren(...lines.map(([name, value]) => element("div", {}, element("span", { textContent: name }), element("b", { textContent: value }))));
    tip.hidden = false;
    const bounds = figure.getBoundingClientRect();
    const left = Math.min(event.clientX - bounds.left + 14, bounds.width - tip.offsetWidth - 4);
    const top = event.clientY - bounds.top - tip.offsetHeight - 10;
    Object.assign(tip.style, { left: `${Math.max(4, left)}px`, top: `${top < 4 ? event.clientY - bounds.top + 18 : top}px` });
  }
  const hideTip = () => { tip.hidden = true; };
  figure.addEventListener("pointerleave", hideTip);
  // The cells of a *row*, a few, for the tip of a point.
  const rowLines = (row, shown) => [...shown, ...model.columns.map((name, column) => [name, cellText(row[column])])
    .filter(([name]) => !shown.some(([taken]) => taken === name))].slice(0, 5);

  function update() {
    width = figure.clientWidth || 640;
    hideTip();
    const [xc, yc] = [Number(x.value), Number(y.value)];
    const draw = { bar: drawBars, histogram: drawHistogram, scatter: drawPoints, line: drawPoints }[kind.value];
    const { chart, label, note } = draw(xc, yc);
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", label);
    figure.replaceChildren(chart, tip);
    caption.textContent = note;
  }

  // Axes: the gridlines and ticks of *ticks* along *scale*, horizontal ones for y.
  function yGrid(ticks, scale, left, right) {
    return svg("g", { class: "plot-axis" }, ...ticks.flatMap((tick) => [
      svg("line", { x1: left, x2: right, y1: scale(tick), y2: scale(tick), class: "plot-grid" }),
      text(left - 8, scale(tick) + 4, tickText(tick), { "text-anchor": "end" }),
    ]));
  }
  function xTicks(ticks, scale, top, bottom) {
    return svg("g", { class: "plot-axis" }, ...ticks.flatMap((tick) => [
      svg("line", { x1: scale(tick), x2: scale(tick), y1: top, y2: bottom, class: "plot-grid" }),
      text(scale(tick), bottom + 16, tickText(tick), { "text-anchor": "middle" }),
    ]));
  }

  function drawBars(xc, yc) {
    const { bars, more } = barData(model, xc, yc);
    const valueName = yc === COUNT ? "rows" : model.columns[yc];
    const labelWidth = Math.min(width * 0.38, Math.max(...bars.map((bar) => bar.label.length), 4) * 6.6 + 8);
    const [band, thickness, top] = [24, 16, 8];
    const left = labelWidth + 12, right = width - 56;
    const low = Math.min(0, ...bars.map((bar) => bar.value)), high = Math.max(0, ...bars.map((bar) => bar.value));
    const scale = linear([low, high], [left, right]);
    const height = top + bars.length * band + 24;
    const ticks = niceTicks(low, high, Math.max(2, Math.round((right - left) / 90))).filter((tick) => tick >= low && tick <= high);
    const chart = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "plot-svg" },
      xTicks(ticks, scale, top, top + bars.length * band),
      ...bars.map((bar, index) => {
        const y0 = top + index * band + (band - thickness) / 2;
        const mark = svg("g", { class: `plot-bar${bar.top ? " plot-top" : ""}` },
          svg("rect", { x: 0, y: top + index * band, width, height: band, class: "plot-hit" }),
          text(labelWidth, y0 + thickness / 2 + 4, fit(bar.label, labelWidth), { "text-anchor": "end", class: "plot-label" }),
          svg("path", { d: barPath(scale(0), scale(bar.value), y0, thickness), class: "plot-mark" }),
          text(Math.max(scale(bar.value), scale(0)) + 6, y0 + thickness / 2 + 4, number(bar.value), { class: "plot-value" }));
        mark.addEventListener("pointermove", (event) => showTip(event, [[model.columns[xc], bar.label], [valueName, number(bar.value)]]));
        return mark;
      }));
    const shown = bars.length ? `${bars.length} bars` : "no bars: no row has a value";
    return {
      chart,
      label: `Bars of ${valueName} by ${model.columns[xc]}, ${shown}`,
      note: `${yc === COUNT ? `The number of rows of each ${model.columns[xc]}, most first` : `${valueName} of each row, in the order of the table`}${more ? `: the first ${bars.length}, ${more} more not shown` : ""}. The largest is in red.`,
    };
  }

  function drawHistogram(xc) {
    const { bins, low, high } = histogram(model, xc);
    const height = 300, top = 12, bottom = height - 40, left = 52, right = width - 16;
    const counts = niceTicks(0, Math.max(1, ...bins.map((bin) => bin.count)), 4);
    const yScale = linear([0, counts.at(-1)], [bottom, top]);
    const xs = niceTicks(low, high, Math.max(2, Math.round((right - left) / 90)));
    const xScale = linear([xs[0], xs.at(-1)], [left, right]);
    const chart = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "plot-svg" },
      yGrid(counts, yScale, left, right),
      svg("g", { class: "plot-axis" }, ...xs.map((value) => text(xScale(value), bottom + 16, tickText(value), { "text-anchor": "middle" })),
        text((left + right) / 2, bottom + 34, model.columns[xc], { "text-anchor": "middle", class: "plot-title" })),
      ...bins.map((bin) => {
        const x0 = xScale(bin.x0) + 1, w = Math.max(1, xScale(bin.x1) - xScale(bin.x0) - 2);
        const mark = svg("g", { class: "plot-bar" },
          svg("rect", { x: x0 - 1, y: top, width: w + 2, height: bottom - top, class: "plot-hit" }),
          ...(bin.count ? [svg("path", { d: columnPath(x0, w, bottom, yScale(bin.count)), class: "plot-mark" })] : []));
        mark.addEventListener("pointermove", (event) => showTip(event, [[model.columns[xc], `${number(bin.x0)} – ${number(bin.x1)}`], ["rows", String(bin.count)]]));
        return mark;
      }),
      svg("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "plot-base" }));
    const total = bins.reduce((sum, bin) => sum + bin.count, 0);
    return { chart, label: `Histogram of ${model.columns[xc]}, ${total} values in ${bins.length} bins`, note: `${total} values of ${model.columns[xc]}, of the rows the filter keeps, in ${bins.length} bins.` };
  }

  let chartStyle = "";
  function drawPoints(xc, yc) {
    const line = kind.value === "line";
    const points = pointData(model, xc, yc, { line });
    const height = 320, top = 12, bottom = height - 40, left = 56, right = width - 16;
    const extent = (values) => [Math.min(...values), Math.max(...values)];
    const ys = niceTicks(...(points.length ? extent(points.map((p) => p.y)) : [0, 1]), 5);
    const xs = niceTicks(...(points.length ? extent(points.map((p) => p.x)) : [0, 1]), Math.max(2, Math.round((right - left) / 90)));
    const xScale = linear([xs[0], xs.at(-1)], [left, right]);
    const yScale = linear([ys[0], ys.at(-1)], [bottom, top]);
    const dots = !line || points.length <= SETTINGS.plot.line_dots;
    // Many points are small and the fainter the more there are, so that where they pile up shows.
    const dense = points.length > SETTINGS.plot.dense_points;
    chartStyle = dense ? `--dot-opacity: ${Math.max(0.07, Math.min(0.5, 200 / points.length)).toFixed(3)}` : "";
    const chart = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "plot-svg", style: chartStyle },
      yGrid(ys, yScale, left, right),
      svg("g", { class: "plot-axis" }, ...xs.map((value) => text(xScale(value), bottom + 16, tickText(value), { "text-anchor": "middle" })),
        text((left + right) / 2, bottom + 34, model.columns[xc], { "text-anchor": "middle", class: "plot-title" }),
        text(left, top - 2, model.columns[yc], { class: "plot-title", dy: "-0.2em" })),
      ...(line && points.length ? [svg("path", { d: points.map((p, i) => `${i ? "L" : "M"}${xScale(p.x)},${yScale(p.y)}`).join(""), class: "plot-line" })] : []),
      ...(dots ? points.map((p) => svg("circle", { cx: xScale(p.x), cy: yScale(p.y), r: dense ? 2 : 4, class: dense ? "plot-dot plot-dense" : "plot-dot" })) : []),
      svg("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "plot-base" }));
    const focus = svg("circle", { r: 6, class: "plot-focus", visibility: "hidden" });
    chart.append(focus);
    // The point nearest to the pointer (for a line, the nearest along x), within reach.
    chart.addEventListener("pointermove", (event) => {
      const bounds = chart.getBoundingClientRect();
      const [px, py] = [event.clientX - bounds.left, event.clientY - bounds.top];
      let best = null, distance = Infinity;
      for (const p of points) {
        const d = line ? Math.abs(xScale(p.x) - px) : Math.hypot(xScale(p.x) - px, yScale(p.y) - py);
        if (d < distance) [best, distance] = [p, d];
      }
      if (!best || distance > (line ? 40 : 24)) {
        focus.setAttribute("visibility", "hidden");
        return hideTip();
      }
      Object.entries({ cx: xScale(best.x), cy: yScale(best.y), visibility: "visible" }).forEach(([name, value]) => focus.setAttribute(name, value));
      showTip(event, rowLines(best.row, [[model.columns[xc], number(best.x)], [model.columns[yc], number(best.y)]]));
    });
    const what = line ? "Line" : "Points";
    return {
      chart,
      label: `${what} of ${model.columns[yc]} against ${model.columns[xc]}, ${points.length} rows`,
      note: `${points.length} rows the filter keeps, with both values${line ? ", in the order of x" : ""}.`,
    };
  }

  return { update };
}
