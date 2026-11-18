// The network tab: the network of an answer of edges (network_model.js) drawn
// as nodes and lines, the stronger an edge the thicker, those of a negative
// weight dashed, with a slider for the weakest edge shown. It zooms (Ctrl or
// Command and the wheel, a pinch, the buttons) and pans (a drag); a node
// found by name or clicked is focused: it and its neighbours stand out, and
// the panel beside lists them, to go from one to the next by keyboard.
import { SETTINGS } from "./settings.js";
import { centreOn, defaultThreshold, findNodes, hierarchy, labelled, layout, neighbours, networkOf, nodeMeasures, treeLayout, zoomAt } from "./network_model.js";

const SVG = "http://www.w3.org/2000/svg";
const { tree_nodes: TREE_NODES, labels: LABELS } = SETTINGS.network;
const [WIDTH, HEIGHT, MARGIN] = [900, 640, 150]; // the drawing, and the room for the labels at its sides
const wide = matchMedia("(min-width: 761px)"); // the panel of a node is then over the figure, on its right
const format = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });

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

// The networks of the *payloads* that are one, a figure each, or null.
export function networkViews(payloads) {
  const views = payloads.map((payload) => [payload, networkOf(payload)]).filter(([, network]) => network);
  if (!views.length) return null;
  return element("div", { className: "networks" }, ...views.map(([payload, network]) =>
    networkView(network, views.length > 1 ? payload.title : "")));
}

export function networkView(network, title = "") {
  const { nodes, edges, columns, dropped } = network;
  const noun = columns.source.replace(/_?1$/, "");
  const weights = [...new Set(edges.map((edge) => Math.abs(edge.weight)))].sort((a, b) => a - b);
  const start = defaultThreshold(network);
  const tree = hierarchy(network);
  const slider = element("input", { type: "range", min: 0, max: weights.length - 1, step: 1, value: weights.indexOf(start) });
  const shownText = element("output");
  const signed = edges.some((edge) => edge.weight < 0); // the threshold is then of the size of the weights
  const measure = signed ? `|${columns.weight}|` : columns.weight;
  const label = element("label", {}, element("span", { textContent: `Edges of ${measure} from` }), slider, shownText);
  label.hidden = !columns.weight || weights.length < 2;
  // A choice of the controls: *options* [value, text], the first chosen unless *value*.
  const choice = (text, options, value = options[0][0]) => {
    const select = element("select", {}, ...options.map(([v, name]) => element("option", { value: v, textContent: name })));
    select.value = value;
    return [select, element("label", {}, element("span", { textContent: text }), select)];
  };
  const measureNames = nodeMeasures(network, edges, tree).map((m) => m.name);
  const byMeasure = [["", "none"], ...measureNames.map((name) => [name, name])];
  // a small hierarchy as a radial tree; a large one is clearer by force, its parts apart
  const [layoutSelect, layoutLabel] = choice("Layout", [["force", "force"], ["tree", "tree"]], tree && !columns.weight && nodes.length <= TREE_NODES ? "tree" : "force");
  layoutLabel.hidden = !tree;
  const [sizeSelect, sizeLabel] = choice("Size", byMeasure, tree ? "nodes under it" : "links");
  const [colourSelect, colourLabel] = choice("Colour", byMeasure, "");
  const [edgeSelect, edgeLabel] = choice("Edge colour", [["", "none"], ...(columns.weight ? [["weight", columns.weight]] : [])], "");
  edgeLabel.hidden = !columns.weight;

  // The bar of the navigation: find a node, zoom, fit.
  const listId = `network-nodes-${Math.random().toString(36).slice(2, 8)}`;
  const search = element("input", { type: "search", placeholder: `Find a ${noun}…`, className: "network-search" });
  search.setAttribute("list", listId);
  search.setAttribute("aria-label", `Find a ${noun} of the network`);
  const names = element("datalist", { id: listId }, ...nodes.map((node) => element("option", { value: node.id })));
  const button = (text, label, action) => {
    const b = element("button", { type: "button", className: "network-button", textContent: text, title: label });
    b.setAttribute("aria-label", label);
    b.addEventListener("click", action);
    return b;
  };
  const status = element("span", { className: "network-status" });
  status.setAttribute("role", "status");
  const bar = element("div", { className: "network-bar" }, search, names,
    element("div", { className: "network-zoom" },
      button("−", "Zoom out", () => zoomBy(1 / 1.5)), button("+", "Zoom in", () => zoomBy(1.5)), button("Fit", "Show the whole network", fit)),
    element("span", { className: "network-hint", textContent: "Ctrl or ⌘ and scroll, or pinch, to zoom; drag to move; click a node" }), status);

  const figure = element("div", { className: "plot-figure network-figure" });
  const tip = element("div", { className: "plot-tip", hidden: true });
  tip.setAttribute("role", "status");
  const details = element("aside", { className: "network-details", hidden: true });
  details.setAttribute("aria-label", `The ${noun} focused`);
  const legend = element("div", { className: "network-legend" });
  const caption = element("p", { className: "note" });
  const box = element("div", { className: "network" },
    ...(title ? [element("h3", { className: "network-title", textContent: title })] : []),
    element("div", { className: "plot-controls" }, label, layoutLabel, sizeLabel, colourLabel, edgeLabel),
    bar, element("div", { className: "network-stage" }, figure, details), legend, caption);
  box.addEventListener("keydown", (event) => { if (event.key === "Escape") focus(null); });

  const top = Math.max(...edges.map((edge) => Math.abs(edge.weight)));
  const layouts = {}; // each computed once, when first chosen
  let position;
  function arrange() {
    const radial = layoutSelect.value === "tree";
    layouts[layoutSelect.value] ??= radial ? treeLayout(network, tree) : layout(network, start);
    // a radial tree is round: a square in the middle of the drawing
    const side = HEIGHT - 60;
    const place = radial
      ? ([x, y]) => [(WIDTH - side) / 2 + x * side, 30 + y * side]
      : ([x, y]) => [MARGIN + x * (WIDTH - 2 * MARGIN), 30 + y * (HEIGHT - 60)];
    position = new Map(nodes.map((node, i) => [node.id, place(layouts[layoutSelect.value][i])]));
  }

  function showTip(event, lines) {
    tip.replaceChildren(...lines.map(([name, value]) => element("div", {}, element("span", { textContent: name }), element("b", { textContent: value }))));
    tip.hidden = false;
    const bounds = figure.getBoundingClientRect();
    const left = Math.min(event.clientX - bounds.left + 14, bounds.width - tip.offsetWidth - 4);
    const above = event.clientY - bounds.top - tip.offsetHeight - 10;
    Object.assign(tip.style, { left: `${Math.max(4, left)}px`, top: `${above < 4 ? event.clientY - bounds.top + 18 : above}px` });
  }
  const hideTip = () => { tip.hidden = true; };
  figure.addEventListener("pointerleave", hideTip);
  // The share, in [0, 1], of a value of a measure: by the square root, so that a few large ones do not flatten the rest.
  const share = (values) => {
    const most = Math.sqrt(Math.max(...values.values(), 0)) || 1;
    return (id) => Math.sqrt(Math.max(values.get(id), 0)) / most;
  };
  const range = (values) => [Math.min(...values.values()), Math.max(...values.values())];

  // What is drawn, kept for the zoom and the focus.
  let view = { k: 1, x: 0, y: 0 };
  let focused = null;
  let shown = [], measures = {}, drawn = new Map(), lines = [];
  const chart = svg("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, class: "plot-svg network-svg", role: "img" });
  const world = svg("g", { class: "network-world" });
  chart.append(world);

  function draw() {
    const threshold = weights[Number(slider.value)] ?? 0;
    shown = edges.filter((edge) => Math.abs(edge.weight) >= threshold);
    measures = Object.fromEntries(nodeMeasures(network, shown, tree).map((m) => [m.name, m.values]));
    const degree = measures.links;
    const sized = sizeSelect.value ? share(measures[sizeSelect.value]) : () => 0;
    const coloured = colourSelect.value ? share(measures[colourSelect.value]) : null;
    shownText.textContent = format.format(threshold);
    lines = [...shown].reverse().map((edge) => { // the strongest drawn last, on top
      const [[x1, y1], [x2, y2]] = [position.get(edge.source), position.get(edge.target)];
      const strength = Math.abs(edge.weight) / top;
      const width = columns.weight ? 0.75 + 4.25 * strength : 1.5;
      const style = edgeSelect.value ? `stroke: color-mix(in oklab, var(${edge.weight < 0 ? "--answer" : "--accent"}) ${Math.round(15 + 85 * strength)}%, var(--panel)); stroke-opacity: 1` : "";
      const line = svg("g", { class: `network-edge${edge.weight < 0 ? " network-negative" : ""}` },
        svg("line", { x1, y1, x2, y2, class: "network-hit" }),
        svg("line", { x1, y1, x2, y2, "stroke-width": width.toFixed(2), class: "network-line", style }));
      line.addEventListener("pointermove", (event) => !dragging && showTip(event, [[columns.source, edge.source], [columns.target, edge.target], ...(columns.weight ? [[columns.weight, format.format(edge.weight)]] : [])]));
      line.edge = edge;
      return line;
    });
    const many = nodes.length > 150; // smaller dots, not to cover each other
    drawn = new Map(nodes.map((node) => {
      const [x, y] = position.get(node.id);
      const radius = (many ? 2.5 : 4) + (many ? 7 : 8) * sized(node.id);
      const fill = coloured ? `fill: color-mix(in oklab, var(--accent) ${Math.round(25 + 75 * coloured(node.id))}%, var(--panel))` : "";
      // in a tree the name of a node with children is above it, not on its branches
      const above = layoutSelect.value === "tree" && tree.children.get(node.id).length;
      const circle = svg("circle", { cx: x, cy: y, style: fill });
      const text = svg("text", { "text-anchor": above ? "middle" : x >= WIDTH / 2 ? "start" : "end" }, node.id);
      const dot = svg("g", { class: `network-node${degree.get(node.id) ? "" : " network-alone"}${many ? " network-small" : ""}` }, circle, text);
      dot.addEventListener("pointermove", (event) => !dragging && showTip(event, [[noun, node.id],
        ...Object.entries(measures).map(([name, values]) => [name, String(values.get(node.id))])]));
      dot.addEventListener("click", () => { if (!moved) focus(node.id); });
      return [node.id, { dot, circle, text, x, y, radius, above }];
    }));
    world.replaceChildren(...lines, ...[...drawn.values()].map((d) => d.dot));
    chart.setAttribute("aria-label", `Network of ${nodes.length} ${noun} nodes, ${shown.length} edges shown`);
    figure.replaceChildren(chart, tip);
    drawLegend(sized, coloured);
    const left = [dropped.edges && `${dropped.edges} weaker edges`, dropped.nodes && `${dropped.nodes} nodes`].filter(Boolean).join(" and ");
    caption.textContent = `${shown.length} of ${edges.length} edges between ${nodes.length} nodes` +
      `${columns.weight ? `, from ${measure} ${format.format(threshold)}` : ""}.` +
      `${left ? ` Not drawn: ${left}.` : ""}${network.truncated ? " The answer has more rows than the table holds: the network is of those shown." : ""}`;
    focus(focused && drawn.has(focused) ? focused : null);
  }

  // The legend of what the size, the shade and the lines show.
  function drawLegend() {
    const items = [];
    const swatch = (...children) => svg("svg", { width: 46, height: 18, viewBox: "0 0 46 18", class: "network-swatch", "aria-hidden": "true" }, ...children);
    if (sizeSelect.value) {
      const [low, high] = range(measures[sizeSelect.value]);
      items.push([swatch(svg("circle", { cx: 7, cy: 9, r: 3 }), svg("circle", { cx: 30, cy: 9, r: 8 })), `size: ${sizeSelect.value}, ${low} to ${high}`]);
    }
    if (colourSelect.value) {
      const [low, high] = range(measures[colourSelect.value]);
      items.push([swatch(...[25, 50, 75, 100].map((p, i) => svg("circle", { cx: 6 + i * 11.5, cy: 9, r: 5, style: `fill: color-mix(in oklab, var(--accent) ${p}%, var(--panel))` }))), `shade: ${colourSelect.value}, ${low} to ${high}`]);
    }
    if (columns.weight) {
      items.push([swatch(svg("line", { x1: 2, y1: 13, x2: 20, y2: 5, class: "network-line", "stroke-width": 1 }), svg("line", { x1: 24, y1: 13, x2: 44, y2: 5, class: "network-line", "stroke-width": 5 })), `edges: the thicker${edgeSelect.value ? " and more vivid" : ""}, the stronger ${columns.weight}`]);
      if (signed) items.push([swatch(svg("g", { class: "network-negative" }, svg("line", { x1: 2, y1: 9, x2: 44, y2: 9, class: "network-line", "stroke-width": 2.5 }))), `dashed: a negative ${columns.weight}`]);
    } else items.push([swatch(svg("line", { x1: 2, y1: 13, x2: 44, y2: 5, class: "network-line", "stroke-width": 1.5 })), `an edge: a row, from ${columns.source} to ${columns.target}`]);
    legend.replaceChildren(...items.map(([mark, text]) => element("span", {}, mark, element("span", { textContent: text }))));
  }

  // The zoom and the pan: the drawing at *view*, its names fewer the farther, as many as there is room for.
  let frame = 0;
  function applyView() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      world.setAttribute("transform", `translate(${view.x.toFixed(2)} ${view.y.toFixed(2)}) scale(${view.k.toFixed(4)})`);
      world.style.setProperty("--k", view.k.toFixed(4));
      const named = labelled(measures.links, Math.round(LABELS * view.k));
      if (focused) [focused, ...neighbours(shown, focused).map((n) => n.id)].forEach((id) => named.add(id));
      const grow = 1 / Math.sqrt(view.k); // the dots grow slower than the drawing
      for (const [id, d] of drawn) {
        const r = d.radius * grow;
        d.circle.setAttribute("r", r.toFixed(2));
        const gap = (r + 5 / view.k);
        if (d.above) {
          d.text.setAttribute("x", d.x);
          d.text.setAttribute("y", (d.y - gap).toFixed(2));
        } else {
          d.text.setAttribute("x", (d.x + (d.x >= WIDTH / 2 ? gap : -gap)).toFixed(2));
          d.text.setAttribute("y", (d.y + 4 / view.k).toFixed(2));
        }
        d.dot.classList.toggle("network-unnamed", !named.has(id));
      }
    });
  }
  // A point of the figure, from one of the page.
  const pointOf = (event) => {
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(chart.getScreenCTM().inverse());
    return [point.x, point.y];
  };
  function zoomBy(factor, at = [WIDTH / 2, HEIGHT / 2]) {
    view = zoomAt(view, factor, at);
    applyView();
  }
  function fit() {
    view = { k: 1, x: 0, y: 0 };
    applyView();
  }
  chart.addEventListener("wheel", (event) => {
    if (!(event.ctrlKey || event.metaKey)) return; // the wheel alone scrolls the page
    event.preventDefault();
    zoomBy(Math.exp(-event.deltaY * 0.0025), pointOf(event));
  }, { passive: false });
  // A drag pans; two fingers pinch.
  const pointers = new Map();
  let dragging = false, moved = false, pinch = null;
  chart.addEventListener("pointerdown", (event) => {
    pointers.set(event.pointerId, pointOf(event));
    chart.setPointerCapture(event.pointerId);
    dragging = true;
    moved = false;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
  });
  chart.addEventListener("pointermove", (event) => {
    if (!pointers.has(event.pointerId)) return;
    const [from, to] = [pointers.get(event.pointerId), pointOf(event)];
    pointers.set(event.pointerId, to);
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomBy(distance / pinch, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
      pinch = distance;
      moved = true;
      return;
    }
    const [dx, dy] = [to[0] - from[0], to[1] - from[1]];
    if (!moved && Math.hypot(dx, dy) < 3) return pointers.set(event.pointerId, from);
    moved = true;
    hideTip();
    chart.classList.add("network-panning");
    view = { ...view, x: view.x + dx, y: view.y + dy };
    applyView();
  });
  const release = (event) => {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size) return;
    dragging = false;
    chart.classList.remove("network-panning");
    // a click on the background, not a drag, ends the focus
    if (!moved && !event.target.closest?.(".network-node")) focus(null);
  };
  chart.addEventListener("pointerup", release);
  chart.addEventListener("pointercancel", release);

  // The focus on a node, *id*, or none: it and its neighbours stand out, the panel lists them.
  function focus(id, { centre = false } = {}) {
    focused = id;
    const near = id ? neighbours(shown, id) : [];
    const close = new Set([id, ...near.map((n) => n.id)]);
    box.classList.toggle("network-focusing", Boolean(id));
    for (const [other, d] of drawn) {
      d.dot.classList.toggle("network-focus", other === id);
      d.dot.classList.toggle("network-near", Boolean(id) && other !== id && close.has(other));
    }
    for (const line of lines) line.classList.toggle("network-near", Boolean(id) && (line.edge.source === id || line.edge.target === id));
    details.hidden = !id;
    if (id) {
      world.append(drawn.get(id).dot); // its name over those of the others
      const list = element("ul", {}, ...near.map((n) => {
        const go = element("button", { type: "button", textContent: n.id });
        go.addEventListener("click", () => focus(n.id, { centre: true }));
        return element("li", {}, go, ...(columns.weight ? [element("span", { textContent: format.format(n.weight) })] : []));
      }));
      const clear = button("×", "End the focus", () => focus(null));
      details.replaceChildren(
        element("div", { className: "network-details-head" }, element("h4", { textContent: id }), clear),
        element("dl", {}, ...Object.entries(measures).flatMap(([name, values]) => [element("dt", { textContent: name }), element("dd", { textContent: String(values.get(id)) })])),
        element("p", { className: "network-details-title", textContent: near.length ? `Linked to ${near.length}${columns.weight ? `, the strongest first (${columns.weight})` : ""}` : "No edge shown" }),
        list);
      // centred in the part of the figure the panel leaves free, on its left
      if (centre) view = centreOn(position.get(id), [WIDTH * (wide.matches ? 0.68 : 1), HEIGHT], Math.max(view.k, 2));
    }
    applyView();
  }
  function find() {
    const [first] = findNodes(nodes, search.value);
    status.textContent = search.value.trim() && !first ? `No ${noun} is named like “${search.value.trim()}”: try a part of a name.` : "";
    if (first) focus(first, { centre: true });
  }
  search.addEventListener("change", find);
  search.addEventListener("keydown", (event) => { if (event.key === "Enter") find(); });

  slider.addEventListener("input", draw);
  for (const select of [sizeSelect, colourSelect, edgeSelect]) select.addEventListener("change", draw);
  layoutSelect.addEventListener("change", () => { arrange(); fit(); draw(); });
  arrange();
  draw();
  return box;
}
