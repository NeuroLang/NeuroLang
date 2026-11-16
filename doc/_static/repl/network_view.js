// The network tab: the network of an answer of edges (network_model.js) drawn
// as nodes and lines, the stronger an edge the thicker, those of a negative
// weight dashed, with a slider for the weakest edge shown.
import { SETTINGS } from "./settings.js";
import { defaultThreshold, hierarchy, labelled, layout, networkOf, nodeMeasures, treeLayout } from "./network_model.js";

const SVG = "http://www.w3.org/2000/svg";
const TREE_NODES = SETTINGS.network.tree_nodes;
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
  const figure = element("div", { className: "plot-figure network-figure" });
  const tip = element("div", { className: "plot-tip", hidden: true });
  tip.setAttribute("role", "status");
  const caption = element("p", { className: "note" });
  const box = element("div", { className: "network" },
    ...(title ? [element("h3", { className: "network-title", textContent: title })] : []),
    element("div", { className: "plot-controls" }, label, layoutLabel, sizeLabel, colourLabel, edgeLabel), figure, caption);

  const top = Math.max(...edges.map((edge) => Math.abs(edge.weight)));
  const layouts = {}; // each computed once, when first chosen
  let width, height, position;
  function arrange() {
    const radial = layoutSelect.value === "tree";
    layouts[layoutSelect.value] ??= radial ? treeLayout(network, tree) : layout(network, start);
    // room for the labels at the sides; a network of many nodes is larger,
    // and a radial tree square, its leaves on the circle of diameter *side*
    const margin = 180;
    const side = Math.max(560, 1.2 * nodes.length);
    [width, height] = radial ? [side + 2 * margin, side + 60] : [900, Math.max(600, 4 * nodes.length)];
    const place = radial
      ? ([x, y]) => [margin + x * side, 30 + y * side]
      : ([x, y]) => [margin + x * (width - 2 * margin), 24 + y * (height - 48)];
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
  figure.addEventListener("pointerleave", () => { tip.hidden = true; });
  // The share, in [0, 1], of a value of a measure: by the square root, so that a few large ones do not flatten the rest.
  const share = (values) => {
    const most = Math.sqrt(Math.max(...values.values(), 0)) || 1;
    return (id) => Math.sqrt(Math.max(values.get(id), 0)) / most;
  };

  function draw() {
    const threshold = weights[Number(slider.value)] ?? 0;
    const shown = edges.filter((edge) => Math.abs(edge.weight) >= threshold);
    const measures = Object.fromEntries(nodeMeasures(network, shown, tree).map((m) => [m.name, m.values]));
    const degree = measures.links;
    const sized = sizeSelect.value ? share(measures[sizeSelect.value]) : () => 0;
    const coloured = colourSelect.value ? share(measures[colourSelect.value]) : null;
    shownText.textContent = format.format(threshold);
    const named = labelled(degree);
    const lines = [...shown].reverse().map((edge) => { // the strongest drawn last, on top
      const [[x1, y1], [x2, y2]] = [position.get(edge.source), position.get(edge.target)];
      const strength = Math.abs(edge.weight) / top;
      const width = columns.weight ? 0.75 + 4.25 * strength : 1.5;
      const style = edgeSelect.value ? `stroke: color-mix(in oklab, var(${edge.weight < 0 ? "--answer" : "--accent"}) ${Math.round(15 + 85 * strength)}%, var(--panel)); stroke-opacity: 1` : "";
      const line = svg("g", { class: `network-edge${edge.weight < 0 ? " network-negative" : ""}` },
        svg("line", { x1, y1, x2, y2, class: "network-hit" }),
        svg("line", { x1, y1, x2, y2, "stroke-width": width.toFixed(2), class: "network-line", style }));
      line.addEventListener("pointermove", (event) => showTip(event, [[columns.source, edge.source], [columns.target, edge.target], ...(columns.weight ? [[columns.weight, format.format(edge.weight)]] : [])]));
      return line;
    });
    const many = nodes.length > 150; // smaller dots, not to cover each other
    const dots = nodes.map((node) => {
      const [x, y] = position.get(node.id);
      const right = x >= width / 2;
      const radius = (many ? 2.5 : 4) + (many ? 7 : 8) * sized(node.id);
      const fill = coloured ? `fill: color-mix(in oklab, var(--accent) ${Math.round(25 + 75 * coloured(node.id))}%, var(--panel))` : "";
      // in a tree the name of a node with children is above it, not on its branches
      const above = layoutSelect.value === "tree" && tree.children.get(node.id).length;
      const dot = svg("g", { class: `network-node${degree.get(node.id) ? "" : " network-alone"}${named.has(node.id) ? "" : " network-unnamed"}${many ? " network-small" : ""}` },
        svg("circle", { cx: x, cy: y, r: radius.toFixed(2), style: fill }),
        above
          ? svg("text", { x, y: y - radius - 5, "text-anchor": "middle" }, node.id)
          : svg("text", { x: x + (right ? 1 : -1) * (radius + 5), y: y + 4, "text-anchor": right ? "start" : "end" }, node.id));
      dot.addEventListener("pointermove", (event) => showTip(event, [[columns.source.replace(/_?1$/, ""), node.id],
        ...Object.entries(measures).map(([name, values]) => [name, String(values.get(node.id))])]));
      return dot;
    });
    const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "plot-svg network-svg", role: "img",
      "aria-label": `Network of ${nodes.length} ${columns.source.replace(/_?1$/, "")} nodes, ${shown.length} edges shown` }, ...lines, ...dots);
    figure.replaceChildren(chart, tip);
    const left = [dropped.edges && `${dropped.edges} weaker edges`, dropped.nodes && `${dropped.nodes} nodes`].filter(Boolean).join(" and ");
    const encoded = [sizeSelect.value && `the size of a node is its ${sizeSelect.value}`, colourSelect.value && `the darker, the more ${colourSelect.value}`].filter(Boolean).join("; ");
    caption.textContent = `${shown.length} of ${edges.length} edges between ${nodes.length} nodes` +
      `${named.size < nodes.length ? ` (the ${named.size} most linked named, the others under the pointer)` : ""}` +
      `${columns.weight ? `, from ${measure} ${format.format(threshold)}; the thicker${edgeSelect.value ? " and darker" : ""}, the stronger${signed ? ", dashed if negative" : ""}` : ""}.` +
      `${encoded ? ` ${encoded[0].toUpperCase()}${encoded.slice(1)}.` : ""}` +
      `${left ? ` Not drawn: ${left}.` : ""}${network.truncated ? " The answer has more rows than the table holds: the network is of those shown." : ""}`;
  }
  slider.addEventListener("input", draw);
  for (const select of [sizeSelect, colourSelect, edgeSelect]) select.addEventListener("change", draw);
  layoutSelect.addEventListener("change", () => { arrange(); draw(); });
  arrange();
  draw();
  return box;
}
