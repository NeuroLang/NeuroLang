// The network tab: the network of an answer of edges (network_model.js) drawn
// as nodes and lines, the stronger an edge the thicker, those of a negative
// weight dashed, with a slider for the weakest edge shown.
import { defaultThreshold, labelled, layout, networkOf } from "./network_model.js";

const SVG = "http://www.w3.org/2000/svg";
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
  const slider = element("input", { type: "range", min: 0, max: weights.length - 1, step: 1, value: weights.indexOf(start) });
  const shownText = element("output");
  const signed = edges.some((edge) => edge.weight < 0); // the threshold is then of the size of the weights
  const measure = signed ? `|${columns.weight}|` : columns.weight;
  const label = element("label", {}, element("span", { textContent: `Edges of ${measure} from` }), slider, shownText);
  label.hidden = !columns.weight || weights.length < 2;
  const figure = element("div", { className: "plot-figure network-figure" });
  const tip = element("div", { className: "plot-tip", hidden: true });
  tip.setAttribute("role", "status");
  const caption = element("p", { className: "note" });
  const box = element("div", { className: "network" },
    ...(title ? [element("h3", { className: "network-title", textContent: title })] : []),
    element("div", { className: "plot-controls" }, label), figure, caption);

  const at = layout(network, start);
  // room for the labels at the sides; a network of many nodes is taller
  const [width, height, margin] = [900, Math.max(600, 4 * nodes.length), 180];
  const place = ([x, y]) => [margin + x * (width - 2 * margin), 24 + y * (height - 48)];
  const position = new Map(nodes.map((node, i) => [node.id, place(at[i])]));
  const top = Math.max(...edges.map((edge) => Math.abs(edge.weight)));

  function showTip(event, lines) {
    tip.replaceChildren(...lines.map(([name, value]) => element("div", {}, element("span", { textContent: name }), element("b", { textContent: value }))));
    tip.hidden = false;
    const bounds = figure.getBoundingClientRect();
    const left = Math.min(event.clientX - bounds.left + 14, bounds.width - tip.offsetWidth - 4);
    const above = event.clientY - bounds.top - tip.offsetHeight - 10;
    Object.assign(tip.style, { left: `${Math.max(4, left)}px`, top: `${above < 4 ? event.clientY - bounds.top + 18 : above}px` });
  }
  figure.addEventListener("pointerleave", () => { tip.hidden = true; });

  function draw() {
    const threshold = weights[Number(slider.value)] ?? 0;
    const shown = edges.filter((edge) => Math.abs(edge.weight) >= threshold);
    const degree = new Map(nodes.map((node) => [node.id, 0]));
    for (const edge of shown) for (const end of [edge.source, edge.target]) degree.set(end, degree.get(end) + 1);
    shownText.textContent = format.format(threshold);
    const named = labelled(degree);
    const lines = [...shown].reverse().map((edge) => { // the strongest drawn last, on top
      const [[x1, y1], [x2, y2]] = [position.get(edge.source), position.get(edge.target)];
      const width = 0.75 + 4.25 * (Math.abs(edge.weight) / top);
      const line = svg("g", { class: `network-edge${edge.weight < 0 ? " network-negative" : ""}` },
        svg("line", { x1, y1, x2, y2, class: "network-hit" }),
        svg("line", { x1, y1, x2, y2, "stroke-width": width.toFixed(2), class: "network-line" }));
      line.addEventListener("pointermove", (event) => showTip(event, [[columns.source, edge.source], [columns.target, edge.target], ...(columns.weight ? [[columns.weight, format.format(edge.weight)]] : [])]));
      return line;
    });
    const dots = nodes.map((node) => {
      const [x, y] = position.get(node.id);
      const right = x >= width / 2;
      const dot = svg("g", { class: `network-node${degree.get(node.id) ? "" : " network-alone"}${named.has(node.id) ? "" : " network-unnamed"}` },
        svg("circle", { cx: x, cy: y, r: 5 + Math.min(4, degree.get(node.id) / 2) }),
        svg("text", { x: x + (right ? 10 : -10), y: y + 4, "text-anchor": right ? "start" : "end" }, node.id));
      dot.addEventListener("pointermove", (event) => showTip(event, [[columns.source.replace(/_?1$/, ""), node.id], ["edges shown", String(degree.get(node.id))]]));
      return dot;
    });
    const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "plot-svg network-svg", role: "img",
      "aria-label": `Network of ${nodes.length} ${columns.source.replace(/_?1$/, "")} nodes, ${shown.length} edges shown` }, ...lines, ...dots);
    figure.replaceChildren(chart, tip);
    const left = [dropped.edges && `${dropped.edges} weaker edges`, dropped.nodes && `${dropped.nodes} nodes`].filter(Boolean).join(" and ");
    caption.textContent = `${shown.length} of ${edges.length} edges between ${nodes.length} nodes` +
      `${named.size < nodes.length ? ` (the ${named.size} most linked named, the others under the pointer)` : ""}` +
      `${columns.weight ? `, from ${measure} ${format.format(threshold)}; the thicker, the stronger${signed ? ", dashed if negative" : ""}` : ""}.` +
      `${left ? ` Not drawn: ${left}.` : ""}${network.truncated ? " The answer has more rows than the table holds: the network is of those shown." : ""}`;
  }
  slider.addEventListener("input", draw);
  draw();
  return box;
}
