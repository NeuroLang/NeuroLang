// What the network tab shows of an answer whose rows are edges: two columns of
// the same sort (concept_1, concept_2), and the weight of each edge, a numeric
// column. network_view.js draws it.
import { SETTINGS } from "./settings.js";

const NUMERIC = new Set(["int", "float"]);
const { max_nodes: MAX_NODES, max_edges: MAX_EDGES, edges_per_node: EDGES_PER_NODE, layout_steps: STEPS, labels: LABELS } = SETTINGS.network;

// The columns of the ends of the edges of *payload*, and of their weight (-1
// without one), or null: the first two columns of the same sort, else named
// as one with the suffixes 1 and 2 (region_1, region_2); the weight is the
// last numeric column of the others.
export function networkColumns({ columns, kinds, sorts = [] }) {
  const pair = () => {
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        if (sorts[i] && sorts[i] === sorts[j]) return [i, j];
      }
    }
    for (let i = 0; i < columns.length; i++) {
      const stem = /^(.+?)_?1$/.exec(columns[i])?.[1];
      const j = stem ? columns.findIndex((name) => name === `${stem}_2` || name === `${stem}2`) : -1;
      if (j > i && !sorts[i] && !sorts[j]) return [i, j];
    }
    return null;
  };
  const found = pair();
  if (!found) return null;
  const [source, target] = found;
  const others = columns.map((_, column) => column).filter((column) => column !== source && column !== target);
  const weight = others.filter((column) => NUMERIC.has(kinds[column])).at(-1) ?? -1;
  return { source, target, weight };
}

const strength = (edge) => Math.abs(edge.weight);
const GRAVITY = 6; // the pull to the centre, against the push of the nodes

// The network of *payload*, or null if it is none: its nodes, its edges
// strongest first, and how many of the answer's edges and nodes it leaves out
// (*dropped*) beyond the most it draws.
export function networkOf(payload, { maxNodes = MAX_NODES, maxEdges = MAX_EDGES } = {}) {
  const found = networkColumns(payload);
  if (!found) return null;
  const { source, target, weight } = found;
  let edges = payload.rows
    .map((row) => ({ source: String(row[source]), target: String(row[target]), weight: weight < 0 ? 1 : Number(row[weight]) }))
    .filter((edge) => edge.source !== edge.target && Number.isFinite(edge.weight));
  if (!edges.length) return null;
  edges.sort((a, b) => strength(b) - strength(a));
  const total = { edges: edges.length, nodes: new Set(edges.flatMap((e) => [e.source, e.target])).size };
  // The nodes of the strongest edges, the most it draws.
  const kept = new Set();
  for (const edge of edges) {
    for (const end of [edge.source, edge.target]) if (kept.size < maxNodes) kept.add(end);
  }
  edges = edges.filter((edge) => kept.has(edge.source) && kept.has(edge.target)).slice(0, maxEdges);
  const nodes = [...new Set(edges.flatMap((e) => [e.source, e.target]))].sort().map((id) => ({ id }));
  return {
    columns: { source: payload.columns[source], target: payload.columns[target], weight: weight < 0 ? null : payload.columns[weight] },
    nodes, edges,
    dropped: { edges: total.edges - edges.length, nodes: total.nodes - nodes.length },
    truncated: Boolean(payload.truncated),
  };
}

// The weight from which an edge is drawn at first: that of the
// EDGES_PER_NODE × nodes strongest edges, all of a sparse network.
export function defaultThreshold({ nodes, edges }) {
  const shown = Math.min(edges.length, Math.max(1, Math.round(EDGES_PER_NODE * nodes.length)));
  return strength(edges[shown - 1]);
}

// The nodes whose names are written, of their *degree* (a Map): all, or of
// more than *limit* nodes the *limit* most linked (the others on hover).
export function labelled(degree, limit = LABELS) {
  const ids = [...degree.keys()];
  if (ids.length <= limit) return new Set(ids);
  return new Set(ids.sort((a, b) => degree.get(b) - degree.get(a) || a.localeCompare(b)).slice(0, limit));
}

// The positions of the *nodes*, in [0, 1] × [0, 1]: a force layout
// (Fruchterman and Reingold) from a circle, deterministic, where the edges
// of positive weight from *threshold* pull their ends together, the more
// the stronger, and the nodes push each other apart.
export function layout({ nodes, edges }, threshold, steps = STEPS) {
  const n = nodes.length;
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const at = nodes.map((_, i) => [0.5 + 0.4 * Math.cos((2 * Math.PI * i) / n), 0.5 + 0.4 * Math.sin((2 * Math.PI * i) / n)]);
  if (n < 3) return at;
  const pulling = edges.filter((edge) => edge.weight > 0 && strength(edge) >= threshold);
  const top = Math.max(...pulling.map(strength), 1e-12);
  const springs = pulling.map((edge) => [index.get(edge.source), index.get(edge.target), edge.weight / top]);
  const k = Math.sqrt(1 / n); // the distance at which pull and push balance
  for (let step = 0; step < steps; step++) {
    const heat = 0.1 * (1 - step / steps);
    const move = at.map(() => [0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = at[i][0] - at[j][0], dy = at[i][1] - at[j][1];
        const d2 = Math.max(dx * dx + dy * dy, 1e-6);
        const push = (k * k) / d2;
        move[i][0] += dx * push; move[i][1] += dy * push;
        move[j][0] -= dx * push; move[j][1] -= dy * push;
      }
    }
    for (const [i, j, w] of springs) {
      const dx = at[i][0] - at[j][0], dy = at[i][1] - at[j][1];
      const pull = (w * Math.sqrt(dx * dx + dy * dy)) / k;
      move[i][0] -= dx * pull; move[i][1] -= dy * pull;
      move[j][0] += dx * pull; move[j][1] += dy * pull;
    }
    for (let i = 0; i < n; i++) {
      // a pull to the centre keeps the nodes of no edge, and the parts, near
      const [cx, cy] = [GRAVITY * (0.5 - at[i][0]), GRAVITY * (0.5 - at[i][1])];
      const [dx, dy] = [move[i][0] + cx, move[i][1] + cy];
      const length = Math.hypot(dx, dy) || 1;
      const by = Math.min(length, heat);
      at[i] = [at[i][0] + (dx / length) * by, at[i][1] + (dy / length) * by];
    }
  }
  // Scaled to fill the square.
  const xs = at.map((p) => p[0]), ys = at.map((p) => p[1]);
  const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
  const span = Math.max(Math.max(...xs) - x0, Math.max(...ys) - y0, 1e-9);
  return at.map(([x, y]) => [(x - x0) / span, (y - y0) / span]);
}
