// What the network tab shows of an answer whose rows are edges: two columns of
// the same sort (concept_1, concept_2), and the weight of each edge, a numeric
// column. network_view.js draws it.
import { SETTINGS } from "./settings.js";

const NUMERIC = new Set(["int", "float"]);
const { max_nodes: MAX_NODES, max_edges: MAX_EDGES, edges_per_node: EDGES_PER_NODE, layout_steps: STEPS, labels: LABELS } = SETTINGS.network;

// The columns of the ends of the edges of *payload*, and of their weight (-1
// without one), or null: the first two columns of the same sort, one column
// wide, else named as one with the suffixes 1 and 2 (region_1, region_2); the
// weight is the last numeric column of the others. A sort of several columns,
// the voxel (i, j, k) of a map, whose *positions* among them go beyond 0, ends
// no edge.
export function networkColumns({ columns, kinds, sorts = [], positions = [] }) {
  const wide = new Set(sorts.filter((sort, i) => sort && positions[i] > 0));
  const pair = () => {
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        if (sorts[i] && sorts[i] === sorts[j] && !wide.has(sorts[i])) return [i, j];
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

// The network as a hierarchy, each row a node (source) and its parent
// (target), or null if it is none: a tree has an edge fewer than nodes
// (here, at most 5% more, for the nodes of two parents), and a root, a node
// without a parent. The *parent* of each
// node (that of its strongest edge), its *children*, the *roots*, and the
// nodes under each (*descendants*). A node a cycle leaves without a root
// is one more root.
export function hierarchy({ nodes, edges }) {
  if (edges.length > 1.05 * nodes.length) return null;
  const parent = new Map();
  for (const edge of edges) if (!parent.has(edge.source)) parent.set(edge.source, edge.target);
  const roots = nodes.map((node) => node.id).filter((id) => !parent.has(id));
  if (!roots.length) return null;
  const children = new Map(nodes.map((node) => [node.id, []]));
  for (const [child, up] of parent) children.get(up).push(child);
  const order = []; // from the roots down, each node once
  const seen = new Set();
  const visit = (root) => {
    const stack = [root];
    while (stack.length) {
      const id = stack.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      order.push(id);
      stack.push(...[...children.get(id)].reverse());
    }
  };
  roots.forEach(visit);
  for (const node of nodes) {
    if (!seen.has(node.id)) {
      roots.push(node.id);
      visit(node.id);
    }
  }
  // the children of a node are those it was reached from: a tree
  for (const [id, list] of children) children.set(id, list.filter((child) => order.indexOf(child) > order.indexOf(id) && !roots.includes(child)));
  const descendants = new Map(order.map((id) => [id, 0]));
  for (const id of [...order].reverse()) for (const child of children.get(id)) descendants.set(id, descendants.get(id) + 1 + descendants.get(child));
  return { parent, children, roots, descendants };
}

// The positions of the nodes of the *tree* (hierarchy), a radial tree in
// [0, 1] × [0, 1]: the roots at the centre (several around it), each level
// on a circle further out (at the square root of its depth), the leaves
// evenly spaced around, and each node at
// the middle of its leaves.
export function treeLayout({ nodes }, tree) {
  const angle = new Map(), depth = new Map();
  let leaf = 0;
  const leaves = [...tree.descendants].filter(([id]) => !tree.children.get(id).length).length;
  const place = (id, level) => { // iterative: the post-order of the subtree of *id*
    const stack = [[id, level, false]];
    while (stack.length) {
      const [node, d, done] = stack.pop();
      const below = tree.children.get(node);
      depth.set(node, d);
      if (!below.length) angle.set(node, (2 * Math.PI * (leaf++ + 0.5)) / leaves);
      else if (done) angle.set(node, (angle.get(below[0]) + angle.get(below.at(-1))) / 2);
      else stack.push([node, d, true], ...[...below].reverse().map((child) => [child, d + 1, false]));
    }
  };
  const first = tree.roots.length > 1 ? 1 : 0; // several roots are on the first circle
  for (const root of tree.roots) place(root, first);
  const deepest = Math.max(1, ...depth.values());
  return nodes.map(({ id }) => {
    const radius = Math.sqrt(depth.get(id) / deepest); // the inner circles, of fewer nodes, not too small
    return [0.5 + 0.5 * radius * Math.cos(angle.get(id)), 0.5 + 0.5 * radius * Math.sin(angle.get(id))];
  });
}

// What the size or the colour of a node can show: the edges shown it has
// (*links*), the rows it is in each column of (as concept, as parent: the
// children of a parent), and, of a hierarchy, the nodes under it.
export function nodeMeasures({ nodes, columns }, shown, tree) {
  const count = (end) => {
    const counts = new Map(nodes.map((node) => [node.id, 0]));
    for (const edge of shown) counts.set(edge[end], counts.get(edge[end]) + 1);
    return counts;
  };
  const links = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of shown) for (const end of [edge.source, edge.target]) links.set(end, links.get(end) + 1);
  return [
    { name: "links", values: links },
    { name: `rows as ${columns.source}`, values: count("source") },
    { name: `rows as ${columns.target}`, values: count("target") },
    ...(tree ? [{ name: "nodes under it", values: tree.descendants }] : []),
  ];
}

// The nodes linked to *id* by the *edges*, the strongest first: each
// { id, weight }.
export function neighbours(edges, id) {
  return edges
    .filter((edge) => edge.source === id || edge.target === id)
    .map((edge) => ({ id: edge.source === id ? edge.target : edge.source, weight: edge.weight }))
    .sort((a, b) => strength(b) - strength(a) || a.id.localeCompare(b.id));
}

// The names of the *nodes* matching *text*, those it starts first, at most *limit*.
export function findNodes(nodes, text, limit = 20) {
  const wanted = text.trim().toLowerCase();
  if (!wanted) return [];
  const ids = nodes.map((node) => node.id).filter((id) => id.toLowerCase().includes(wanted));
  const starts = (id) => (id.toLowerCase().startsWith(wanted) ? 0 : 1);
  return ids.sort((a, b) => starts(a) - starts(b) || a.localeCompare(b)).slice(0, limit);
}

export const ZOOM = { min: 0.5, max: 16 };

// The view *view* ({ k, x, y }: a point p of the drawing is at k p + (x, y))
// zoomed by *factor* about the point *at* of the figure, which stays put.
export function zoomAt(view, factor, [px, py]) {
  const k = Math.min(ZOOM.max, Math.max(ZOOM.min, view.k * factor));
  const ratio = k / view.k;
  return { k, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
}

// The view that puts the point *point* of the drawing at the centre of a
// figure of *size* [width, height], at the zoom *k*.
export function centreOn([x, y], [width, height], k) {
  return { k, x: width / 2 - k * x, y: height / 2 - k * y };
}
