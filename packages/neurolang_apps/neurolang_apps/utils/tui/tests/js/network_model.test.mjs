import assert from "node:assert/strict";
import test from "node:test";
import { ZOOM, centreOn, defaultThreshold, findNodes, hierarchy, labelled, layout, neighbours, networkColumns, networkOf, nodeMeasures, treeLayout, zoomAt } from "../../../../../../../doc/_static/repl/network_model.js";

const payload = (columns, kinds, sorts, rows) => ({ columns, kinds, sorts, rows, total: rows.length, truncated: false });
const concepts = () => payload(
  ["concept_1", "concept_2", "correlation"], ["str", "str", "float"], ["cogat_term", "cogat_term", null],
  [["fear", "pain", 0.8], ["fear", "reward", -0.3], ["pain", "reward", 0.1], ["memory", "reward", 0.5]],
);

test("an answer of two columns of the same sort is a network, weighted by its numeric column", () => {
  assert.deepEqual(networkColumns(concepts()), { source: 0, target: 1, weight: 2 });
  // named region_1, region_2 when the sorts are unknown; without a weight
  assert.deepEqual(networkColumns(payload(["region_1", "region_2"], ["str", "str"], [null, null], [])), { source: 0, target: 1, weight: -1 });
});

test("the answers that are no network have none", () => {
  const lift = payload(["term", "lift", "studies"], ["str", "float", "int"], ["cogat_term", null, null], [["fear", 2, 40]]);
  const titles = payload(["study", "title"], ["int", "str"], ["study", "text"], [[1, "a"]]);
  assert.equal(networkColumns(lift), null);
  // a map of voxels: i, j and k are of one sort, of three columns
  const map = { ...payload(["i", "j", "k", "lift"], ["int", "int", "int", "float"], ["voxel", "voxel", "voxel", null], [[1, 2, 3, 2.5]]), positions: [0, 1, 2, null] };
  assert.equal(networkColumns(map), null);
  assert.equal(networkOf(map), null);
  const labels = { ...payload(["label_a", "label_b", "r"], ["str", "str", "float"], ["label", "label", null], [["a", "b", 0.5]]), positions: [0, 0, null] };
  assert.deepEqual(networkColumns(labels), { source: 0, target: 1, weight: 2 });
  assert.equal(networkOf(titles), null);
});

test("the network keeps the strongest edges, by the size of their weight, and says what it leaves out", () => {
  const network = networkOf(concepts());
  assert.deepEqual(network.nodes.map((node) => node.id), ["fear", "memory", "pain", "reward"]);
  assert.deepEqual(network.edges.map((edge) => edge.weight), [0.8, 0.5, -0.3, 0.1]);
  assert.deepEqual(network.columns, { source: "concept_1", target: "concept_2", weight: "correlation" });
  const small = networkOf(concepts(), { maxNodes: 3, maxEdges: 1 });
  assert.deepEqual(small.edges.map((edge) => edge.weight), [0.8]);
  assert.deepEqual(small.dropped, { edges: 3, nodes: 2 });
});

test("the edges shown at first are the strongest, two per node", () => {
  const network = networkOf(concepts());
  assert.equal(defaultThreshold(network), 0.1); // 8 edges for 4 nodes: all 4
  const many = payload(["a_1", "a_2", "w"], ["str", "str", "float"], [null, null, null],
    Array.from({ length: 45 }, (_, i) => [`n${i % 10}`, `m${Math.floor(i / 10)}`, i]));
  const big = networkOf(many);
  assert.equal(defaultThreshold(big), 44 - 2 * big.nodes.length + 1);
});

test("the layout is deterministic, within the square, and pulls the linked nodes together", () => {
  const network = networkOf(payload(["a_1", "a_2", "w"], ["str", "str", "float"], [null, null, null],
    [["a", "b", 1], ["c", "d", 1], ["a", "c", 0.01], ["e", "f", 1]]));
  const first = layout(network, 0.5);
  assert.deepEqual(layout(network, 0.5), first);
  for (const [x, y] of first) assert.ok(x >= 0 && x <= 1 && y >= 0 && y <= 1);
  const at = Object.fromEntries(network.nodes.map((node, i) => [node.id, first[i]]));
  const distance = (u, v) => Math.hypot(at[u][0] - at[v][0], at[u][1] - at[v][1]);
  assert.ok(distance("a", "b") < distance("a", "e"));
  assert.ok(distance("c", "d") < distance("c", "f"));
});

test("of many nodes only the most linked are named", () => {
  const degree = new Map([["a", 1], ["b", 3], ["c", 2], ["d", 3]]);
  assert.deepEqual([...labelled(degree, 2)].sort(), ["b", "d"]);
  assert.equal(labelled(degree, 4).size, 4);
});

const ontology = () => networkOf(payload(["concept", "parent"], ["str", "str"], ["cogat_term", "cogat_term"],
  [["memory", "Concept"], ["attention", "Concept"], ["visual memory", "memory"], ["iconic memory", "visual memory"], ["spatial memory", "memory"]]));

test("an answer of nodes and their parents is a hierarchy, with the nodes under each", () => {
  const tree = hierarchy(ontology());
  assert.deepEqual(tree.roots, ["Concept"]);
  assert.deepEqual(tree.children.get("memory").sort(), ["spatial memory", "visual memory"]);
  assert.equal(tree.descendants.get("Concept"), 5);
  assert.equal(tree.descendants.get("memory"), 3);
  assert.equal(tree.descendants.get("iconic memory"), 0);
  // a network of many more edges than nodes is none, nor one whose cycle has no root
  assert.equal(hierarchy(networkOf(payload(["a_1", "a_2"], ["str", "str"], [null, null], [["a", "b"], ["a", "c"], ["b", "c"], ["a", "d"], ["b", "d"], ["c", "d"]]))), null);
  assert.equal(hierarchy(networkOf(payload(["a_1", "a_2"], ["str", "str"], [null, null], [["a", "b"], ["b", "a"]]))), null);
});

test("the radial tree puts the root at the centre and each level on a circle further out", () => {
  const network = ontology();
  const at = Object.fromEntries(network.nodes.map((node, i) => [node.id, treeLayout(network, hierarchy(network))[i]]));
  const radius = (id) => Math.hypot(at[id][0] - 0.5, at[id][1] - 0.5);
  const level = (id, depth) => assert.ok(Math.abs(radius(id) - Math.sqrt(depth / 3) / 2) < 1e-9, id); // 3 levels below the root
  level("Concept", 0);
  ["memory", "attention"].forEach((id) => level(id, 1));
  ["visual memory", "spatial memory"].forEach((id) => level(id, 2));
  level("iconic memory", 3);
});

test("the measures of the nodes count their links, their rows in each column and the nodes under them", () => {
  const network = ontology();
  const measures = Object.fromEntries(nodeMeasures(network, network.edges, hierarchy(network)).map((m) => [m.name, m.values]));
  assert.deepEqual(Object.keys(measures), ["links", "rows as concept", "rows as parent", "nodes under it"]);
  assert.equal(measures["rows as parent"].get("memory"), 2); // its children
  assert.equal(measures.links.get("memory"), 3);
  assert.equal(measures["nodes under it"].get("memory"), 3);
});

test("the neighbours of a node are listed the strongest first", () => {
  const network = networkOf(concepts());
  assert.deepEqual(neighbours(network.edges, "reward"), [
    { id: "memory", weight: 0.5 }, { id: "fear", weight: -0.3 }, { id: "pain", weight: 0.1 },
  ]);
  assert.deepEqual(neighbours(network.edges, "nothing"), []);
});

test("a node is found by a part of its name, those it starts first", () => {
  const nodes = ["visual memory", "memory", "working memory", "attention"].map((id) => ({ id }));
  assert.deepEqual(findNodes(nodes, "Mem"), ["memory", "visual memory", "working memory"]);
  assert.deepEqual(findNodes(nodes, "  "), []);
  assert.deepEqual(findNodes(nodes, "xyz"), []);
});

test("the zoom keeps the point under the pointer in place, within its limits", () => {
  const view = zoomAt({ k: 1, x: 0, y: 0 }, 2, [100, 50]);
  assert.deepEqual(view, { k: 2, x: -100, y: -50 });
  const at = (v, [x, y]) => [v.k * x + v.x, v.k * y + v.y];
  assert.deepEqual(at(view, [100, 50]), [100, 50]);
  assert.equal(zoomAt({ k: 10, x: 0, y: 0 }, 100, [0, 0]).k, ZOOM.max);
  assert.equal(zoomAt({ k: 1, x: 0, y: 0 }, 0.01, [0, 0]).k, ZOOM.min);
  assert.deepEqual(at(centreOn([30, 40], [900, 640], 3), [30, 40]), [450, 320]);
});
