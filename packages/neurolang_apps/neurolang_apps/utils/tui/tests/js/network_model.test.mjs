import assert from "node:assert/strict";
import test from "node:test";
import { defaultThreshold, layout, networkColumns, networkOf } from "../../../../../../../doc/_static/repl/network_model.js";

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
