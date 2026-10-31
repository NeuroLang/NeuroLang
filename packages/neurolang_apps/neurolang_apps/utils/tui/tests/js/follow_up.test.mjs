import assert from "node:assert/strict";
import test from "node:test";
import { followUps } from "../../../../../../../doc/_static/repl/follow_up.js";

const texts = (...args) => followUps(...args).map((action) => action.text);

test("the query is given again, its answer a relation, with a new answer on it", () => {
  assert.deepEqual(texts({ columns: ["s", "p"] }, "ans(s, p) :- PROB[a(s)] = p\n"), [
    "result(s, p) :- PROB[a(s)] = p\nans(s, p) :- result(s, p), ",
  ]);
});

test("only the answers' heads are renamed, rules above them stay", () => {
  const query = "a(s) :- b(s), ans2(s)\n  ans(s) :- a(s)";
  assert.deepEqual(texts({ columns: ["s"] }, query), ["a(s) :- b(s), ans2(s)\n  result(s) :- a(s)\nans(s) :- result(s), "]);
});

test("without a query that has an answer, in SQUALL, or with odd columns, there are none", () => {
  assert.deepEqual(followUps({ columns: ["s"] }, "a(s) :- b(s)"), []);
  assert.deepEqual(followUps({ columns: ["s"] }, ""), []);
  assert.deepEqual(followUps({ columns: ["s"] }, "ans(s) :- b(s)", { squall: true }), []);
  assert.deepEqual(followUps({ columns: ["study id"] }, "ans(s) :- b(s)"), []);
  assert.deepEqual(followUps({ columns: [] }, "ans() :- b()"), []);
});

test("they can be asked for again", () => {
  const payload = { columns: ["s"] };
  assert.equal(followUps(payload, "ans(s) :- b(s)").length, 1);
  assert.equal(followUps(payload, "ans(s) :- b(s)").length, 1);
});

test("a map of the studies counts, by voxel, those that report a peak there", () => {
  const [, map] = followUps({ columns: ["s"], kinds: ["int"] }, "ans(s) :- mentions(s, 'pain')", { peaks: true });
  assert.equal(map.label, "Map the studies");
  assert.equal(map.text, "result(s) :- mentions(s, 'pain')\n" +
    "ans(i, j, k, n) :- AGGREGATE[i, j, k](result(s), peak_reported(i, j, k, s) @ count(s)) = n");
});

test("the map is offered by study column, with names no column has", () => {
  const payload = { columns: ["i", "n", "term"], kinds: ["int", "int", "str"] };
  const actions = followUps(payload, "ans(i, n, term) :- a(i, n, term)", { peaks: true });
  assert.deepEqual(actions.map((a) => a.label), ["Use in a query", "Map the studies in i", "Map the studies in n"]);
  assert.match(actions[1].text, /^ans\(i_, j, k, n_\) :- AGGREGATE\[i_, j, k\]\(result\(i, n, term\), peak_reported\(i_, j, k, i\) @ count\(i\)\) = n_$/m);
});

test("no map without the peaks, or without a column of integers", () => {
  assert.equal(followUps({ columns: ["s"], kinds: ["int"] }, "ans(s) :- b(s)").length, 1);
  assert.equal(followUps({ columns: ["t"], kinds: ["str"] }, "ans(t) :- b(t)", { peaks: true }).length, 1);
});
