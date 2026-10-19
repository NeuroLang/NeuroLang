import assert from "node:assert/strict";
import test from "node:test";
import { TableModel } from "../../../../../doc/_static/repl/table_model.js";

const payload = (rows, kinds = ["str", "int"], columns = ["name", "n"]) => ({
  title: "", columns, kinds, rows, total: rows.length, truncated: false,
});
const people = () => new TableModel(payload([["bob", 5], ["alice", 3], ["carol", 10], ["dave", null]]));
const names = (model) => model.page(0, 100).rows.map((r) => r.cells[0]);

test("rows come in their order, numbered from 1", () => {
  const { rows } = people().page(0, 10);
  assert.deepEqual(rows.map((r) => r.number), [1, 2, 3, 4]);
});

test("a click sorts ascending, then descending, then not at all", () => {
  const model = people();
  model.toggleSort(0);
  assert.deepEqual(names(model), ["alice", "bob", "carol", "dave"]);
  model.toggleSort(0);
  assert.deepEqual(names(model), ["dave", "carol", "bob", "alice"]);
  model.toggleSort(0);
  assert.deepEqual(names(model), ["bob", "alice", "carol", "dave"]);
  assert.equal(model.sort, null);
});

test("numbers sort as numbers, and missing values come last either way", () => {
  const model = people();
  model.toggleSort(1);
  assert.deepEqual(names(model), ["alice", "bob", "carol", "dave"]);
  model.toggleSort(1);
  assert.deepEqual(names(model), ["carol", "bob", "alice", "dave"]);
});

test("text with numbers sorts naturally", () => {
  const model = new TableModel(payload([["x10"], ["x9"], ["x100"]], ["str"], ["id"]));
  model.toggleSort(0);
  assert.deepEqual(names(model), ["x9", "x10", "x100"]);
});

test("infinities sort as numbers", () => {
  const model = new TableModel(payload([[1], ["Infinity"], [-2]], ["float"], ["v"]));
  model.toggleSort(0);
  assert.deepEqual(model.page(0, 9).rows.map((r) => r.cells[0]), [-2, 1, "Infinity"]);
});

test("the filter keeps the rows with every word, in any cell", () => {
  const model = people();
  model.setFilter("A 3");
  assert.deepEqual(names(model), ["alice"]);
  model.setFilter("");
  assert.equal(model.visibleCount, 4);
});

test("column:text filters one column", () => {
  const model = new TableModel(payload([["bob", "bob"], ["alice", "bob"]], ["str", "str"], ["name", "friend"]));
  model.setFilter("name:bob");
  assert.deepEqual(names(model), ["bob"]);
  model.setFilter("fri:bob");
  assert.equal(model.visibleCount, 2);
});

test("a filter and a sort go together, and the rows keep their numbers", () => {
  const model = people();
  model.setFilter("o");
  model.toggleSort(1);
  const rows = model.page(0, 10).rows;
  assert.deepEqual(rows.map((r) => [r.number, r.cells[0]]), [[1, "bob"], [3, "carol"]]);
});

test("pages have a size, and the page asked for is kept in range", () => {
  const model = new TableModel(payload(Array.from({ length: 25 }, (_, i) => [i]), ["int"], ["i"]));
  assert.equal(model.pageCount(10), 3);
  const last = model.page(7, 10);
  assert.equal(last.index, 2);
  assert.deepEqual([last.start, last.end], [20, 25]);
  assert.equal(new TableModel(payload([], ["int"], ["i"])).pageCount(10), 1);
});

test("the range of a numeric column ignores missing values", () => {
  assert.deepEqual(people().range(1), { low: 3, high: 10 });
  assert.equal(people().range(0), null);
});

test("copies and downloads hold the visible rows, in order", () => {
  const model = people();
  model.setFilter("a");
  model.toggleSort(1);
  assert.equal(model.toTSV(), "name\tn\nalice\t3\ncarol\t10\ndave\t\n");
  assert.equal(model.toCSV(), "name,n\nalice,3\ncarol,10\ndave,\n");
});

test("CSV quotes what needs it", () => {
  const model = new TableModel(payload([['a,"b"'], ["line\nbreak"]], ["str"], ["text"]));
  assert.equal(model.toCSV(), 'text\n"a,""b"""\n"line\nbreak"\n');
});
