import assert from "node:assert/strict";
import test from "node:test";
import { TableModel } from "../../../../../../../doc/_static/repl/table_model.js";
import { barData, histogram, kindsOf, niceTicks, plotChoice, pointData } from "../../../../../../../doc/_static/repl/plot_model.js";

const payload = (rows, kinds, columns) => ({ columns, kinds, rows, total: rows.length, truncated: false });
const regions = () => new TableModel(payload(
  [["insula", 0.4], ["front sup", 0.9], ["precentral", 0.2], ["orbital", null]], ["str", "float"], ["r", "p"],
));
const peaks = () => new TableModel(payload(
  [[1, 10.5, -3], [2, 20, 4], [3, 15, 1]], ["int", "float", "int"], ["study", "x", "y"],
));

test("the kinds of plot are those the columns allow", () => {
  assert.deepEqual(kindsOf(regions()), ["bar", "histogram"]);
  assert.deepEqual(kindsOf(peaks()), ["bar", "histogram", "scatter", "line"]);
  assert.deepEqual(kindsOf(new TableModel(payload([["a"]], ["str"], ["s"]))), ["bar"]);
});

test("a result of probabilities is plotted as bars of them, a numeric one as a scatter", () => {
  assert.deepEqual(plotChoice(regions()), { kind: "bar", x: 0, y: 1 });
  assert.deepEqual(plotChoice(peaks()), { kind: "scatter", x: 1, y: 2 });
  assert.deepEqual(plotChoice(new TableModel(payload([[0.5], [2]], ["float"], ["w"]))), { kind: "histogram", x: 0, y: -1 });
  // Ids beside text (studies and their terms): how many rows each term has, not a histogram of ids.
  assert.deepEqual(plotChoice(new TableModel(payload([[1, "pain"], [2, "pain"]], ["int", "str"], ["s", "t"]))), { kind: "bar", x: 1, y: -1 });
  // Text only: how many rows each value has.
  assert.deepEqual(plotChoice(new TableModel(payload([["a"], ["b"], ["a"]], ["str"], ["s"]))), { kind: "bar", x: 0, y: -1 });
});

test("bars follow the order of the table, the first ones only, and the largest is marked", () => {
  const model = regions(); // sorted by p, the most probable first
  const { bars, more } = barData(model, 0, 1, 2);
  assert.deepEqual(bars.map((bar) => [bar.label, bar.value]), [["front sup", 0.9], ["insula", 0.4]]);
  assert.equal(more, 1); // precentral; the missing value is not a bar
  assert.deepEqual(bars.map((bar) => bar.top), [true, false]);
});

test("bars without a value column count the rows of each value, most first", () => {
  const model = new TableModel(payload([["a"], ["b"], ["a"], ["c"], ["a"], ["b"]], ["str"], ["s"]));
  const { bars, more } = barData(model, 0, -1, 2);
  assert.deepEqual(bars.map((bar) => [bar.label, bar.value]), [["a", 3], ["b", 2]]);
  assert.equal(more, 1);
});

test("the rows the filter keeps are those plotted", () => {
  const model = regions();
  model.setFilter("a"); // insula, precentral and orbital, whose p is missing
  assert.deepEqual(barData(model, 0, 1).bars.map((bar) => bar.label), ["insula", "precentral"]);
  assert.deepEqual(histogram(model, 1, 2).bins.reduce((sum, bin) => sum + bin.count, 0), 2);
});

test("a histogram has bins of equal width over the values, the largest in the last", () => {
  const { bins, low, high } = histogram(peaks(), 1, 2);
  assert.equal(low, 10.5);
  assert.equal(high, 20);
  assert.deepEqual(bins.map((bin) => bin.count), [2, 1]);
  assert.deepEqual(bins.map((bin) => [bin.x0, bin.x1]), [[10.5, 15.25], [15.25, 20]]);
  // A single value is one bin.
  const one = histogram(new TableModel(payload([[2], [2]], ["int"], ["n"])), 0);
  assert.deepEqual(one.bins.map((bin) => bin.count), [2]);
});

test("points pair two numeric columns, missing values left out, a line in order of x", () => {
  const model = new TableModel(payload([[3, 1], [1, null], [2, 5], [1, 2]], ["int", "int"], ["a", "b"]));
  assert.deepEqual(pointData(model, 0, 1).map(({ x, y }) => [x, y]), [[3, 1], [2, 5], [1, 2]]);
  assert.deepEqual(pointData(model, 0, 1, { line: true }).map(({ x, y }) => [x, y]), [[1, 2], [2, 5], [3, 1]]);
});

test("the ticks of an axis are round numbers covering the values", () => {
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(niceTicks(3, 97, 5), [0, 20, 40, 60, 80, 100]);
  assert.deepEqual(niceTicks(-3, 4, 4), [-4, -2, 0, 2, 4]);
  assert.deepEqual(niceTicks(5, 5, 5), [3.5, 4, 4.5, 5, 5.5, 6, 6.5]); // a single value is widened
});
