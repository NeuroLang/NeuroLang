// What a plot of a result shows: bars, a histogram, points or a line, from
// the columns of its table (table_model.js), of the rows its filter keeps,
// in the table's order. plot_view.js draws them.
import { SETTINGS } from "./settings.js";
import { cellText } from "./table_model.js";

const NUMERIC = new Set(["int", "float"]);
const numericColumns = (model) => model.columns.map((_, column) => column).filter((column) => NUMERIC.has(model.kinds[column]));
const shownRows = (model) => model.order.map((index) => model.rows[index]);
const isNumber = (value) => value !== null && value !== "" && Number.isFinite(Number(value));

// The kinds of plot of the table: bars always, a histogram of a numeric
// column, points and lines of two.
export function kindsOf(model) {
  const numeric = numericColumns(model).length;
  return ["bar", ...(numeric ? ["histogram"] : []), ...(numeric > 1 ? ["scatter", "line"] : [])];
}

// The plot a table starts with: its probabilities as bars, by the first other
// column; the last two numeric columns as points (ids come first); the one
// numeric column as a histogram, unless it is whole numbers beside text
// (ids); or how many rows each value of the first text column has. *y* is -1 when the plot has no value column.
export function plotChoice(model) {
  if (model.probability >= 0) {
    const label = model.columns.findIndex((_, column) => column !== model.probability);
    if (label >= 0) return { kind: "bar", x: label, y: model.probability };
  }
  const numeric = numericColumns(model);
  if (numeric.length > 1) return { kind: "scatter", x: numeric.at(-2), y: numeric.at(-1) };
  // A column of whole numbers beside text is more likely ids than measures.
  const text = model.columns.findIndex((_, column) => !NUMERIC.has(model.kinds[column]));
  if (numeric.length === 1 && (model.kinds[numeric[0]] === "float" || text < 0)) return { kind: "histogram", x: numeric[0], y: -1 };
  return { kind: "bar", x: Math.max(0, text), y: -1 };
}

// The bars of the column *y* labelled by the column *x*, the first *limit* in
// the table's order, and how many *more* there are; without *y* (-1), how
// many rows each value of *x* has, most first. The largest is *top*.
export function barData(model, x, y, limit = SETTINGS.plot.bars) {
  let all;
  if (y >= 0) {
    all = shownRows(model).filter((row) => isNumber(row[y])).map((row) => ({ label: cellText(row[x]), value: Number(row[y]) }));
  } else {
    const counts = new Map();
    for (const row of shownRows(model)) {
      const label = cellText(row[x]);
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    all = [...counts].map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
  }
  const largest = Math.max(...all.map((bar) => bar.value));
  const bars = all.slice(0, limit).map((bar) => ({ ...bar, top: bar.value === largest }));
  return { bars, more: all.length - bars.length };
}

// The histogram of the numeric column *x*: *count* bins of equal width from
// the smallest value, *low*, to the largest, *high* (by default Sturges'
// number of them), each with the rows from *x0* up to *x1*, the last with *x1*.
export function histogram(model, x, count = 0) {
  const values = shownRows(model).filter((row) => isNumber(row[x])).map((row) => Number(row[x]));
  if (!values.length) return { bins: [], low: 0, high: 0 };
  const low = Math.min(...values), high = Math.max(...values);
  const number = high === low ? 1 : count || Math.min(SETTINGS.plot.histogram_bins, Math.max(1, Math.ceil(Math.log2(values.length)) + 1));
  const width = (high - low) / number;
  const bins = Array.from({ length: number }, (_, index) => ({
    x0: low + index * width, x1: index === number - 1 ? high : low + (index + 1) * width, count: 0,
  }));
  for (const value of values) bins[Math.min(number - 1, width ? Math.floor((value - low) / width) : 0)].count += 1;
  return { bins, low, high };
}

// The points (*x*, *y*) of two numeric columns, each with its *row*; for a
// *line*, in the order of x.
export function pointData(model, x, y, { line = false } = {}) {
  const points = shownRows(model).filter((row) => isNumber(row[x]) && isNumber(row[y]))
    .map((row) => ({ x: Number(row[x]), y: Number(row[y]), row }));
  return line ? points.sort((a, b) => a.x - b.x) : points;
}

// Round numbers from below *low* to above *high*, about *count* of them, as an axis is marked.
export function niceTicks(low, high, count = 5) {
  if (high === low) {
    const pad = Math.abs(low) / 4 || 1;
    [low, high] = [low - pad, high + pad];
  }
  const raw = (high - low) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = magnitude * ([1, 2, 5, 10].find((nice) => nice >= raw / magnitude - 1e-9) ?? 10);
  const first = Math.floor(low / step + 1e-9), last = Math.ceil(high / step - 1e-9);
  return Array.from({ length: last - first + 1 }, (_, index) => Number(((first + index) * step).toPrecision(12)));
}
