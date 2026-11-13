import assert from "node:assert/strict";
import test from "node:test";
import { SETTINGS, parseSettings } from "../../../../../../../doc/_static/repl/settings.js";

test("the settings are read as YAML: nested mappings, numbers, lists, comments", () => {
  const text = [
    "# a comment",
    "table:",
    "  page_sizes: [20, 50, 100]   # rows per page",
    "  name: \"a # not a comment\"",
    "plot:",
    "  bars: 30",
    "  share: 0.45",
    "  on: true",
    "  none: null",
    "  word: plain text",
  ].join("\n");
  assert.deepEqual(parseSettings(text), {
    table: { page_sizes: [20, 50, 100], name: "a # not a comment" },
    plot: { bars: 30, share: 0.45, on: true, none: null, word: "plain text" },
  });
});

test("the page's settings are those of settings.yaml", () => {
  assert.deepEqual(SETTINGS.table.page_sizes, [20, 50, 100, 500]);
  assert.equal(SETTINGS.plot.bars, 30);
  assert.equal(SETTINGS.layout.head_share, 0.45);
  assert.equal(SETTINGS.history.queries, 500);
});
