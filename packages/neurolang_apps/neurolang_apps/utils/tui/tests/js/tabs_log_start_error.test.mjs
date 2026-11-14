import assert from "node:assert/strict";
import test from "node:test";
import { answerLabel, tabAfter } from "../../../../../../../doc/_static/repl/tabs.js";
import { logSummary } from "../../../../../../../doc/_static/repl/log_view.js";
import { explainStartError } from "../../../../../../../doc/_static/repl/start_error.js";

test("the arrow keys move between the shown tabs, round", () => {
  const tabs = ["table", "brain"];
  assert.equal(tabAfter(tabs, "table", "ArrowRight"), "brain");
  assert.equal(tabAfter(tabs, "brain", "ArrowRight"), "table");
  assert.equal(tabAfter(tabs, "table", "ArrowLeft"), "brain");
  assert.equal(tabAfter(tabs, "brain", "Home"), "table");
  assert.equal(tabAfter(tabs, "table", "End"), "brain");
  assert.equal(tabAfter(tabs, "table", "a"), null);
  assert.equal(tabAfter(["table"], "table", "ArrowRight"), "table");
});

test("an answer's tab is its query's name and its number of rows", () => {
  assert.equal(answerLabel({ title: "terms", total: 12 }, 0), "terms · 12");
  assert.equal(answerLabel({ title: "regions_front", total: 1 }, 1), "regions_front · 1");
  assert.equal(answerLabel({ title: "", total: 0 }, 2), "Answer 3 · 0");
});

test("a folded log is its last line", () => {
  assert.equal(logSummary("Running query...\n516 rows\nQuery completed in 1.76 s\n\n"), "Query completed in 1.76 s");
  assert.equal(logSummary(" \n"), "");
});

test("a start failure is told in plain words, with the message kept", () => {
  const missing = explainStartError("Unexpected token '<', \"<!DOCTYPE \"... is not valid JSON");
  assert.match(missing.summary, /not found on the server/);
  assert.match(missing.detail, /DOCTYPE/);
  assert.match(explainStartError("Failed to fetch").summary, /connection/);
  assert.match(explainStartError("out of memory").summary, /memory/);
  assert.match(explainStartError(undefined).summary, /could not start/);
});
