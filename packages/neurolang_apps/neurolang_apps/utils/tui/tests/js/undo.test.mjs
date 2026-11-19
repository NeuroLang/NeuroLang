import assert from "node:assert/strict";
import test from "node:test";
import { TextHistory, editKind } from "../../../../../../../doc/_static/repl/undo.js";

const at = (text) => ({ text, start: text.length, end: text.length });
const typed = (history, text, time, kind = "type") => history.record(at(text), kind, time);

test("typing without a pause is one step, a pause starts another", () => {
  const h = new TextHistory(at(""));
  ["a", "ab", "abc"].forEach((text, i) => typed(h, text, i * 100));
  typed(h, "abc d", 1500);
  assert.equal(h.undo().text, "abc");
  assert.equal(h.undo().text, "");
  assert.equal(h.undo(), null);
  assert.equal(h.redo().text, "abc");
  assert.equal(h.redo().text, "abc d");
  assert.equal(h.redo(), null);
});

test("deleting after typing, and a paste, are steps of their own", () => {
  const h = new TextHistory(at(""));
  typed(h, "ab", 0);
  typed(h, "a", 100, editKind("deleteContentBackward"));
  typed(h, "a x", 200, editKind("insertFromPaste"));
  typed(h, "a xy", 300);
  assert.deepEqual([h.undo(), h.undo(), h.undo(), h.undo()].map((s) => s.text), ["a x", "a", "ab", ""]);
});

test("what the page wrote is a step, which undo brings back", () => {
  const h = new TextHistory(at(""));
  h.record(at("ans(x) :- p(x)")); // an example put in the box
  typed(h, "ans(x) :- p(x) ", 0);
  assert.equal(h.undo().text, "ans(x) :- p(x)");
  assert.equal(h.undo().text, "");
});

test("an edit after undo forgets the steps undone, and the oldest go", () => {
  const h = new TextHistory(at(""), 3);
  ["a", "b", "c", "d"].forEach((text) => h.record(at(text)));
  assert.deepEqual(h.states.map((s) => s.text), ["b", "c", "d"]);
  h.undo();
  h.record(at("e"));
  assert.equal(h.redo(), null);
  assert.deepEqual(h.states.map((s) => s.text), ["b", "c", "e"]);
});

test("the kinds of edits", () => {
  assert.equal(editKind("insertText"), "type");
  assert.equal(editKind("insertLineBreak"), "type");
  assert.equal(editKind("deleteWordBackward"), "delete");
  assert.equal(editKind("deleteByCut"), null);
  assert.equal(editKind("insertFromPaste"), null);
});
