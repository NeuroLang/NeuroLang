import assert from "node:assert/strict";
import test from "node:test";
import { History } from "../../../../../doc/_static/repl/history.js";

const memory = () => {
  const store = new Map();
  return { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
};
const history = (...entries) => {
  const h = new History();
  entries.forEach((e) => h.add(e));
  return h;
};

test("empty entries and immediate repeats are not kept", () => {
  const h = history("a", "a", "  ", "b", "a");
  assert.deepEqual(h.entries, ["a", "b", "a"]);
});

test("the history lasts between visits and is limited", () => {
  const storage = memory();
  const first = new History(storage, "k", 3);
  ["1", "2", "3", "4"].forEach((e) => first.add(e));
  assert.deepEqual(new History(storage, "k", 3).entries, ["2", "3", "4"]);
});

test("a broken or foreign saved history is ignored", () => {
  assert.deepEqual(new History({ getItem: () => "{nope", setItem() {} }).entries, []);
  assert.deepEqual(new History({ getItem: () => '{"a":1}', setItem() {} }).entries, []);
  assert.deepEqual(new History({ getItem: () => '["a", 3]', setItem() {} }).entries, ["a"]);
});
