import assert from "node:assert/strict";
import test from "node:test";
import { History, ReverseSearch } from "../../../../../doc/_static/repl/history.js";

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

test("a search finds the most recent entry holding the text, ignoring case", () => {
  const search = new ReverseSearch(history("ans(s) :- mentions(s, 'pain')", "ans(i) :- voxel(i)", "ANS(t) :- Term(t)"));
  assert.equal(search.type("mentions"), true);
  assert.equal(search.text, "ans(s) :- mentions(s, 'pain')");
  assert.equal(search.type("ans"), true);
  assert.equal(search.text, "ANS(t) :- Term(t)");
});

test("again goes to older matches, and the other way to newer ones", () => {
  const search = new ReverseSearch(history("a1", "b", "a2", "c", "a3"));
  search.type("a");
  assert.equal(search.text, "a3");
  assert.equal(search.older(), true);
  assert.equal(search.text, "a2");
  assert.equal(search.older(), true);
  assert.equal(search.text, "a1");
  assert.equal(search.older(), false); // none older: stays
  assert.equal(search.text, "a1");
  assert.equal(search.newer(), true);
  assert.equal(search.text, "a2");
});

test("no match leaves the draft", () => {
  const search = new ReverseSearch(history("a"), "my draft");
  assert.equal(search.type("zzz"), false);
  assert.equal(search.text, "my draft");
});

test("typing more of the query searches again from the newest", () => {
  const search = new ReverseSearch(history("abc", "abx", "xyz"));
  search.type("ab");
  search.older();
  search.type("abc");
  assert.equal(search.text, "abc");
});

test("an empty query shows the newest entry, then older ones", () => {
  const search = new ReverseSearch(history("one", "two"));
  search.type("");
  assert.equal(search.text, "two");
  search.older();
  assert.equal(search.text, "one");
});

test("a query that no longer matches goes back to the draft", () => {
  const search = new ReverseSearch(history("abc"), "draft");
  search.type("ab");
  assert.equal(search.text, "abc");
  assert.equal(search.type("abz"), false);
  assert.equal(search.text, "draft");
});
