import assert from "node:assert/strict";
import test from "node:test";
import { AskedHistory } from "../../../../../../../doc/_static/repl/asked.js";

class Storage {
  constructor() { this.items = {}; }
  getItem(key) { return this.items[key] ?? null; }
  setItem(key, value) { this.items[key] = String(value); }
}

const answer = (question, query, extra = {}) => ({ question, query, language: "datalog", valid: true, explanation: "", ...extra });

test("a question is kept with the query it gave, between visits", () => {
  const storage = new Storage();
  const asked = new AskedHistory(storage, () => 1000);
  asked.add(answer("which studies mention pain?", "ans(s) :- mentions(s, 'pain')", { explanation: "The studies." }));
  const again = new AskedHistory(storage);
  assert.deepEqual(again.entries, [{
    question: "which studies mention pain?", query: "ans(s) :- mentions(s, 'pain')",
    language: "datalog", valid: true, explanation: "The studies.", at: 1000,
  }]);
});

test("the newest come first; the same answer twice in a row is kept once", () => {
  const asked = new AskedHistory(new Storage());
  asked.add(answer("a?", "ans(x) :- a(x)"));
  asked.add(answer("a?", "ans(x) :- a(x)"));
  asked.add(answer("b?", "ans(x) :- b(x)"));
  assert.deepEqual(asked.newest().map((entry) => entry.question), ["b?", "a?"]);
});

test("a question with no valid query is kept, to ask again", () => {
  const asked = new AskedHistory(new Storage());
  asked.add(answer("nonsense?", "", { valid: false }));
  assert.equal(asked.newest()[0].valid, false);
  asked.add(answer("", "ans(x) :- a(x)")); // no question: not kept
  assert.equal(asked.entries.length, 1);
});

test("the oldest go past the limit; a broken or refused storage keeps nothing", () => {
  const asked = new AskedHistory(new Storage(), Date.now, 2);
  ["a?", "b?", "c?"].forEach((question) => asked.add(answer(question, "q")));
  assert.deepEqual(asked.newest().map((entry) => entry.question), ["c?", "b?"]);
  const broken = new Storage();
  broken.setItem("neurolang-repl-asked", "{not json");
  assert.deepEqual(new AskedHistory(broken).entries, []);
  const refused = { getItem() { throw new Error("no"); }, setItem() { throw new Error("no"); } };
  const none = new AskedHistory(refused);
  none.add(answer("a?", "q"));
  assert.equal(none.entries.length, 1); // in this visit still
});
