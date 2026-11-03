import assert from "node:assert/strict";
import test from "node:test";
import { Session, definesRules, parseSession, replayTexts } from "../../../../../../../doc/_static/repl/session.js";

test("a rule is told from a query by its head", () => {
  assert.equal(definesRules("similar(s) :- study(s), v > 1"), true);
  assert.equal(definesRules("ans(s) :- study(s)"), false);
  assert.equal(definesRules("a(x) ← b(x)\nans(x) :- a(x)"), true);
  assert.equal(definesRules("q(x, PROB) :- r(x)"), true);
  assert.equal(definesRules("% a(x) is a comment\nans(x) :- r(x)"), false);
  assert.equal(definesRules("define as Popular every Study that mentions pain.", "squall"), true);
  assert.equal(definesRules("obtain every Study.", "squall"), false);
});

test("the session keeps what was run, not the commands, and is bounded", () => {
  const session = new Session();
  session.add("  a(x) :- b(x)  ", "datalog");
  session.add(".help", "datalog");
  session.add("   ", "datalog");
  session.add("x", "klingon");
  assert.deepEqual(session.entries, [{ text: "a(x) :- b(x)", mode: "datalog" }]);
  for (let i = 0; i < 600; i += 1) session.add(`r${i}(x) :- b(x)`, "datalog");
  assert.equal(session.entries.length, 500);
  assert.equal(session.entries.at(-1).text, "r599(x) :- b(x)");
});

test("the rules are run again in their language, and the language is given back", () => {
  const entries = [
    { text: "a(x) :- b(x)", mode: "datalog" },
    { text: "ans(x) :- a(x)", mode: "datalog" },
    { text: "define as Popular every Study.", mode: "squall" },
    { text: "c(x) :- a(x)", mode: "datalog" },
  ];
  assert.deepEqual(replayTexts(entries, "datalog", { clear: true }), [
    ".clear", ".mode datalog", "a(x) :- b(x)", ".mode squall", "define as Popular every Study.",
    ".mode datalog", "c(x) :- a(x)",
  ]);
  assert.deepEqual(replayTexts([], "squall"), [".mode squall"]);
  assert.deepEqual(replayTexts([{ text: "ans(x) :- a(x)", mode: "datalog" }], null), []);
});

test("a saved session is read back, and a wrong file says why", () => {
  const session = new Session();
  session.add("a(x) :- b(x)", "datalog");
  const saved = session.serialise("ans(x) :- a(x)", "datalog");
  assert.deepEqual(parseSession(saved), {
    entries: [{ text: "a(x) :- b(x)", mode: "datalog" }], box: "ans(x) :- a(x)", mode: "datalog",
  });
  assert.throws(() => parseSession("nope"), /not JSON/);
  assert.throws(() => parseSession("{}"), /not a saved NeuroLang session/);
  assert.throws(() => parseSession(JSON.stringify({ format: "neurolang-repl-session", version: 1, entries: [{ text: 1, mode: "datalog" }] })), /not valid/);
  assert.throws(() => parseSession(JSON.stringify({ format: "neurolang-repl-session", version: 1, entries: [{ text: "a", mode: "x" }] })), /not valid/);
});
