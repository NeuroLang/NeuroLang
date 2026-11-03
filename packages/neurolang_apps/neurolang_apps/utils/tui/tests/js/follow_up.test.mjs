import assert from "node:assert/strict";
import test from "node:test";
import { followUpQuery } from "../../../../../../../doc/_static/repl/follow_up.js";

test("the query is given again, its answer a relation, with a new answer on it", () => {
  assert.equal(
    followUpQuery({ columns: ["s", "p"] }, "ans(s, p) :- PROB[a(s)] = p\n"),
    "result(s, p) :- PROB[a(s)] = p\nans(s, p) :- result(s, p), ",
  );
});

test("only the answers' heads are renamed, rules above them stay", () => {
  const query = "a(s) :- b(s), ans2(s)\n  ans(s) :- a(s)";
  assert.equal(followUpQuery({ columns: ["s"] }, query), "a(s) :- b(s), ans2(s)\n  result(s) :- a(s)\nans(s) :- result(s), ");
});

test("without a query that has an answer, in SQUALL, or with odd columns, there is none", () => {
  assert.equal(followUpQuery({ columns: ["s"] }, "a(s) :- b(s)"), null);
  assert.equal(followUpQuery({ columns: ["s"] }, ""), null);
  assert.equal(followUpQuery({ columns: ["s"] }, "ans(s) :- b(s)", true), null);
  assert.equal(followUpQuery({ columns: ["study id"] }, "ans(s) :- b(s)"), null);
  assert.equal(followUpQuery({ columns: [] }, "ans() :- b()"), null);
});

test("it can be asked for again", () => {
  const payload = { columns: ["s"] };
  assert.notEqual(followUpQuery(payload, "ans(s) :- b(s)"), null);
  assert.notEqual(followUpQuery(payload, "ans(s) :- b(s)"), null);
});
