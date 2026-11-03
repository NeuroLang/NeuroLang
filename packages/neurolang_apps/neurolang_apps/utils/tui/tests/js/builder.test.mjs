import assert from "node:assert/strict";
import test from "node:test";
import { TEMPLATES, availableTemplates, buildQuery, quote } from "../../../../../../../doc/_static/repl/builder.js";

const template = (id) => TEMPLATES.find((t) => t.id === id);

test("a term is a string literal, in the quotes it does not have", () => {
  assert.equal(quote("pain"), '"pain"');
  assert.equal(quote('say "hi"'), `'say "hi"'`);
  assert.equal(quote("a\\b"), '"a\\b"');
  assert.throws(() => quote(`it's "it"`), /both kinds of quotes/);
  assert.throws(() => quote("a\nb"), /line break/);
});

test("the queries of the questions", () => {
  assert.equal(buildQuery(template("studies-of-term"), { term: " pain " }), 'ans(s) :- mentions(s, "pain")');
  assert.equal(buildQuery(template("studies-of-both"), { first: "pain", second: "memory" }), 'ans(s) :- mentions(s, "pain"), mentions(s, "memory")');
  assert.equal(buildQuery(template("studies-of-one-not-other"), { first: "pain", second: "memory" }), 'ans(s) :- mentions(s, "pain"), ~mentions(s, "memory")');
  assert.equal(buildQuery(template("studies-by-weight"), { term: "pain", weight: "0.2" }), 'ans(s, w) :- weighted_mention(s, "pain", w), (w > 0.2)');
  assert.equal(buildQuery(template("peaks-of-term"), { term: "pain" }), 'ans(i, j, k, s) :- mentions(s, "pain"), peak_reported(i, j, k, s)');
  assert.equal(buildQuery(template("terms-of-study"), { study: "9862924" }), "ans(t) :- mentions(9862924, t)");
});

test("a field that is empty or not a number is told", () => {
  assert.throws(() => buildQuery(template("studies-of-term"), { term: "  " }), /Term is empty/);
  assert.throws(() => buildQuery(template("terms-of-study"), { study: "abc" }), /whole number/);
  assert.throws(() => buildQuery(template("studies-by-weight"), { term: "pain", weight: "1e3" }), /must be a number/);
});

test("a question is offered when the engine has its predicates", () => {
  assert.deepEqual(availableTemplates([]), []);
  assert.deepEqual(availableTemplates(["mentions"]).map((t) => t.id), ["studies-of-term", "studies-of-both", "studies-of-one-not-other", "terms-of-study"]);
  assert.equal(availableTemplates(["mentions", "weighted_mention", "peak_reported"]).length, TEMPLATES.length);
});
