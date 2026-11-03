import assert from "node:assert/strict";
import test from "node:test";
import { TOPICS, topicAt } from "../../../../../../../doc/_static/repl/docs.js";
import { setWords } from "../../../../../../../doc/_static/repl/highlight.js";
import { matches, predicateText } from "../../../../../../../doc/_static/repl/guide.js";
import { decodeShare, encodeShare } from "../../../../../../../doc/_static/repl/share.js";

test("a shared query comes back as it was, accents, newlines and all", () => {
  const code = "% étude\nans(s) :- mentions(s, 'douleur')\n";
  assert.deepEqual(decodeShare(`#${encodeShare(code, "squall")}`), { code, mode: "squall" });
  assert.deepEqual(decodeShare(encodeShare(code)), { code, mode: "datalog" });
});

test("a long query is shared whole", () => {
  const code = "p(x) :- q(x, 'é')\n".repeat(5000);
  assert.equal(decodeShare(encodeShare(code)).code, code);
});

test("what is not a shared query is not one", () => {
  assert.equal(decodeShare(""), null);
  assert.equal(decodeShare("#other=1"), null);
  assert.equal(decodeShare("#q=!!not-base64!!"), null);
  assert.equal(decodeShare("#q=_w"), null); // bytes that are not UTF-8
});

test("the fragment keeps the query out of the address's path and query", () => {
  assert.doesNotMatch(encodeShare("ans(s) :- p(s)"), /[#?/]/);
});

test("the keyword under the cursor tells where it is documented", () => {
  const text = "ans(s) :- PROB[a(s)] = p";
  assert.equal(topicAt(text, text.indexOf("PROB") + 2).word, "PROB");
  assert.equal(topicAt(text, text.indexOf("PROB") + 4).word, "PROB"); // at its end
  assert.match(topicAt(text, 1).url, /the-query-predicate-ans$/);
  assert.match(topicAt(text, text.indexOf(":-")).url, /tutorial_datalog\.html#rules/);
  assert.equal(topicAt(text, text.indexOf("a(s)")), null);
  assert.equal(topicAt("", 0), null);
});

test("it finds the cursor on a later line, and in SQUALL", () => {
  const text = "q(x) :- r(x)\nans(x) :- ~q(x)";
  assert.equal(topicAt(text, text.indexOf("~") + 1).word, "~");
  setWords({ squall_keywords: ["obtain", "every"] });
  const squall = "obtain every Study";
  assert.match(topicAt(squall, 3, true).url, /tutorial_squall\.html$/);
  assert.equal(topicAt(squall, squall.indexOf("Study") + 2, true), null);
});

test("predicates match by the words of their name, signature, kind or description", () => {
  const p = { name: "mentions", signature: "mentions(study, term)", kind: "derived", description: "Studies mentioning a term" };
  assert.ok(matches(p, ""));
  assert.ok(matches(p, "MENTION term"));
  assert.ok(matches(p, "derived studies"));
  assert.ok(!matches(p, "voxel"));
  assert.equal(predicateText(p, false), "mentions(");
  assert.equal(predicateText(p, true), "mentions");
  assert.ok(Object.keys(TOPICS).includes("PROB"));
});
