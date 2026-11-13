import assert from "node:assert/strict";
import test from "node:test";
import { layoutQuery } from "../../../../../../../doc/_static/repl/layout.js";

test("a rule that fits its line is left as it is", () => {
  assert.equal(layoutQuery("ans(s) :- mentions(s, 'pain')", 60), "ans(s) :- mentions(s, 'pain')");
});

test("a long rule has its head on the left and an atom of its body a line, after :-", () => {
  const rule = "ans(s) :- cogat_term(c, 'working memory'), cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t)";
  assert.equal(layoutQuery(rule, 50), [
    "ans(s) :- cogat_term(c, 'working memory'),",
    "          cogat_subclass_of(x, c),",
    "          cogat_term(x, t),",
    "          mentions(s, t)",
  ].join("\n"));
});

test("commas within atoms, brackets and strings do not break the line", () => {
  const rule = "ans(s, w) :- weighted_mention(s, 'pain, heat', w), (w > 0.1), PROB[a(s), b(s)] = w";
  assert.deepEqual(layoutQuery(rule, 40).split("\n").map((line) => line.trim()), [
    "ans(s, w) :- weighted_mention(s, 'pain, heat', w),",
    "(w > 0.1),",
    "PROB[a(s), b(s)] = w",
  ]);
});

test("a long probability breaks before its condition, under what it conditions", () => {
  const rule = 'ans(r, p) :- PROB[destrieux_activated(r, s) // (mentions(s, "pain"), selected_study(s))] = p';
  assert.equal(layoutQuery(rule, 70), [
    "ans(r, p) :- PROB[destrieux_activated(r, s)",
    '                  // (mentions(s, "pain"), selected_study(s))] = p',
  ].join("\n"));
});

test("a long head is followed by a short indent, not a column far to the right", () => {
  const rule = "a_very_long_head_name(first_argument, second_argument, third) :- b(first_argument), c(second_argument)";
  assert.equal(layoutQuery(rule, 60).split("\n")[1], "    c(second_argument)");
});

test("each line is laid out alone; commands, comments, facts and lines already broken are kept", () => {
  const text = [
    ".help",
    "% a comment, long enough to be wider than the box, and with commas in it",
    "fact('a long string, with commas, that does not fit the width of the box')",
    "a(x) :- b(x),",
    "        c(x)",
    "ans(x) :- first_long_relation(x), second_long_relation(x), third_long_relation(x)",
  ].join("\n");
  const lines = layoutQuery(text, 40).split("\n");
  assert.deepEqual(lines.slice(0, 5), text.split("\n").slice(0, 5));
  assert.deepEqual(lines.slice(5), ["ans(x) :- first_long_relation(x),", "          second_long_relation(x),", "          third_long_relation(x)"]);
});

test("a conjunction of a probability still too long breaks at its commas, inside its brackets", () => {
  const rule = 'ans(r, p) :- PROB[destrieux_activated(r, s) // (mentions(s, "pain"), selected_study(s))] = p';
  assert.equal(layoutQuery(rule, 50), [
    "ans(r, p) :- PROB[destrieux_activated(r, s)",
    '                  // (mentions(s, "pain"),',
    "                      selected_study(s))] = p",
  ].join("\n"));
  const both = "ans(a, b, p) :- PROB[(activated(a, s), activated(b, s), selected_study(s)) // (mentions(s, 'pain'), selected_study(s))] = p";
  assert.equal(layoutQuery(both, 50), [
    "ans(a, b, p) :- PROB[(activated(a, s),",
    "                      activated(b, s),",
    "                      selected_study(s))",
    "                     // (mentions(s, 'pain'),",
    "                         selected_study(s))] = p",
  ].join("\n"));
});
