import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DATALOG_DEFAULTS, setWords, tokens } from "../../../../../../../doc/_static/repl/highlight.js";

// Adjacent fragments of one kind are one: the two splits of a plain text agree.
const merged = (fragments) => {
  const out = [];
  for (const [kind, text] of fragments) {
    if (out.length && out.at(-1)[0] === kind) out.at(-1)[1] += text;
    else out.push([kind, text]);
  }
  return out;
};

const fixturesPath = process.env.HIGHLIGHT_FIXTURES;
const fixtures = fixturesPath ? JSON.parse(readFileSync(fixturesPath, "utf8")) : null;

test("the fragments are those of the terminal's highlighting", { skip: !fixtures && "run by pytest, which writes the fixtures" }, () => {
  setWords(fixtures.words);
  for (const { line, squall, tokens: expected } of fixtures.lines) {
    assert.deepEqual(merged(tokens(line, squall)), merged(expected), line);
  }
});

test("the defaults are the Datalog words of the terminal", { skip: !fixtures && "run by pytest, which writes the fixtures" }, () => {
  assert.deepEqual([...DATALOG_DEFAULTS.datalog_keywords].sort(), fixtures.words.datalog_keywords);
  assert.deepEqual([...DATALOG_DEFAULTS.datalog_operators].sort(), fixtures.words.datalog_operators);
});

test("a call is a predicate, a keyword is not, a comment ends the line", () => {
  setWords({});
  assert.deepEqual(
    [...tokens("ans(x) :- p(x, 'a') % note")].filter(([kind]) => kind),
    [["keyword", "ans"], ["operator", ":-"], ["predicate", "p"], ["string", "'a'"], ["comment", "% note"]],
  );
});

test("the text is kept whole", () => {
  const line = "q(x)  :-  r( x ), 'unclosed";
  assert.equal([...tokens(line)].map(([, text]) => text).join(""), line);
});
