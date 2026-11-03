import assert from "node:assert/strict";
import test from "node:test";
import { AgentLog, callText, oneLine } from "../../../../../../../doc/_static/repl/agent_view.js";

const ask = { event: "ask", question: "Who?", provider: "p", model: "m", strategy: "react" };
const reply = (call, text, toolCalls = [], tokens = 15) => ({
  event: "reply", call, text, tool_calls: toolCalls, usage: { input: tokens - 5, output: 5 }, seconds: 1.5,
});
const check = { name: "check_query", arguments: { query: "ans(x) :- p(x)" } };
const step = (action, observation = "ok", thought = "") => ({ event: "step", thought, action, arguments: {}, observation });

test("a request waits, its reply shows the call before its tool is done", () => {
  const log = new AgentLog();
  log.apply(ask);
  log.apply({ event: "request", call: 1 }, 1000);
  assert.deepEqual(log.waiting, { call: 1, since: 1000 });
  assert.equal(log.count, 0);
  log.apply(reply(1, "I check it", [check]));
  assert.equal(log.waiting, null);
  assert.equal(log.count, 1);
  assert.equal(log.pending.action, "check_query");
  assert.equal(log.pending.thought, "I check it");
  assert.deepEqual(log.steps, []);
});

test("the step that ends a call replaces the pending one", () => {
  const log = new AgentLog();
  log.apply(ask);
  log.apply(reply(1, "", [check]));
  log.apply(step("check_query", "The query is valid"));
  assert.equal(log.pending, null);
  assert.equal(log.count, 1);
  assert.equal(log.steps[0].observation, "The query is valid");
});

test("a step takes all the model calls since the last one, as a verification does", () => {
  const log = new AgentLog();
  log.apply(ask);
  log.apply(reply(1, "Reads back as x"));
  log.apply(reply(2, "VERDICT: yes"));
  assert.equal(log.count, 1);
  log.apply(step("verify"));
  assert.equal(log.count, 1);
  assert.equal(log.calls.length, 0);
});

test("the text protocol shows the raw reply until its step is parsed", () => {
  const log = new AgentLog();
  log.apply(ask);
  log.apply(reply(1, "Thought: look\nAction: list_predicates"));
  assert.equal(log.pending.raw, "Thought: look\nAction: list_predicates");
  assert.equal(log.pending.action, "");
});

test("tokens and seconds add up and the end closes the run", () => {
  const log = new AgentLog();
  log.apply(ask);
  assert.ok(log.running);
  log.apply(reply(1, "a", [check], 20));
  log.apply(reply(2, "b", [check], 30));
  assert.equal(log.tokens, 50);
  assert.equal(log.seconds, 3);
  log.apply({ event: "done", valid: true, calls: 2, tokens: 50, seconds: 3 });
  assert.ok(!log.running);
  assert.equal(log.count, 0);
});

test("a new question starts the log again", () => {
  const log = new AgentLog();
  log.apply(ask);
  log.apply(step("a"));
  log.apply({ ...ask, question: "Again?" });
  assert.equal(log.steps.length, 0);
  assert.equal(log.header.question, "Again?");
});

test("a thought is on one line and cut; a call leaves its query out", () => {
  assert.equal(oneLine("a  b\n c"), "a b c");
  assert.equal(oneLine("x".repeat(300)).length, 201);
  assert.equal(callText("search_values", { query: "q", pattern: "pain" }), 'search_values(pattern="pain")');
  assert.equal(callText("list_predicates"), "list_predicates()");
});
