import assert from "node:assert/strict";
import test from "node:test";

// The worker's globals: what it posts to the page, the server's answers by
// action, and the stream of the server's events.
const posted = [];
const asked = [];
const answers = {
  stop: () => ({ stopped: false }),
  start: ({ engine }) => ({ banner: `<b>${engine}</b>`, providers: [] }),
  submit: ({ text }) => (text === "boom" ? { status: 500, body: { error: "ValueError: it broke" } } : { html: text, failed: false }),
  llm: ({ key }) => (key === "bad" ? { status: 500, body: { error: "ValueError: Unknown provider 'x'." } } : {}),
  rate: ({ verdict }) => ({ ok: true, message: verdict }),
  download: () => ({ bytes: new Uint8Array([97, 44, 98]), headers: { "X-File-Name": "r%C3%A9sultat.csv", "Content-Type": "text/csv" } }),
};
let events = null;

globalThis.postMessage = (message) => posted.push(message);
globalThis.onmessage = null;
globalThis.EventSource = class {
  constructor(url) {
    this.url = url;
    events = this;
  }
  close() {}
};
globalThis.fetch = async (url, { body }) => {
  const action = url.replace("/api/", "");
  const data = JSON.parse(body);
  asked.push([action, data]);
  const answer = answers[action](data);
  if (answer.bytes) {
    return new Response(answer.bytes, { status: 200, headers: answer.headers });
  }
  const status = answer.status ?? 200;
  return new Response(JSON.stringify(answer.body ?? answer), { status, headers: { "Content-Type": "application/json" } });
};

await import("../../../../../../../doc/_static/repl/local_worker.js");

const send = async (data) => {
  posted.length = 0;
  await globalThis.onmessage({ data });
  return posted.slice();
};

test("nothing is answered before the start", async () => {
  assert.deepEqual(await send({ text: "ans(x) :- r(x)" }), []);
});

test("the start stops a query at work, then tells the page it is ready", async () => {
  const messages = await send({ type: "start", engine: "neurosynth" });
  assert.deepEqual(asked.slice(0, 2).map(([action]) => action), ["stop", "start"]);
  assert.equal(messages[0].type, "status");
  assert.deepEqual(messages.at(-1), { type: "ready", banner: "<b>neurosynth</b>", providers: [] });
  assert.equal(events.url, "/api/events");
});

test("the events of the server go to the page", () => {
  posted.length = 0;
  events.onmessage({ data: JSON.stringify({ type: "agent", event: "ask" }) });
  assert.deepEqual(posted, [{ type: "agent", event: "ask" }]);
});

test("a query is answered as worker.js answers it", async () => {
  assert.deepEqual(await send({ id: 3, text: "ans(x) :- r(x)" }), [{ type: "reply", id: 3, html: "ans(x) :- r(x)", failed: false }]);
  const [failed] = await send({ id: 4, text: "boom" });
  assert.equal(failed.failed, true);
  assert.match(failed.html, /ValueError: it broke/);
});

test("the key of the assistant, and what is wrong with it", async () => {
  assert.deepEqual(await send({ type: "llm", provider: "anthropic", key: "k" }), [{ type: "llm", ok: true }]);
  assert.deepEqual(await send({ type: "llm", provider: "x", key: "bad" }), [{ type: "llm", ok: false, message: "Unknown provider 'x'." }]);
});

test("a file of the server is given to the page", async () => {
  const [file] = await send({ type: "download", block: 0, kind: "csv" });
  assert.equal(file.name, "résultat.csv");
  assert.equal(file.mime, "text/csv");
  assert.deepEqual([...file.bytes], [97, 44, 98]);
});

test("a rating is told back", async () => {
  assert.deepEqual(await send({ type: "rate", verdict: "good" }), [{ type: "rated", ok: true, message: "good" }]);
});
