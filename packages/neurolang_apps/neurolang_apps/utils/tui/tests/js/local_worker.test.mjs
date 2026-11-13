import assert from "node:assert/strict";
import test from "node:test";

// The worker's globals: what it posts to the page, the server's answers by
// action, and the stream of the server's events.
const posted = [];
const asked = [];
const answers = {
  stop: () => ({ stopped: false }),
  start: ({ engine }) => ({ banner: `<b>${engine}</b>`, providers: [] }),
  submit: ({ text }) =>
    text === "boom"
      ? { status: 500, body: { error: "ValueError: it broke" } }
      : text === "gone"
        ? { status: 409, body: { error: "This session ended: restart it", ended: true } }
        : { html: text, failed: false },
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
const urls = [];
globalThis.fetch = async (url, { body }) => {
  urls.push(url);
  const action = url.replace("/api/", "").split("?")[0];
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

test("a session the server ended is the page's to bring back, with what was run", async () => {
  assert.deepEqual(await send({ id: 5, text: "gone" }), [{ type: "ended", id: 5, text: "gone", message: "This session ended: restart it" }]);
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

test("the session of the page's tab goes with each request", async () => {
  urls.length = 0;
  await send({ type: "start", engine: "neurosynth", session: "tab 1" });
  await send({ type: "submit", id: 9, text: "ans(x) :- p(x)" });
  assert.ok(urls.length >= 3);
  assert.ok(urls.every((url) => url.endsWith("?session=tab%201")), urls.join());
  assert.equal(events.url, "/api/events?session=tab%201");
});

test("an expired login has the page log in again", async () => {
  const submit = answers.submit;
  answers.submit = () => ({ status: 401, body: { error: "log in first: /login" } });
  try {
    const messages = await send({ id: 10, text: "ans(x) :- p(x)" });
    assert.deepEqual(messages[0], { type: "login" });
    assert.equal(messages[1].failed, true);
  } finally {
    answers.submit = submit;
  }
});
