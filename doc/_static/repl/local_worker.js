// The worker of the page when NeuroLang runs in the Python of this machine
// (neurolang-repl-serve --local, whose bundle.json says "backend": "local").
// It takes the messages of the page as worker.js does, and has the server
// answer them (neurolang_apps.utils.tui.local_backend): the engine is built
// there, from its data directory, rather than in this tab with Pyodide. What
// the server tells while it works (the steps of a build, of the assistant)
// comes as events.

const API = "/api/";
// The session of the page's tab, on a server that keeps one per tab.
let session = "";
const url = (action) => API + action + (session ? `?session=${encodeURIComponent(session)}` : "");

// The answer of the server to *action* with the JSON *body*: its JSON, or
// the response itself when *raw*. An error of the server is thrown with its
// message.
async function ask(action, body = {}, { raw = false } = {}) {
  const response = await fetch(url(action), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.status === 401) postMessage({ type: "login" }); // the login expired
  if (!response.ok) {
    const { error, ended } = await response.json().catch(() => ({ error: response.statusText }));
    // *ended*: the server ended this tab's session (idle, room for another, a query too long).
    throw Object.assign(new Error(error || `the server answered ${response.status}`), { ended: Boolean(ended) });
  }
  return raw ? response : response.json();
}

let events = null;
let started = false;

// The messages the server sends while it works go to the page as they come.
function listen() {
  events?.close();
  events = new EventSource(url("events"));
  events.onmessage = ({ data }) => postMessage(JSON.parse(data));
}

// The error of Python, its last line without the name of its class.
const lastLine = (error) => String(error.message ?? error).trim().split("\n").at(-1).replace(/^[\w.]+Error: /, "");

const handlers = {
  async start({ engine, session: tab }) {
    session = tab ?? "";
    listen();
    // A query of the page this worker replaces may still run: a restart,
    // or a stop, interrupts it first.
    await ask("stop");
    postMessage({ type: "status", message: `Building the engine ${engine} in the Python of this machine…` });
    const ready = await ask("start", { engine });
    started = true;
    postMessage({ type: "ready", ...ready });
  },
  async llm({ provider, key, model, baseUrl }) {
    try {
      await ask("llm", { provider, key, model, baseUrl });
      postMessage({ type: "llm", ok: true });
    } catch (error) {
      postMessage({ type: "llm", ok: false, message: lastLine(error) });
    }
  },
  async rate({ verdict, comment }) {
    try {
      postMessage({ type: "rated", ...(await ask("rate", { verdict, comment: comment ?? "" })) });
    } catch (error) {
      postMessage({ type: "rated", ok: false, message: lastLine(error) });
    }
  },
  async download({ block, kind, value }) {
    try {
      const response = await ask("download", { block, kind, value: value ?? "" }, { raw: true });
      const bytes = new Uint8Array(await response.arrayBuffer());
      const name = decodeURIComponent(response.headers.get("X-File-Name") ?? "result");
      const mime = response.headers.get("Content-Type") ?? "application/octet-stream";
      postMessage({ type: "file", name, mime, bytes }, [bytes.buffer]);
    } catch (error) {
      postMessage({ type: "file", error: lastLine(error) });
    }
  },
  async submit({ id, text }) {
    try {
      postMessage({ type: "reply", id, ...(await ask("submit", { text })) });
    } catch (error) {
      // The page brings the session back, then runs *text* again.
      if (error.ended) return postMessage({ type: "ended", id, text, message: lastLine(error) });
      const message = String(error.message ?? error).replace(/&/g, "&amp;").replace(/</g, "&lt;");
      postMessage({ type: "reply", id, failed: true, viewers: [], tables: [], html: `<pre>${message}</pre>` });
    }
  },
};

onmessage = async ({ data }) => {
  if (data.type === "start") {
    try {
      await handlers.start(data);
    } catch (error) {
      postMessage({ type: "error", message: String(error.message ?? error) });
    }
    return;
  }
  if (!started) return;
  await (handlers[data.type] ?? handlers.submit)(data);
};
