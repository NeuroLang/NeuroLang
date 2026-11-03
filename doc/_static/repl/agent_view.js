// The work of the AI assistant as it happens: the model's calls, the tool
// calls of the ReAct agent and what they gave back, as in the terminal. The
// worker sends the events of neurolang_apps.utils.tui.assistant.AGENT_EVENTS
// while it is busy; AgentLog keeps them, renderAgentLog shows them.

// The thought of a step as shown: on one line, not too long.
export function oneLine(text, limit = 200) {
  const line = String(text ?? "").split(/\s+/).filter(Boolean).join(" ");
  return line.length > limit ? `${line.slice(0, limit)}…` : line;
}

// A tool call as `name(a='v')`, without its query, which is shown apart.
export function callText(action, args = {}) {
  const shown = Object.entries(args)
    .filter(([key]) => key !== "query")
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`);
  return `${action}(${shown.join(", ")})`;
}

export class AgentLog {
  constructor() {
    this.reset();
  }

  reset() {
    this.header = null; // what was asked, of whom
    this.steps = []; // the finished steps
    this.calls = []; // the replies of the model since the last finished step
    this.waiting = null; // the request to the model that is not answered
    this.done = null; // the end of the run
    this.tokens = 0;
    this.seconds = 0;
  }

  // Take the *event* of the worker in: ask, request, reply, step or done.
  apply(event, now = Date.now()) {
    if (event.event === "ask") {
      this.reset();
      this.header = event;
    } else if (event.event === "request") {
      this.waiting = { call: event.call, since: now };
    } else if (event.event === "reply") {
      this.waiting = null;
      this.tokens += Object.values(event.usage ?? {}).reduce((a, b) => a + b, 0);
      this.seconds += event.seconds ?? 0;
      this.calls.push(event);
    } else if (event.event === "step") {
      // A step ends the calls of the model it took: a verification takes two.
      this.calls = [];
      this.steps.push(event);
    } else if (event.event === "done") {
      this.waiting = null;
      this.calls = [];
      this.done = event;
    }
  }

  // The number of steps shown, the one under way included.
  get count() {
    return this.steps.length + (this.calls.length ? 1 : 0);
  }

  get running() {
    return Boolean(this.header) && !this.done;
  }

  // The step under way: what the model said and asked, before its tool is done.
  get pending() {
    const last = this.calls.at(-1);
    if (!last) return null;
    const [call] = last.tool_calls ?? [];
    return {
      thought: call ? last.text : "", // the text protocol's reply is the call itself
      raw: call ? "" : last.text,
      action: call?.name ?? "",
      arguments: call?.arguments ?? {},
      error: last.error ?? "",
    };
  }
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// What one step shows: its thought, its call, the query of the call, the observation.
function stepNode(step, { pending = false, renderQuery }) {
  const item = element("li", pending ? "agent-step agent-pending" : "agent-step");
  const thought = oneLine(step.thought || step.raw);
  if (thought) item.append(element("div", "agent-thought", thought));
  if (step.action && step.action !== "answer") {
    item.append(element("div", "agent-call", `→ ${callText(step.action, step.arguments)}`));
    const query = step.arguments?.query;
    if (typeof query === "string" && query.trim()) {
      const block = element("pre", "agent-query");
      block.append(renderQuery(query));
      item.append(block);
    }
  }
  if (step.error) {
    item.append(element("div", "agent-obs agent-bad", step.error));
  } else if (pending) {
    item.append(element("div", "agent-obs", "running…"));
  } else if (step.observation) {
    const lines = step.observation.trim().split("\n");
    if (lines.length === 1) {
      item.append(element("div", "agent-obs", lines[0]));
    } else {
      const more = element("details", "agent-obs");
      more.append(element("summary", "", `${lines[0]} (+${lines.length - 1} lines)`), element("pre", "", lines.slice(1).join("\n")));
      item.append(more);
    }
  }
  return item;
}

// Show *log* in the list *list*; *renderQuery* gives the node of a query's text.
export function renderAgentLog(log, list, { renderQuery = (text) => document.createTextNode(text), now = Date.now() } = {}) {
  const items = [];
  if (log.header) {
    const { question, provider, model, strategy } = log.header;
    items.push(element("li", "agent-head", `Asking ${provider} (${model}, ${strategy}): ${question}`));
  }
  items.push(...log.steps.map((step) => stepNode(step, { renderQuery })));
  if (log.pending) items.push(stepNode(log.pending, { pending: true, renderQuery }));
  if (log.waiting) {
    const wait = element("li", "agent-wait", "asking the model…");
    wait.dataset.since = log.waiting.since;
    wait.dataset.base = "asking the model…";
    items.push(wait);
  }
  if (log.done) {
    const { valid, calls, tokens, seconds, error } = log.done;
    const summary = `${calls} LLM call${calls === 1 ? "" : "s"}, ${tokens.toLocaleString("en-US")} tokens, ${Math.round(seconds)} s`;
    items.push(element("li", valid ? "agent-done" : "agent-done agent-bad", valid ? `✓ query ready: ${summary}` : `✗ ${error || "no valid query"}: ${summary}`));
  }
  list.replaceChildren(...items);
  tickWaiting(list, now);
}

// A model call that lasts shows for how long, so that it does not look stuck.
export function tickWaiting(list, now = Date.now()) {
  for (const wait of list.querySelectorAll(".agent-wait")) {
    const seconds = Math.round((now - Number(wait.dataset.since)) / 1000);
    wait.textContent = seconds >= 3 ? `${wait.dataset.base} (${seconds} s)` : wait.dataset.base;
  }
}
