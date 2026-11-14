// The page of the in-browser REPL: a box for the AI assistant, a box for
// the code, and the result as a table or in the brain viewer. It shows what
// worker.js answers.
import { SETTINGS } from "./settings.js";
import { History } from "./history.js";
import { AskedHistory } from "./asked.js";
import { LlmSettings } from "./llm_settings.js";
import { imageButtons, renderTable } from "./table_view.js";
import { AgentLog, noAnswer, renderAgentLog, tickWaiting } from "./agent_view.js";
import { DOCS_URL, PAGES, topicAt } from "./docs.js";
import { Editor } from "./editor.js";
import { layoutQuery } from "./layout.js";
import { renderCommands, renderExamples, renderPredicates, startingQuery, suggestedQuestion } from "./guide.js";
import { highlightNodes, loadWords } from "./highlight.js";
import { decodeShare, encodeShare } from "./share.js";
import { TEMPLATES, availableTemplates, buildQuery } from "./builder.js";
import { followUps } from "./follow_up.js";
import { logSummary } from "./log_view.js";
import { explainStartError } from "./start_error.js";
import { Session, parseSession, replayTexts } from "./session.js";
import { PmcLookup } from "./study_links.js";
import { frameHeight, themeMessage, themed } from "./frames.js";
import { tabAfter } from "./tabs.js";
import { bundleEngines, engineChoice, withEngine } from "./engines.js";
import { tabSession } from "./tab_session.js";

const $ = (id) => document.getElementById(id);
const [statusBox, out, input, question, history_] = ["status", "out", "input", "question", "history"].map($);
const llmForm = $("llm");
const results = $("results");
const TABS = ["table", "brain", "graph"]; // the tabs of the results, in order
const DEFAULT_REPO = "NeuroLang/NeuroLang";
const busyControls = [input, question, $("run"), $("ask-run"), $("view"), $("explain"), $("session-open")];
// The code box colours its text as the language of the mode select.
const editor = new Editor(input, $("mirror"), () => $("mode").value === "squall");
// A query the page puts in the box, laid out for its width: a long rule a line
// per atom of its body (layout.js). What the visitor typed is kept as it is.
const laidOut = (text) => ($("mode").value === "squall" ? text : layoutQuery(text, editor.columns()));
const putQuery = (text) => { editor.value = laidOut(text); };
function formatQuery() {
  const text = laidOut(editor.value);
  if (text !== editor.value) editor.value = text;
  input.focus();
}
loadWords().then(() => editor.refresh()); // the words of the grammars, from highlight.json

const store = (kind) => {
  try {
    return window[kind];
  } catch {
    return null; // refused, e.g. in a private window
  }
};
const history = new History(store("localStorage"));
// The questions asked to the assistant, each with the query it gave.
const asked = new AskedHistory(store("localStorage"));
let asking = null; // the question being answered, and its language
const llmSettings = new LlmSettings(store("localStorage"), store("sessionStorage"));
let worker = null;
let providers = []; // those of the AI assistant, from the worker
// What the page downloads (bundle.json): the engines it offers, chosen with
// the select of the header or ?engine=NAME.
let bundle = {};
const engineSelect = $("engine");

// The work of the AI assistant, shown as it comes from the worker.
const agentLog = new AgentLog();
const live = $("live");
const liveLog = $("live-log");
// What the query the assistant wrote does, above the box it is in: the model's
// explanation, and the query read back in English, with the check's doubts.
const summary = $("summary");
function hideSummary() {
  summary.hidden = true;
}
function showSummary({ valid, explanation, back_translation: reads, verified, verification }, question = "") {
  if (!valid || !(explanation || reads || question)) return hideSummary();
  $("summary-question").textContent = question ? `Asked: ${question}` : "";
  $("summary-text").textContent = explanation ?? "";
  $("summary-reads").textContent = reads ? `Reads back as: ${reads}` : "";
  $("summary-warning").textContent = verified === false ? `The check found that the query may not answer the question: ${verification}` : "";
  summary.hidden = false;
}

// Rating the last query, or the assistant's answer: "Yes" sends it at once,
// "No" asks what was wrong first. Shown only where answers can be rated
// (feedback.json), after a query or an answer, not after other commands.
let canRate = false;
const rate = $("rate");
const rateButtons = ["rate-good", "rate-bad", "rate-send"].map($);
function hideRating() {
  rate.hidden = true;
  $("rate-note").textContent = "";
  $("rate-comment").value = "";
  $("rate-comment").hidden = $("rate-send").hidden = true;
  rateButtons.forEach((button) => (button.disabled = false));
}
function showRating() {
  rate.hidden = !canRate;
}
function sendRating(verdict) {
  rateButtons.forEach((button) => (button.disabled = true));
  $("rate-note").textContent = "Sending…";
  worker.postMessage({ type: "rate", verdict, comment: $("rate-comment").value.trim() });
}
$("rate-good").addEventListener("click", () => sendRating("good"));
$("rate-bad").addEventListener("click", () => {
  $("rate-comment").hidden = $("rate-send").hidden = false;
  $("rate-comment").focus();
});
$("rate-send").addEventListener("click", () => sendRating("bad"));
$("rate-comment").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  sendRating("bad");
});

function showAgent() {
  live.hidden = false;
  live.classList.toggle("running", agentLog.running);
  $("live-summary").textContent = agentLog.count ? `Assistant steps (${agentLog.count})` : "Assistant steps";
  const squall = agentLog.header?.language === "squall";
  const renderQuery = (text) => {
    const fragment = document.createDocumentFragment();
    fragment.append(...highlightNodes(text, squall));
    return fragment;
  };
  renderAgentLog(agentLog, liveLog, { renderQuery });
  if (agentLog.running) liveLog.scrollTop = liveLog.scrollHeight;
}

// A long step shows how long it has lasted, so that a slow one does not look stuck.
const step = { message: "", since: 0 };
const setStatus = (message, state = "") => {
  Object.assign(step, { message, since: Date.now() });
  Object.assign(statusBox, { textContent: message, className: state });
};
setInterval(() => {
  const seconds = Math.round((Date.now() - step.since) / 1000);
  if (statusBox.className === "busy" && seconds >= 2) statusBox.textContent = `${step.message} (${seconds} s)`;
  tickWaiting(liveLog);
}, 1000);
const html = (markup) => Object.assign(document.createElement("div"), { innerHTML: markup }); // rich escapes its text
const setBusy = (busy) => {
  document.body.toggleAttribute("data-busy", busy);
  busyControls.forEach((control) => (control.disabled = busy));
  if (!busy) {
    $("stop").hidden = true;
    input.focus();
  }
};

// The tables of the last result, for the downloads of the brain tab, which
// shows a result that .view may have taken from an earlier reply.
let lastTables = [];

// The file of a result: the worker makes it, the page offers it to be saved.
function requestDownload(block, kind, value = "") {
  setBusy(true);
  setStatus("Preparing the file…", "busy");
  worker.postMessage({ type: "download", block, kind, value });
}

function saveFile({ name, mime, bytes }) {
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([bytes], { type: mime })), download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// The documentation: its pages in the menu, and what the keyword under the cursor is.
const link = (href, text) => Object.assign(document.createElement("a"), { href, target: "_blank", rel: "noopener", textContent: text });
$("docs-links").replaceChildren(...PAGES.map(([label, page]) => link(DOCS_URL + page, label)));

function showHint() {
  const topic = topicAt(input.value, input.selectionStart, $("mode").value === "squall");
  $("hint").replaceChildren(
    ...(topic ? [Object.assign(document.createElement("code"), { textContent: topic.word }), ` ${topic.text} · `, link(topic.url, "documentation ↗")] : []),
  );
}
["keyup", "click", "focus"].forEach((type) => input.addEventListener(type, showHint));

// The questions and examples of the engine and its predicates, which put their text in a box.
let guide = { examples: {}, questions: [], predicates: [], commands: [] };
function showGuide() {
  if (!$("guide").open) return;
  const squall = $("mode").value === "squall";
  $("guide-questions-box").hidden = !guide.questions.length;
  renderExamples($("guide-questions"), guide.questions, (text) => {
    question.value = text;
    question.focus();
  }, "Put this question in the box of the assistant");
  const list = $("guide-examples");
  const width = Math.max(30, Math.floor((list.clientWidth - 24) / 7.5)); // 12.5px monospace
  renderExamples(list, guide.examples[squall ? "squall" : "datalog"], (query) => {
    putQuery(query);
    hideSummary();
    input.focus();
  }, undefined, (query) => (squall || !list.clientWidth ? query : layoutQuery(query, width)));
  renderPredicates($("guide-relations"), guide.predicates, $("guide-filter").value, (text) => editor.insert(text), squall, SETTINGS.guide.relations_shown, (query) => {
    putQuery(query);
    hideSummary();
    input.focus();
  });
  renderCommands($("guide-commands"), guide.commands, $("guide-filter").value, (name) => editor.insert(name), (topic) => run(`.help ${topic}`));
  showBuilder();
}
// The form of the common questions: those the engine has the predicates of, in Datalog.
const builder = $("builder");
function showBuilder() {
  const templates = availableTemplates(guide.predicates.map((predicate) => predicate.name));
  builder.hidden = $("mode").value === "squall" || !templates.length;
  if (builder.hidden) return;
  const select = $("builder-template");
  const key = templates.map((template) => template.id).join();
  if (select.dataset.key === key) return; // the fields keep what was typed in them
  select.dataset.key = key;
  select.replaceChildren(...templates.map((template) => new Option(template.label, template.id)));
  showFields();
}
function showFields() {
  const template = TEMPLATES.find((t) => t.id === $("builder-template").value);
  $("builder-error").textContent = "";
  $("builder-fields").replaceChildren(...template.fields.map((field) => Object.assign(document.createElement("input"), {
    name: field.name, placeholder: field.placeholder, value: field.value ?? "", spellcheck: false, autocomplete: "off",
    ariaLabel: field.label, title: field.label,
  })));
}
$("builder-template").addEventListener("change", showFields);
builder.addEventListener("submit", (event) => {
  event.preventDefault();
  const template = TEMPLATES.find((t) => t.id === $("builder-template").value);
  try {
    const query = buildQuery(template, Object.fromEntries(new FormData(builder)));
    putQuery(query);
    hideSummary();
    input.focus();
    $("builder-error").textContent = "";
  } catch (error) {
    $("builder-error").textContent = error.message;
  }
});
$("guide").addEventListener("toggle", showGuide);
$("guide-queries").addEventListener("toggle", showGuide); // the examples are laid out for the width of their list
$("guide-filter").addEventListener("input", showGuide);

// A query shared as a link is put in the box, not run.
function restoreShared() {
  const shared = decodeShare(location.hash);
  if (!shared) return;
  if ($("mode").value !== shared.mode) {
    $("mode").value = shared.mode;
    run(`.mode ${shared.mode}`);
  }
  editor.value = shared.code;
}

// An answer of the assistant with no query, *missing* (noAnswer), to the
// *asked* question: said where the question is, why, and what to try; null
// hides it.
let missingQuery = "";
function showNoAnswer(missing, asked = "") {
  $("no-answer").hidden = !missing;
  if (!missing) return;
  $("no-answer-question").textContent = asked;
  $("no-answer-reason").textContent = missing.reason;
  Object.assign($("no-answer-said"), { textContent: missing.said ? `In its words: ${missing.said}` : "", hidden: !missing.said });
  missingQuery = missing.query;
  $("no-answer-query").hidden = !missing.query;
  $("no-answer-box").textContent = input.value.trim()
    ? "The query box still holds the query that was there before; the result shown is not an answer to this question."
    : "";
  $("no-answer-edit").dataset.question = asked;
}
$("no-answer-edit").addEventListener("click", (event) => {
  question.value = event.currentTarget.dataset.question || question.value;
  question.focus();
  question.select();
});
$("no-answer-query").addEventListener("click", () => {
  putQuery(missingQuery);
  hideSummary();
  input.focus();
});
$("no-answer-steps").addEventListener("click", () => {
  live.hidden = false;
  live.open = true;
  live.scrollIntoView({ behavior: "smooth", block: "nearest" });
});
$("no-answer-examples").addEventListener("click", () => {
  $("guide").open = true;
  showGuide();
  $("guide").scrollIntoView({ behavior: "smooth", block: "start" });
});

// The past questions, the newest first, each with the query it gave: to put
// that query back in the box, in its language, or to ask the question again.
function showAsked() {
  const entries = asked.newest();
  $("asked").hidden = !entries.length;
  $("asked-summary").textContent = `Past questions (${entries.length})`;
  const when = (at) => new Date(at).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
  $("asked-list").replaceChildren(...entries.slice(0, SETTINGS.history.questions_shown).map((entry) => {
    const item = document.createElement("li");
    const text = Object.assign(document.createElement("p"), { className: "asked-question", textContent: entry.question });
    const shown = entry.valid && entry.query
      ? Object.assign(document.createElement("pre"), { className: "asked-query" })
      : Object.assign(document.createElement("p"), { className: "asked-none", textContent: "No valid query came of it." });
    const squall = entry.language === "squall";
    if (shown.tagName === "PRE") shown.append(...highlightNodes(squall ? entry.query : layoutQuery(entry.query, SETTINGS.history.question_width), squall));
    const actions = Object.assign(document.createElement("div"), { className: "asked-actions" });
    if (entry.valid && entry.query) {
      actions.append(Object.assign(document.createElement("button"), {
        type: "button", textContent: "Use this query", title: "Put this query in the box, with what the assistant said of it",
        onclick: () => useAsked(entry),
      }));
    }
    actions.append(
      Object.assign(document.createElement("button"), {
        type: "button", className: "quiet", textContent: "Ask again", title: "Put this question in the box of the assistant",
        onclick: () => { question.value = entry.question; question.focus(); },
      }),
      Object.assign(document.createElement("time"), { dateTime: new Date(entry.at).toISOString(), textContent: when(entry.at) }),
    );
    item.append(text, shown, actions);
    return item;
  }));
}

// The query of a past question back in the box, in its language, with its summary.
function useAsked(entry) {
  if ($("mode").value !== entry.language) {
    $("mode").value = entry.language;
    run(`.mode ${entry.language}`);
  }
  putQuery(entry.query);
  showSummary({ valid: true, explanation: entry.explanation }, entry.question);
  input.focus();
}

function showHistory() {
  history_.replaceChildren(new Option("History", ""), ...[...history.entries].reverse().map((e) => new Option(e.replace(/\s+/g, " ").slice(0, 90), e)));
}

// Show *what* in the results, in the tab *tab*.
function show(tab, ...nodes) {
  $(tab).replaceChildren(...nodes);
  $(`tab-${tab}`).hidden = false;
  results.classList.add("has");
  selectTab(tab);
}

// The tab shown of the results, as the tab list tells it.
function selectTab(tab) {
  results.dataset.tab = tab;
  for (const name of TABS) {
    $(`tab-${name}`).setAttribute("aria-selected", name === tab);
    $(`tab-${name}`).tabIndex = name === tab ? 0 : -1;
  }
}

// A failed start, in plain words, with the technical message folded.
function fail(message) {
  const { summary, detail } = explainStartError(message);
  setStatus("The REPL could not start.", "error");
  $("fatal-text").textContent = summary;
  $("fatal-detail").textContent = detail;
  $("fatal").hidden = false;
  $("boot").hidden = true;
}

// The AI assistant: the form, and the key given to the worker.
const choices = () => {
  const data = Object.fromEntries(new FormData(llmForm));
  return { provider: data.provider, model: data.model.trim(), baseUrl: (data.baseUrl ?? "").trim(), key: data.key.trim(), remember: llmForm.elements.remember.checked };
};

function showProvider() {
  const preset = providers.find((p) => p.id === llmForm.elements.provider.value);
  llmForm.elements.model.placeholder = preset?.model || "model";
  llmForm.elements.baseUrl.hidden = !preset || Boolean(preset.base_url);
  llmForm.elements.baseUrl.required = !llmForm.elements.baseUrl.hidden;
}

// The assistant can answer: a key is in use, or the REPL has one.
let assistantReady = false;
// The example question put in the box, while it is left as it is.
let suggested = "";

// An example question of the engine, at random, in the box of the
// assistant, when the box is empty and the assistant can answer.
function suggestQuestion() {
  const pick = suggestedQuestion(guide.questions, question.value, assistantReady);
  if (pick) suggested = question.value = pick;
}

// Whether the page gives a key of its own, rather than the server's, and
// whether the user may (a server of users may forbid it).
let pageKey = false;
let ownKeyAllowed = true;

// What the server's assistant is: *ready* to answer (the key of the user's
// account, their group or the server), and whether the user may give their
// own key; a key the page gives stays in use.
function serverAssistant(ready, ownKey) {
  ownKeyAllowed = ownKey;
  $("key").hidden = $("llm-details").hidden = !ownKey;
  if (!ownKey) {
    pageKey = $("llm-details").open = false;
    $("llm-summary").textContent = "bring your own API key";
    $("key").textContent = "API key";
  }
  if (pageKey) return;
  assistantReady = Boolean(ready);
  Object.assign($("ask-run"), { title: ready ? "" : "Add an API key to use the assistant" }).classList.toggle("ghost", !ready);
  $("explain").hidden = !ready;
  $("ask-note").hidden = false;
  if (ready) {
    $("ask-note").textContent = `Asking uses the AI assistant ${ownKey ? "set for you on this server, unless you give your own API key" : "your admin set for you"}.`;
    suggestQuestion();
  } else if (bundle.users) {
    $("ask-note").textContent = ownKey
      ? "Asking needs an API key from an AI provider: click API key, or save one on your Account page."
      : "Asking needs an AI assistant, which your admin has not set for you.";
  }
}

function useKey() {
  pageKey = true;
  assistantReady = true;
  suggestQuestion();
  $("llm-error").hidden = true;
  const chosen = choices();
  worker.postMessage({ type: "llm", ...chosen });
  Object.assign($("ask-run"), { title: "" }).classList.remove("ghost");
  $("llm-summary").textContent = `${providers.find((p) => p.id === chosen.provider)?.label ?? chosen.provider}, key in use`;
  $("key").textContent = "API key ✓";
  $("ask-note").hidden = true;
  $("explain").hidden = false;
}

function setProviders(list) {
  providers = list;
  const saved = llmSettings.load();
  const { elements } = llmForm;
  elements.provider.replaceChildren(...list.map((p) => new Option(p.label, p.id)));
  if (list.some((p) => p.id === saved.provider)) elements.provider.value = saved.provider;
  Object.assign(elements.model, { value: saved.model });
  Object.assign(elements.baseUrl, { value: saved.baseUrl });
  Object.assign(elements.key, { value: saved.key });
  elements.remember.checked = saved.remember;
  elements.provider.disabled = !list.length;
  llmForm.querySelector("button").disabled = !list.length;
  showProvider();
  $("llm-summary").textContent = list.length ? "bring your own API key" : "not in this version of NeuroLang";
  if (list.length && saved.key && ownKeyAllowed) useKey();
}

// Offer the engines of the bundle: a select, when there are several.
async function loadEngines() {
  try {
    bundle = await (await fetch("bundle.json", { cache: "no-store" })).json();
  } catch {
    bundle = {}; // the worker tells that the bundle is missing
  }
  const engines = Object.entries(bundleEngines(bundle));
  engineSelect.replaceChildren(
    ...engines.map(([name, entry]) => Object.assign(document.createElement("option"), { value: name, textContent: entry.label })),
  );
  engineSelect.hidden = engines.length < 2;
}

// The id of this tab's session, on a server that keeps one per tab, once found.
let tabId = null;

// The version to run, from the page's address: ?repo=OWNER/NAME&ref=BRANCH,
// and the engine: ?engine=NAME.
function start({ restore = null } = {}) {
  if (!restore) session.replace([]); // a new engine has none of the rules
  rerun = null; // a stop, a restart, end what ran again
  worker?.terminate();
  const params = new URLSearchParams(location.search);
  const [repo, ref] = [params.get("repo") || DEFAULT_REPO, params.get("ref") || ""];
  const { name: engine, full } = engineChoice(params, bundle);
  engineSelect.value = engine;
  Object.assign(version.elements.repo, { value: repo });
  Object.assign(version.elements.ref, { value: ref });
  $("version-summary").textContent =
    bundle.backend === "local" ? "the Python of this machine" : ref ? `${repo} at ${ref}` : "the build of this page";
  out.replaceChildren();
  out.classList.remove("quiet");
  results.classList.remove("has");
  $("table").replaceChildren();
  $("brain").replaceChildren();
  $("tab-brain").hidden = true;
  selectTab("table");
  $("fatal").hidden = true;
  $("boot").hidden = false;
  agentLog.reset();
  live.hidden = true;
  setStatus("Starting…", "busy");
  setBusy(true);
  input.disabled = false; // the query can be written while the engine loads
  // A page served with NeuroLang in the Python of this machine
  // (neurolang-repl-serve --local) has it answer, not Pyodide: its
  // assistant is that of the machine's configuration, unless a key is given.
  const local = bundle.backend === "local";
  // A server of users (--users): who is logged in, their account, logging out.
  $("account").hidden = $("logout").hidden = !bundle.users;
  if (bundle.users) {
    fetch("/account/api/me").then((response) => response.ok && response.json()).then((me) => {
      if (me?.name) $("account").textContent = me.orcid_name || me.name; // the name ORCID gives
      if (me?.guest) $("account").removeAttribute("href"); // a guest has no account
    }).catch(() => {});
  }
  worker = new Worker(local ? "local_worker.js" : "worker.js", { type: "module" });
  if (local) {
    $("ask-note").textContent =
      "Asking uses the AI assistant configured on this machine, as neurolang-query does, or the API key you give.";
  }
  worker.onerror = (event) => fail(event.message || "the browser ended its worker, perhaps out of memory");
  worker.onmessage = ({ data }) => {
    if (data.type === "reply" && replaying) replayStep(data);
    else workerMessages[data.type]?.(data, restore);
  };
  const message = { type: "start", repo, ref, full, engine };
  if (!local) return worker.postMessage(message);
  // The server keeps a session per tab: this tab's, unless restarted since.
  const current = worker;
  tabId ??= tabSession(store("sessionStorage"));
  tabId.then((session) => worker === current && current.postMessage({ ...message, session }));
}

// What the page does with each kind of message of the worker; *restore* is what a stop gave to give back.
const workerMessages = {
  // The assistant of the user changed on the server (their account page, an admin).
  assistant: (data) => serverAssistant(data.ready, data.ownKey !== false),
  // The login expired: the login page, then back here.
  login: () => location.assign(`/login?next=${encodeURIComponent(location.pathname + location.search)}`),
  status: (data) => setStatus(data.message, "busy"),
  error: (data) => fail(data.message),
  ready: (data, restore) => {
    setStatus("Ready.");
    $("boot").hidden = true;
    $("banner").replaceChildren(html(data.banner));
    ownKeyAllowed = data.ownKey !== false; // before a saved key is sent
    setProviders(data.providers ?? []);
    if ("assistant" in data) serverAssistant(data.assistant, ownKeyAllowed);
    canRate = Boolean(data.canRate);
    if (data.ratingNote) $("rate-disclosure").textContent = data.ratingNote;
    // This server logs what the user runs (--activity-db).
    $("log-note").textContent = data.logNote ?? "";
    $("log-note").hidden = !data.logNote;
    guide = { examples: data.examples ?? {}, questions: data.questions ?? [], predicates: data.predicates ?? [], commands: data.commands ?? [] };
    suggestQuestion();
    if (!input.value && !decodeShare(location.hash)) guideForStart = $("guide").open = true; // examples to start from
    $("guide-queries").open = !guide.questions.length; // the queries, when there is no question to start from
    showGuide();
    if (restore) return replay(restore);
    setBusy(false);
    restoreShared();
    // A first visit has a query to run in the box, not an empty one.
    const first = decodeShare(location.hash) ? "" : startingQuery(guide.examples, $("mode").value, { box: input.value, visited: history.entries.length > 0 });
    if (first) putQuery(first);
  },
  file: (data) => {
    setBusy(false);
    if (data.error) return setStatus(`The file could not be made: ${data.error}`, "error");
    saveFile(data);
    setStatus("Ready.");
  },
  agent: (data) => {
    agentLog.apply(data);
    showAgent();
    if (data.event === "ask") {
      showNoAnswer(null);
      asking = { question: data.question ?? "", language: data.language ?? $("mode").value };
      hideSummary();
      hideRating();
    }
    if (data.event === "done") {
      showNoAnswer(noAnswer(data), asking?.question);
      showSummary(data, asking?.question);
      showRating();
      if (asking) asked.add({ ...asking, query: data.query ?? "", valid: data.valid, explanation: data.explanation });
      asking = null;
      showAsked();
    }
  },
  rated: (data) => {
    $("rate-note").textContent = data.message;
    $("rate-note").focus();
    if (!data.ok) rateButtons.forEach((button) => (button.disabled = false));
    else $("rate-comment").hidden = $("rate-send").hidden = true;
  },
  llm: (data) => {
    if (!data.ok) Object.assign($("llm-error"), { textContent: data.message, hidden: false });
  },
  // The server ended this tab's session (unused for long, room for another, a query too long): its rules
  // come back, as after a stop, then what was run runs again. Ended again while they come back, or at once
  // after, it is told, not looped.
  ended: (data) => {
    if (replaying || rerun !== null) {
      replaying = rerun = null;
      return workerMessages.reply({ failed: true, viewers: [], tables: [], html: `<pre>${data.message.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre>` });
    }
    const entries = session.entries.map((entry) => ({ ...entry }));
    setStatus("The server had ended this session: its rules come back, then this runs again…", "busy");
    start({ restore: { entries, box: editor.value, mode: $("mode").value, then: data.text } });
  },
  reply: (data) => {
    rerun = null;
    setStatus(data.failed ? "Ready (the last statement failed)." : "Ready.", data.failed ? "warn" : "");
    if (!data.failed) session.add(lastRun, $("mode").value);
    showLog(html(data.html), !data.failed && Boolean(data.tables?.length));
    if (data.ratable) showRating();
    applyReply(data);
    // A failure without a table hides the last result, not to be taken for its own.
    if (data.failed && !data.tables?.length) results.classList.remove("has");
    if (data.tables?.length) showTables(data);
    showGraph(data);
    if (data.viewers?.length) showViewer(data.viewers.at(-1));
    else if (data.tables?.length) markViewerStale();
    else if (/^\.view\b/.test(lastRun.trim())) dropViewer(); // the view failed: its output says why
    setBusy(false);
    // On a narrow screen the result is under the query: it is brought into view.
    if (stacked.matches && !$("no-answer").hidden) $("no-answer").scrollIntoView({ behavior: "smooth", block: "start" });
    else if (stacked.matches && (data.tables?.length || data.graph || data.failed)) $("result-title").scrollIntoView({ behavior: "smooth", block: "start" });
  },
};

// What a reply changes in the page besides its output: the predicates, the language, and the box.
function applyReply(data) {
  if (data.predicates) guide.predicates = data.predicates;
  if (data.mode) $("mode").value = data.mode; // the assistant may have switched it
  showGuide();
  if (data.prompt) putQuery(data.prompt); // what the assistant wrote
  else editor.refresh();
}

// The tables of a reply, each with the queries that go on from it when it is the only one.
function showTables({ tables, prompt }) {
  lastTables = tables;
  // A result is a relation to go on from only if the query that made it is known.
  const query = prompt || (lastRun.trim().startsWith(".") ? "" : lastRun);
  const squall = $("mode").value === "squall";
  show("table", ...tables.map((table) => renderTable(table, {
    download: requestDownload,
    pmc,
    followUps: tables.length === 1 ? (payload) => followUps(payload, query, { squall, peaks: hasPeaks() }) : null,
    use: goOn,
  })));
}

// The brain viewer page of a reply, in the brain tab, with the buttons of the images of the tables.
function showViewer(page) {
  const viewer = Object.assign(document.createElement("iframe"), { sandbox: "allow-scripts", srcdoc: themed(page, theme()), title: "NeuroLang brain viewer" });
  show("brain", ...[imageButtons(lastTables, requestDownload), viewer].filter(Boolean));
  setViewerStale(false);
}

// The viewer shown is of an earlier result once a query gives a new one, or
// there is none yet: selecting its tab then views the last result (.view).
let viewerStale = false;
function setViewerStale(stale) {
  viewerStale = stale;
  $("tab-brain").title = stale ? "View the last result in the brain" : "";
}
// A result can always be tried in the brain: its tab is there, and views it.
function markViewerStale() {
  $("tab-brain").hidden = false;
  setViewerStale(true);
}

// No viewer: that of an earlier result is not to be taken for the last one's.
function dropViewer() {
  $("brain").replaceChildren();
  $("tab-brain").hidden = true;
  setViewerStale(false);
  if (results.dataset.tab === "brain") selectTab("table");
}

// Open the tab *tab* of the results: the brain tab of an earlier result views the last one.
function openTab(tab) {
  if (tab === "brain" && viewerStale && !$("run").disabled) return run(".view");
  selectTab(tab);
}

let graphUrl = null; // the address of the page of the graph shown

// The pages in frames, the brain viewer and the graph: they are as tall as
// they say, so that they have no scrollbars within the page, and have its theme.
const resultFrames = () => [...document.querySelectorAll("#brain iframe, #graph iframe")];
const theme = () => document.documentElement.dataset.theme;
window.addEventListener("message", (event) => {
  const asked = frameHeight(event, resultFrames());
  if (!asked) return;
  const { frame, height } = asked;
  frame.style.height = `${height + frame.offsetHeight - frame.clientHeight}px`; // its border too
});

// The graph of the query a reply ran, in the graph tab, shown when the reply
// has no table (.graph); a query that has no graph hides that of the last one.
function showGraph({ graph, tables }) {
  const shown = tables?.length > 0;
  if (!graph) {
    if (shown) $("tab-graph").hidden = true;
    return;
  }
  const page = Object.assign(document.createElement("iframe"), { sandbox: "allow-scripts", srcdoc: themed(graph, theme()), title: "Graph of the query" });
  // The graph is often wider than the page: it can open in a tab of its own.
  if (graphUrl) URL.revokeObjectURL(graphUrl);
  graphUrl = URL.createObjectURL(new Blob([themed(graph, theme())], { type: "text/html" }));
  const open = Object.assign(document.createElement("a"), { href: graphUrl, target: "_blank", rel: "noopener", textContent: "Open the graph in a new tab" });
  const nodes = [Object.assign(document.createElement("p"), { className: "note" }), page];
  nodes[0].append(open);
  if (shown) {
    $("graph").replaceChildren(...nodes);
    $("tab-graph").hidden = false;
  } else show("graph", ...nodes);
}

// The log of a query, under its last line, which folds it: folded when the
// query gave a table, open otherwise.
function showLog(log, fold) {
  const last = logSummary(log.textContent);
  out.classList.toggle("quiet", Boolean(last));
  if (!last) return out.replaceChildren(log);
  const details = Object.assign(document.createElement("details"), { open: !fold });
  details.append(Object.assign(document.createElement("summary"), { textContent: last }), log);
  out.replaceChildren(details);
}

// The query that goes on from a result takes the place of what is in the box, the caret at its end.
// Whether the engine has the peaks the studies report, from which a result of studies is mapped.
const hasPeaks = () => guide.predicates.some((predicate) => predicate.name === "peak_reported");
const pmc = new PmcLookup(); // the PubMed Central records of the studies shown, from Europe PMC
let lastRun = ""; // what was last submitted
// The guide the page opened for a first visit, to close at the first run so
// that the result is not pushed below it; one opened by hand stays open.
let guideForStart = false;
const session = new Session();
let replaying = null; // the statements being run again, silently, and how many failed
let rerun = null; // what runs again once the rules of a session the server ended are back
function goOn(text) {
  putQuery(text);
  hideSummary();
  input.focus();
  input.setSelectionRange(editor.value.length, editor.value.length);
}

// Run again, one after the other, the statements that give back a session's rules.
function replay({ entries, box, mode, then }, { clear = false } = {}) {
  replaying = { queue: replayTexts(entries, mode, { clear }), total: 0, failed: 0, box, mode, then };
  replaying.total = replaying.queue.length;
  setBusy(true);
  replayNext();
}

function replayNext() {
  const text = replaying.queue.shift();
  if (text === undefined) {
    const { failed, box, mode, then } = replaying;
    replaying = null;
    $("mode").value = mode;
    editor.refresh();
    if (box) editor.value = box;
    setStatus(failed ? `Ready. ${failed} statement${failed === 1 ? "" : "s"} of the session did not run again.` : "Ready: the rules of the session are back.");
    setBusy(false);
    if (then) run((rerun = then)); // what the session's end interrupted
    return;
  }
  setStatus(`Restoring the rules of the session (${replaying.total - replaying.queue.length} of ${replaying.total})…`, "busy");
  worker.postMessage({ text });
}

function replayStep(data) {
  if (data.failed) replaying.failed += 1;
  replayNext();
}

function run(text, { remember = false } = {}) {
  text = text.trim();
  if (!text) return;
  if (remember) {
    history.add(text);
    showHistory();
  }
  setBusy(true);
  $("stop").hidden = false;
  lastRun = text;
  if (guideForStart) guideForStart = $("guide").open = false;
  setStatus("Running…", "busy");
  if (!text.startsWith(".ask") && !text.startsWith(".llm")) {
    hideSummary();
    showNoAnswer(null); // a query of one's own: the notice is of the past
  }
  hideRating();
  worker.postMessage({ text });
}

// The theme: light, dark or the computer's, which the page's head applied.
const THEME = "neurolang-repl-theme";
const system = matchMedia("(prefers-color-scheme: light)");
const themeButtons = [...document.querySelectorAll("#theme button")];
function applyTheme(choice) {
  const light = choice === "light" || (choice === "system" && system.matches);
  document.documentElement.dataset.theme = light ? "light" : "dark";
  document.documentElement.dataset.themeChoice = choice;
  themeButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.theme === choice));
  for (const frame of resultFrames()) frame.contentWindow?.postMessage(themeMessage(theme()), "*");
}
const chosen = () => store("localStorage")?.getItem(THEME) || "system";
for (const button of themeButtons) {
  button.addEventListener("click", () => {
    store("localStorage")?.setItem(THEME, button.dataset.theme);
    applyTheme(button.dataset.theme);
  });
}
system.addEventListener("change", () => applyTheme(chosen()));
applyTheme(chosen());

// The question box is narrow on a phone: its example is left out there.
const askPlaceholder = question.placeholder;
const narrow = matchMedia("(max-width: 40rem)");
// The panes of the page side by side (index.html), or one under the other.
const stacked = matchMedia("(max-width: 71.99rem)");
const placeAsk = () => (question.placeholder = narrow.matches ? "Ask in English…" : askPlaceholder);
narrow.addEventListener("change", placeAsk);
placeAsk();

const version = $("version");
$("code").addEventListener("submit", (event) => {
  event.preventDefault();
  run(input.value, { remember: true });
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    run(input.value, { remember: true });
  } else if (event.altKey && event.shiftKey && event.code === "KeyF") {
    event.preventDefault();
    formatQuery();
  }
});
// A dot-command is a line: the lines of the question are joined.
const ask = () => {
  const text = question.value.trim().replace(/\s*\n\s*/g, " ");
  if (text) run(`.ask ${text}`);
};
$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});
// The example question is selected when the box is entered: typing
// replaces it, Ask asks it.
question.addEventListener("focus", () => {
  if (suggested && question.value === suggested) question.select();
});
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    ask();
  }
});
$("share").addEventListener("click", async (event) => {
  const url = `${location.origin}${location.pathname}${location.search}#${encodeShare(editor.value, $("mode").value)}`;
  window.history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    event.target.textContent = "Copied";
    setTimeout(() => { event.target.textContent = "Copy link"; }, 1200);
  } catch {
    window.prompt("Copy this link", url); // no clipboard here, e.g. an unsecured address
  }
});
$("key").addEventListener("click", () => {
  const details = $("llm-details");
  details.open = !details.open;
  if (details.open) llmForm.elements.key.focus();
});
$("mode").addEventListener("change", (event) => {
  editor.refresh();
  showGuide();
  run(`.mode ${event.target.value}`);
});
$("view").addEventListener("click", () => run(".view"));
$("format").addEventListener("click", () => {
  formatQuery();
  $("format").closest("details").open = false;
});
$("explain").addEventListener("click", () => run(".explain"));
history_.addEventListener("change", () => {
  if (history_.value) {
    editor.value = history_.value;
    hideSummary();
  }
  history_.value = "";
  input.focus();
});
for (const tab of TABS) $(`tab-${tab}`).addEventListener("click", () => openTab(tab));
$("tabs").addEventListener("keydown", (event) => {
  const shown = TABS.filter((tab) => !$(`tab-${tab}`).hidden);
  const next = tabAfter(shown, results.dataset.tab, event.key);
  if (!next) return;
  event.preventDefault();
  openTab(next);
  $(`tab-${next}`).focus();
});
$("retry").addEventListener("click", start);
llmForm.elements.provider.addEventListener("change", showProvider);
llmForm.addEventListener("submit", (event) => {
  event.preventDefault();
  llmSettings.save(choices());
  useKey();
  $("llm-details").open = false;
});
$("llm-forget").addEventListener("click", () => {
  llmSettings.forget();
  Object.assign(llmForm.elements.key, { value: "" });
  llmForm.elements.remember.checked = false;
  worker.postMessage({ type: "llm", key: "" });
  pageKey = false;
  assistantReady = false;
  if (suggested && question.value === suggested) question.value = "";
  suggested = "";
  Object.assign($("ask-run"), { title: "Add an API key to use the assistant" }).classList.add("ghost");
  $("llm-error").hidden = true;
  $("llm-summary").textContent = "bring your own API key";
  $("key").textContent = "API key";
  $("ask-note").hidden = false;
  $("explain").hidden = true;
});
$("restart").addEventListener("click", () => start());
// The next user of this browser does not have this one's key.
$("logout").addEventListener("click", () => llmSettings.forget());
$("stop").addEventListener("click", () => {
  const entries = session.entries.map((entry) => ({ ...entry }));
  worker.terminate();
  setStatus("Stopped. The engine restarts, with the rules of the session (its data are kept by the browser)…", "busy");
  start({ restore: { entries, box: editor.value, mode: $("mode").value } });
});
$("session-save").addEventListener("click", () => {
  saveFile({ name: "neurolang-session.json", mime: "application/json", bytes: session.serialise(editor.value, $("mode").value) });
});
$("session-open").addEventListener("click", () => $("session-file").click());
$("session-file").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  event.target.value = "";
  if (!file) return;
  try {
    const opened = parseSession(await file.text());
    session.replace(opened.entries);
    $("mode").value = opened.mode;
    replay(opened, { clear: true });
  } catch (error) {
    out.classList.remove("quiet");
    out.replaceChildren(Object.assign(document.createElement("pre"), { textContent: error.message }));
  }
});
version.addEventListener("submit", (event) => {
  event.preventDefault();
  const params = new URLSearchParams();
  const { repo, ref } = Object.fromEntries(new FormData(version));
  const engine = new URLSearchParams(location.search).get("engine");
  if (engine) params.set("engine", engine);
  if (ref.trim()) {
    params.set("ref", ref.trim());
    if (repo.trim() !== DEFAULT_REPO) params.set("repo", repo.trim());
  }
  window.history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
  start();
});
engineSelect.addEventListener("change", () => {
  const params = withEngine(new URLSearchParams(location.search), engineSelect.value, bundle);
  window.history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
  start();
});
showHistory();
showAsked();
loadEngines().then(() => start());
