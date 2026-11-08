// The page of the in-browser REPL: a box for the AI assistant, a box for
// the code, and the result as a table or in the brain viewer. It shows what
// worker.js answers.
import { History } from "./history.js";
import { LlmSettings } from "./llm_settings.js";
import { imageButtons, renderTable } from "./table_view.js";
import { AgentLog, renderAgentLog, tickWaiting } from "./agent_view.js";
import { DOCS_URL, PAGES, topicAt } from "./docs.js";
import { Editor } from "./editor.js";
import { renderCommands, renderExamples, renderPredicates } from "./guide.js";
import { highlightNodes, loadWords } from "./highlight.js";
import { decodeShare, encodeShare } from "./share.js";
import { TEMPLATES, availableTemplates, buildQuery } from "./builder.js";
import { followUps } from "./follow_up.js";
import { logSummary } from "./log_view.js";
import { explainStartError } from "./start_error.js";
import { Session, parseSession, replayTexts } from "./session.js";
import { PmcLookup } from "./study_links.js";
import { tabAfter } from "./tabs.js";

const $ = (id) => document.getElementById(id);
const [statusBox, out, input, question, history_] = ["status", "out", "input", "question", "history"].map($);
const llmForm = $("llm");
const results = $("results");
const DEFAULT_REPO = "NeuroLang/NeuroLang";
const busyControls = [input, question, $("run"), $("ask-run"), $("view"), $("explain"), $("session-open")];
// The code box colours its text as the language of the mode select.
const editor = new Editor(input, $("mirror"), () => $("mode").value === "squall");
loadWords().then(() => editor.refresh()); // the words of the grammars, from highlight.json

const store = (kind) => {
  try {
    return window[kind];
  } catch {
    return null; // refused, e.g. in a private window
  }
};
const history = new History(store("localStorage"));
const llmSettings = new LlmSettings(store("localStorage"), store("sessionStorage"));
let worker = null;
let providers = []; // those of the AI assistant, from the worker

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
function showSummary({ valid, explanation, back_translation: reads, verified, verification }) {
  if (!valid || !(explanation || reads)) return hideSummary();
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

// The examples of the engine and its predicates, which put their text in the box.
let guide = { examples: {}, predicates: [], commands: [] };
function showGuide() {
  if (!$("guide").open) return;
  const squall = $("mode").value === "squall";
  renderExamples($("guide-examples"), guide.examples[squall ? "squall" : "datalog"], (query) => {
    editor.value = query;
    hideSummary();
    input.focus();
  });
  renderPredicates($("guide-relations"), guide.predicates, $("guide-filter").value, (text) => editor.insert(text), squall, 200, (query) => {
    editor.value = query;
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
    editor.value = query;
    hideSummary();
    input.focus();
    $("builder-error").textContent = "";
  } catch (error) {
    $("builder-error").textContent = error.message;
  }
});
$("guide").addEventListener("toggle", showGuide);
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
  for (const name of ["table", "brain"]) {
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

function useKey() {
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
  if (list.length && saved.key) useKey();
}

// The version to run, from the page's address: ?repo=OWNER/NAME&ref=BRANCH.
function start({ restore = null } = {}) {
  if (!restore) session.replace([]); // a new engine has none of the rules
  worker?.terminate();
  const params = new URLSearchParams(location.search);
  const [repo, ref] = [params.get("repo") || DEFAULT_REPO, params.get("ref") || ""];
  Object.assign(version.elements.repo, { value: repo });
  Object.assign(version.elements.ref, { value: ref });
  $("version-summary").textContent = ref ? `${repo} at ${ref}` : "the build of this page";
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
  worker = new Worker("worker.js", { type: "module" });
  worker.postMessage({ type: "start", repo, ref, full: params.get("engine") === "full" });
  worker.onerror = (event) => fail(event.message || "the browser ended its worker, perhaps out of memory");
  worker.onmessage = ({ data }) => {
    if (data.type === "reply" && replaying) replayStep(data);
    else workerMessages[data.type]?.(data, restore);
  };
}

// What the page does with each kind of message of the worker; *restore* is what a stop gave to give back.
const workerMessages = {
  status: (data) => setStatus(data.message, "busy"),
  error: (data) => fail(data.message),
  ready: (data, restore) => {
    setStatus("Ready.");
    $("boot").hidden = true;
    $("banner").replaceChildren(html(data.banner));
    setProviders(data.providers ?? []);
    canRate = Boolean(data.canRate);
    guide = { examples: data.examples ?? {}, predicates: data.predicates ?? [], commands: data.commands ?? [] };
    if (!input.value && !decodeShare(location.hash)) guideForStart = $("guide").open = true; // examples to start from
    showGuide();
    if (restore) return replay(restore);
    setBusy(false);
    restoreShared();
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
      hideSummary();
      hideRating();
    }
    if (data.event === "done") {
      showSummary(data);
      showRating();
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
  reply: (data) => {
    setStatus(data.failed ? "Ready (the last statement failed)." : "Ready.");
    if (!data.failed) session.add(lastRun, $("mode").value);
    showLog(html(data.html), !data.failed && Boolean(data.tables?.length));
    if (data.ratable) showRating();
    applyReply(data);
    // A failure without a table hides the last result, not to be taken for its own.
    if (data.failed && !data.tables?.length) results.classList.remove("has");
    if (data.tables?.length) showTables(data);
    if (data.viewers?.length) showViewer(data.viewers.at(-1));
    setBusy(false);
  },
};

// What a reply changes in the page besides its output: the predicates, the language, and the box.
function applyReply(data) {
  if (data.predicates) guide.predicates = data.predicates;
  if (data.mode) $("mode").value = data.mode; // the assistant may have switched it
  showGuide();
  if (data.prompt) editor.value = data.prompt; // what the assistant wrote
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
  const viewer = Object.assign(document.createElement("iframe"), { sandbox: "allow-scripts", srcdoc: page, title: "NeuroLang brain viewer" });
  show("brain", ...[imageButtons(lastTables, requestDownload), viewer].filter(Boolean));
}

// The log of a query that gave a table is folded to its last line, which says how long it took.
function showLog(log, fold) {
  const last = fold && logSummary(log.textContent);
  out.classList.toggle("quiet", Boolean(last));
  if (!last) return out.replaceChildren(log);
  const details = document.createElement("details");
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
function goOn(text) {
  editor.value = text;
  hideSummary();
  input.focus();
  input.setSelectionRange(text.length, text.length);
}

// Run again, one after the other, the statements that give back a session's rules.
function replay({ entries, box, mode }, { clear = false } = {}) {
  replaying = { queue: replayTexts(entries, mode, { clear }), total: 0, failed: 0, box, mode };
  replaying.total = replaying.queue.length;
  setBusy(true);
  replayNext();
}

function replayNext() {
  const text = replaying.queue.shift();
  if (text === undefined) {
    const { failed, box, mode } = replaying;
    replaying = null;
    $("mode").value = mode;
    editor.refresh();
    if (box) editor.value = box;
    setStatus(failed ? `Ready. ${failed} statement${failed === 1 ? "" : "s"} of the session did not run again.` : "Ready: the rules of the session are back.");
    setBusy(false);
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
  if (!text.startsWith(".ask") && !text.startsWith(".llm")) hideSummary();
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
  }
});
$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  if (question.value.trim()) run(`.ask ${question.value}`);
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
$("explain").addEventListener("click", () => run(".explain"));
history_.addEventListener("change", () => {
  if (history_.value) {
    editor.value = history_.value;
    hideSummary();
  }
  history_.value = "";
  input.focus();
});
for (const tab of ["table", "brain"]) $(`tab-${tab}`).addEventListener("click", () => selectTab(tab));
$("tabs").addEventListener("keydown", (event) => {
  const shown = ["table", "brain"].filter((tab) => !$(`tab-${tab}`).hidden);
  const next = tabAfter(shown, results.dataset.tab, event.key);
  if (!next) return;
  event.preventDefault();
  selectTab(next);
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
  Object.assign($("ask-run"), { title: "Add an API key to use the assistant" }).classList.add("ghost");
  $("llm-error").hidden = true;
  $("llm-summary").textContent = "bring your own API key";
  $("key").textContent = "API key";
  $("ask-note").hidden = false;
  $("explain").hidden = true;
});
$("restart").addEventListener("click", () => start());
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
  if (ref.trim()) {
    params.set("ref", ref.trim());
    if (repo.trim() !== DEFAULT_REPO) params.set("repo", repo.trim());
  }
  window.history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
  start();
});
showHistory();
start();
